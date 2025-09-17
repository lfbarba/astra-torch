"""Astra laminography utilities.

This module provides GPU-friendly PyTorch functions for laminography reconstruction
using ASTRA toolbox with parallel beam geometry.

Public functions
----------------
fbp_reconstruction_masked(...):
    Perform FBP (Filtered Back Projection) reconstruction on (possibly masked) subset of views for laminography.

gd_reconstruction_masked(...):
    Gradient descent reconstruction using differentiable laminography projector.

build_lamino_projector(...):
    Factory that returns a torch.autograd.Function for laminography projection.

Notes
-----
* Requires the ASTRA toolbox compiled with CUDA support.
* All tensors are float32 and (by default) CUDA if available.
* Projection tensor layout is (V, R, C) following the convention (views, detector_rows, detector_cols).
* Laminography geometry uses lamino_angle (tilt of detector/source) and rotation angles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional, Dict, Any
import numpy as np
import torch
from tqdm import tqdm

try:
    import astra  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("astra toolbox is required for chip.astra.lamino module") from e


@dataclass
class LaminoAcquisition:
    """Laminography acquisition parameters."""
    proj_rows: int
    proj_cols: int
    angles_deg: np.ndarray  # rotation angles in degrees
    lamino_angle_deg: float  # laminography angle (detector/source tilt)
    tilt_angle_deg: float   # additional tilt parameter
    vol_shape: Tuple[int, int, int]
    # voxel_size_mm: float
    # voxel_per_mm: int = 1


# ---------------------------------------------------------------------------
# Geometry setup helpers
# ---------------------------------------------------------------------------

def _make_volume_geom(vol_shape: Tuple[int, int, int], voxel_size_mm: float=1):
    """Create ASTRA volume geometry with proper scaling and centering.
    
    Parameters
    ----------
    vol_shape : (nz, ny, nx)
        Volume shape in PyTorch convention (depth, height, width)
    voxel_size_mm : float
        Voxel size in mm
        
    Returns
    -------
    vol_geom : dict
        ASTRA volume geometry
    """
    nz, ny, nx = vol_shape  # PyTorch convention: (depth, height, width)
    
    # ASTRA expects (nx, ny, nz) convention
    vol_geom = astra.create_vol_geom(nx, ny, nz)
    
    # Set the volume to be centered around the origin
    # ASTRA default is [0, nx] x [0, ny] x [0, nz], we want [-nx/2, nx/2] etc.
    vol_geom['option']['WindowMinX'] = -nx * voxel_size_mm / 2.0
    vol_geom['option']['WindowMaxX'] = nx * voxel_size_mm / 2.0
    vol_geom['option']['WindowMinY'] = -ny * voxel_size_mm / 2.0
    vol_geom['option']['WindowMaxY'] = ny * voxel_size_mm / 2.0
    vol_geom['option']['WindowMinZ'] = -nz * voxel_size_mm / 2.0
    vol_geom['option']['WindowMaxZ'] = nz * voxel_size_mm / 2.0
    
    return vol_geom


def _create_lamino_geometry(
    vol_shape: Tuple[int, int, int],
    det_shape: Tuple[int, int],  # (rows, cols)
    angles_deg: np.ndarray,
    lamino_angle_deg: float,
    tilt_angle_deg: float = 0.0,
    voxel_size_mm: float = 1.0,
    det_spacing_mm: float = 1.0,
) -> Tuple[dict, dict]:
    """Create ASTRA geometries for laminography.
    
    Parameters
    ----------
    vol_shape : (nx, ny, nz)
        Volume dimensions
    det_shape : (rows, cols) 
        Detector dimensions
    angles_deg : array
        Rotation angles in degrees (0-360)
    lamino_angle_deg : float
        Laminography angle (detector/source tilt relative to sample)
    tilt_angle_deg : float
        Additional tilt parameter
    voxel_size_mm : float
        Voxel size in mm
    det_spacing_mm : float
        Detector pixel spacing in mm
    source_detector_distance_mm : float
        IGNORED - Not used in parallel beam geometry (kept for API compatibility)
    source_origin_distance_mm : float
        IGNORED - Not used in parallel beam geometry (kept for API compatibility)
        
    Returns
    -------
    vol_geom : dict
        ASTRA volume geometry
    proj_geom : dict
        ASTRA projection geometry
        
    Notes
    -----
    This function implements TRUE parallel beam geometry where all rays are parallel.
    The source_detector_distance_mm and source_origin_distance_mm parameters are 
    ignored as they have no meaning in parallel beam geometry.
    """
    vol_geom = _make_volume_geom(vol_shape, voxel_size_mm)
    
    det_rows, det_cols = det_shape
    n_angles = len(angles_deg)
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles_deg)
    lamino_rad = np.deg2rad(lamino_angle_deg)
    tilt_rad = np.deg2rad(tilt_angle_deg)
    
    # Initialize geometry vectors for laminography
    vectors = np.zeros((n_angles, 12))
    
    # No detector center offset - let ASTRA handle detector centering
    # The detector should be centered by default
    
    for i, theta in enumerate(angles_rad):
        # For TRUE parallel beam geometry, ray direction is determined solely by geometry angles
        # Ray direction: all rays are parallel, coming from the tilted direction
        ray_x = np.cos(theta) * np.cos(lamino_rad)
        ray_y = np.sin(theta) * np.cos(lamino_rad)
        ray_z = np.sin(lamino_rad)
        
        # Detector position can be arbitrary - place it at a convenient location
        # We'll place it at a fixed distance from origin, perpendicular to rays
        det_distance = 100.0  # Arbitrary fixed distance (doesn't affect parallel beam)
        det_x = -det_distance * ray_x
        det_y = -det_distance * ray_y
        det_z = -det_distance * ray_z
        
        # Detector u-direction (horizontal, tangent to rotation)
        u_x = -np.sin(theta) * det_spacing_mm
        u_y = np.cos(theta) * det_spacing_mm
        u_z = 0.0
        
        # Detector v-direction (vertical, adjusted for lamino angle)
        # In laminography, the v-direction is tilted by the lamino angle
        v_x = -np.cos(theta) * np.sin(lamino_rad) * det_spacing_mm
        v_y = -np.sin(theta) * np.sin(lamino_rad) * det_spacing_mm  
        v_z = np.cos(lamino_rad) * det_spacing_mm
        
        # Apply tilt angle if needed (rotation around detector u-axis)
        if tilt_rad != 0:
            # Rotate v-direction by tilt angle around u-axis
            v_y_new = v_y * np.cos(tilt_rad) - v_z * np.sin(tilt_rad)
            v_z_new = v_y * np.sin(tilt_rad) + v_z * np.cos(tilt_rad)
            v_y, v_z = v_y_new, v_z_new
        
        # For parallel beam geometry: [ray_x, ray_y, ray_z, det_x, det_y, det_z, u_x, u_y, u_z, v_x, v_y, v_z]
        vectors[i, :] = [ray_x, ray_y, ray_z,      # ray direction (normalized)
                        det_x, det_y, det_z,       # detector center
                        u_x, u_y, u_z,            # detector u-direction
                        v_x, v_y, v_z]            # detector v-direction
    
    proj_geom = astra.create_proj_geom('parallel3d_vec', det_rows, det_cols, vectors)
    
    return vol_geom, proj_geom


# ---------------------------------------------------------------------------
# Parallel beam reconstruction with optional masking  
# ---------------------------------------------------------------------------

def fbp_reconstruction_masked(
    projs_vrc: torch.Tensor,
    angles_deg: np.ndarray,
    lamino_angle_deg: float,
    mask: Optional[Sequence[Any]] = None,
    tilt_angle_deg: float = 0.0,
    voxel_per_mm: int = 1,
    vol_shape: Optional[Tuple[int, int, int]] = None,
    det_spacing_mm: Optional[float] = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run parallel beam reconstruction (FBP) on a subset of laminography views.

    Parameters
    ----------
    projs_vrc : (V,R,C) tensor 
        Laminography projections (CPU or CUDA)
    angles_deg : (V,) numpy array
        Rotation angles in degrees (0-360)
    lamino_angle_deg : float
        Laminography angle (detector/source tilt)
    mask : optional boolean sequence length V
        If None, all views are used
    tilt_angle_deg : float
        Additional tilt parameter
    voxel_per_mm : int
        Resolution parameter 
    vol_shape : optional explicit (nx,ny,nz)
        Volume dimensions
    det_spacing_mm : float
        Detector pixel spacing in mm
    device : torch.device
        Compute device
        Detector pixel spacing
    source_detector_distance_mm : float
        Not used for parallel beam (kept for compatibility)
    source_origin_distance_mm : float
        Not used for parallel beam (kept for compatibility)
    device : torch.device, optional
        Target device

    Returns
    -------
    vol : (nx,ny,nz) float32 tensor on device
    """
    if device is None:
        device = projs_vrc.device if isinstance(projs_vrc, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mask is not None:
        # Support torch.Tensor or numpy / sequence masks (boolean or index)
        if isinstance(mask, torch.Tensor):
            if mask.dtype == torch.bool:
                if mask.numel() != len(angles_deg):
                    raise ValueError("boolean mask length must match number of views")
                mask_cpu = mask.detach().cpu().numpy()
                sel_angles = angles_deg[mask_cpu]
                sel_projs = projs_vrc[mask]
            else:
                idx = mask.detach().cpu().numpy()
                sel_angles = angles_deg[idx]
                sel_projs = projs_vrc[idx]
        else:
            mask_arr = np.asarray(mask)
            if mask_arr.dtype == bool:
                if mask_arr.shape[0] != len(angles_deg):
                    raise ValueError("boolean mask length must match number of views")
                sel_angles = angles_deg[mask_arr]
                sel_projs = projs_vrc[mask_arr]
            else:
                sel_angles = angles_deg[mask_arr]
                sel_projs = projs_vrc[mask_arr]
    else:
        sel_angles = angles_deg
        sel_projs = projs_vrc

    # Ensure CPU numpy for ASTRA (ASTRA works with numpy arrays)
    sel_projs_np = sel_projs.detach().cpu().numpy()  # (V,R,C)
    # Convert to ASTRA (R,V,C)
    sino_rvc = np.transpose(sel_projs_np, (1, 0, 2)).copy()
    det_rows, n_views, det_cols = sino_rvc.shape

    if vol_shape is None:
        raise ValueError("vol_shape must be specified for FBP reconstruction")
    
    voxel_size_mm = 1.0 / voxel_per_mm

    vol_rec = np.zeros(vol_shape, dtype=np.float32)

    # Create laminography geometry
    vol_geom, proj_geom = _create_lamino_geometry(
        vol_shape=vol_shape,
        det_shape=(det_rows, det_cols),
        angles_deg=sel_angles,
        lamino_angle_deg=lamino_angle_deg,
        tilt_angle_deg=tilt_angle_deg,
        voxel_size_mm=voxel_size_mm,
        det_spacing_mm=det_spacing_mm,
    )

    vol_id = astra.data3d.link('-vol', vol_geom, vol_rec)
    proj_id = astra.data3d.link('-sino', proj_geom, sino_rvc)

    # Use SIRT3D_CUDA for parallel beam reconstruction (laminography)
    # SIRT (Simultaneous Iterative Reconstruction Technique) works well with parallel beam
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = vol_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 50)  # Run 50 SIRT iterations
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

    vol_t = torch.from_numpy(vol_rec).to(torch.float32).to(device)
    return vol_t


# ---------------------------------------------------------------------------
# Gradient descent reconstruction
# ---------------------------------------------------------------------------

def gd_reconstruction_masked(
    projs_vrc: torch.Tensor,
    angles_deg: np.ndarray,
    lamino_angle_deg: float,
    mask: Optional[Sequence[Any]] = None,
    tilt_angle_deg: float = 0.0,
    voxel_per_mm: int = 1,
    vol_shape: Optional[Tuple[int, int, int]] = None,
    det_spacing_mm: float = 1.0,
    device: Optional[torch.device] = None,
    # Optimization hyper-parameters
    max_epochs: int | Sequence[int] = 30,
    batch_size: int | Sequence[int] = 20,
    lr: float | Sequence[float] = 1e-3,
    clamp_min: float = 0.0,
    # Optimizer settings
    optimizer_type: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    # Initialization
    vol_init: Optional[torch.Tensor] = None,
    # Control verbosity
    verbose: bool = False,
    # Restart optimizer
    restart_optimizer: bool = True, # Restart optimizer for each segment
) -> torch.Tensor:
    """Gradient-descent laminography reconstruction using differentiable projector.

    This function performs iterative reconstruction by minimizing the mean squared error
    between measured projections and forward projections of the volume using mini-batches.

    Parameters
    ----------
    projs_vrc : (V,R,C) torch.Tensor
        Laminography projections (full set of views) on any device
    angles_deg : (V,) np.ndarray
        Rotation angles in degrees
    lamino_angle_deg : float
        Laminography angle (detector/source tilt)
    mask : sequence or boolean np.ndarray, optional
        Subset of views to use. If None, all views are used
    tilt_angle_deg : float
        Additional tilt parameter
    voxel_per_mm : int
        Resolution parameter
    vol_shape : (nx,ny,nz), optional
        Explicit volume shape
    det_spacing_mm : float
        Detector pixel spacing
    source_detector_distance_mm : float
        Source to detector distance
    source_origin_distance_mm : float
        Source to rotation axis distance
    device : torch.device, optional
        Target device for reconstruction
    max_epochs : int or Sequence[int]
        Number of optimization epochs (can be staged)
    batch_size : int or Sequence[int]
        Number of views per mini-batch
    lr : float or Sequence[float]
        Learning rate(s) for optimizer
    clamp_min : float
        Minimum value to clamp volume after each step
    optimizer_type : str
        Type of optimizer ("adam" or "sgd")
    momentum : float
        Momentum for SGD optimizer
    weight_decay : float
        Weight decay (L2 regularization) parameter
    vol_init : torch.Tensor, optional
        Initial volume (if None, uses zero initialization)
    verbose : bool
        Show progress bars
    restart_optimizer : bool
        If True, restart optimizer for each segment (useful for staged learning rates)

    Returns
    -------
    vol : (nx,ny,nz) torch.Tensor
        Final reconstructed volume on the specified device
    """
    if device is None:
        device = projs_vrc.device if isinstance(projs_vrc, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select masked subset of projections & angles (reuse logic from FDK function)
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            if mask.dtype == torch.bool:
                if mask.numel() != len(angles_deg):
                    raise ValueError("boolean mask length must match number of views")
                mask_cpu = mask.detach().cpu().numpy()
                sel_angles = angles_deg[mask_cpu]
                sel_projs = projs_vrc[mask]
            else:
                idx = mask.detach().cpu().numpy()
                sel_angles = angles_deg[idx]
                sel_projs = projs_vrc[idx]
        else:
            mask_arr = np.asarray(mask)
            if mask_arr.dtype == bool:
                if mask_arr.shape[0] != len(angles_deg):
                    raise ValueError("boolean mask length must match number of views")
                sel_angles = angles_deg[mask_arr]
                sel_projs = projs_vrc[mask_arr]
            else:
                sel_angles = angles_deg[mask_arr]
                sel_projs = projs_vrc[mask_arr]
    else:
        sel_angles = angles_deg
        sel_projs = projs_vrc

    V_sel = len(sel_angles)
    det_rows, det_cols = sel_projs.shape[1], sel_projs.shape[2]

    if vol_shape is None:
        raise ValueError("vol_shape must be specified for gradient descent reconstruction")
    voxel_size_mm = 1.0 / voxel_per_mm

    # Initialization
    if vol_init is not None:
        if vol_init.ndim == 5:
            vol0 = vol_init[0, 0]
        elif vol_init.ndim == 4:  # (1,nx,ny,nz)
            vol0 = vol_init[0]
        elif vol_init.ndim == 3:
            vol0 = vol_init
        else:
            raise ValueError("Unsupported vol_init shape")
        # If vol_shape not supplied, infer from vol_init
        if vol_shape is None:
            vol_shape = tuple(vol0.shape)
        if tuple(vol0.shape) != vol_shape:
            raise ValueError(f"vol_init shape {tuple(vol0.shape)} does not match target vol_shape {vol_shape}")
        recon = vol0.clone().to(device)
    else:
        # Default: zero initialization
        recon = torch.zeros(vol_shape, dtype=torch.float32, device=device)

    recon = recon.unsqueeze(0).unsqueeze(0)  # (1,1,nx,ny,nz)
    recon.requires_grad_(True)

    # Measurements tensor shaped (1,V,R,C) for convenience
    meas_full = sel_projs.to(device).unsqueeze(0)

    # Build learning rate / epoch schedule
    if isinstance(batch_size, int):
        batch_list = [int(batch_size)]
    else:
        batch_list = [int(b) for b in batch_size]
        if len(batch_list) == 0:
            raise ValueError("batch_size sequence must be non-empty")
        
    if isinstance(max_epochs, int):
        epochs_list = [int(max_epochs)]
    else:
        epochs_list = [int(e) for e in max_epochs]
        if len(epochs_list) == 0:
            raise ValueError("max_epochs sequence must be non-empty")
    if isinstance(lr, (float, int)):
        lr_list = [float(lr)] * len(epochs_list)
    else:
        lr_list = [float(x) for x in lr]
        if len(lr_list) == 0:
            raise ValueError("lr sequence must be non-empty")
        if len(lr_list) != len(epochs_list):
            raise ValueError("lr and max_epochs sequences must have same length")

    total_epochs = sum(epochs_list)

    # Outer progress bar
    if verbose:
        outer_pbar = tqdm(total=total_epochs, leave=True, disable=not verbose)
    epoch_global = 0

    if verbose:
        print(lr_list, epochs_list)
    optimizer = None

    for seg_idx, (lr_i, seg_epochs, batch_size) in enumerate(zip(lr_list, epochs_list, batch_list)):
        # Create optimizer
        if optimizer is None or restart_optimizer:
            if optimizer_type.lower() == "adam":
                optimizer = torch.optim.Adam([recon], lr=lr_i, weight_decay=weight_decay)
            elif optimizer_type.lower() == "sgd":
                optimizer = torch.optim.SGD([recon], lr=lr_i, momentum=momentum, weight_decay=weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported: 'adam', 'sgd'")
        
        for local_epoch in range(seg_epochs):
            perm = torch.randperm(V_sel, device=device)
            iters = math.ceil(V_sel / batch_size)
            
            epoch_loss = 0.0
            for it in range(iters):
                sel = perm[it * batch_size:(it + 1) * batch_size]
                angles_batch = sel_angles[sel.cpu().numpy()]
                proj_layer = build_lamino_projector(
                    vol_shape=vol_shape,
                    det_shape=(det_rows, det_cols),
                    angles_deg=angles_batch,
                    lamino_angle_deg=lamino_angle_deg,
                    tilt_angle_deg=tilt_angle_deg,
                    voxel_size_mm=voxel_size_mm,
                    det_spacing_mm=det_spacing_mm,
                    device=device
                )
                meas_batch = meas_full[:, sel]  # (1,k,R,C)
                optimizer.zero_grad(set_to_none=True)
                pred = proj_layer(recon)  # (1,k,R,C)
                loss = torch.mean((pred - meas_batch) ** 2)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    if clamp_min is not None:
                        recon.clamp_(min=clamp_min)
                epoch_loss += loss.item() * sel.numel()
                if verbose:
                    denom = (it * batch_size + sel.numel())
                    inner_postfix = {
                        'loss': f"{loss.item():.3e}",
                        'avg': f"{epoch_loss/denom:.3e}",
                        'k': sel.numel(),
                        'lr': f"{lr_i:.1e}",
                        'opt': optimizer_type.upper()
                    }
                    outer_pbar.set_postfix(inner_postfix)  # type: ignore[attr-defined]
            epoch_global += 1
            if verbose:
                outer_pbar.update(1)

    if verbose:
        outer_pbar.close()

    final_vol = recon.detach()[0, 0]
    return final_vol


# ---------------------------------------------------------------------------
# Differentiable laminography projector
# ---------------------------------------------------------------------------

class _AstraLaminoOp:
    """ASTRA laminography forward/adjoint operator."""
    
    def __init__(self, vol_geom, proj_geom, vol_shape, det_rows, det_cols, n_views):
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
        self.vol_shape = vol_shape
        self.det_rows = det_rows
        self.det_cols = det_cols
        self.n_views = n_views
    
    def forward(self, vol_t, out_sino_t=None):
        """Forward projection: volume -> sinogram."""
        if out_sino_t is None:
            out_sino_t = torch.empty((self.det_rows, self.n_views, self.det_cols), device=vol_t.device, dtype=torch.float32)
        vol_id  = astra.data3d.link('-vol',  self.vol_geom, vol_t.detach())
        sino_id = astra.data3d.link('-sino', self.proj_geom, out_sino_t.detach())
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['VolumeDataId'] = vol_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(vol_id)
        return out_sino_t

    def adjoint(self, sino_t, out_vol_t=None):
        """Adjoint/backprojection: sinogram -> volume."""
        if out_vol_t is None:
            out_vol_t = torch.zeros(self.vol_shape, device=sino_t.device, dtype=torch.float32)
        vol_id  = astra.data3d.link('-vol',  self.vol_geom, out_vol_t.detach())
        sino_id = astra.data3d.link('-sino', self.proj_geom, sino_t.detach())
        cfg = astra.astra_dict('BP3D_CUDA')
        cfg['ReconstructionDataId'] = vol_id
        cfg['ProjectionDataId']     = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(vol_id)
        return out_vol_t


def build_lamino_projector(
    vol_shape: Tuple[int, int, int],
    det_shape: Tuple[int, int],  # (rows, cols)
    angles_deg: np.ndarray,
    lamino_angle_deg: float,
    tilt_angle_deg: float = 0.0,
    voxel_size_mm: float = 1.0,
    det_spacing_mm: float = 1.0,
    device: Optional[torch.device] = None,
):
    """Return a differentiable laminography projection layer.

    Returns a function handle that acts like: y = projector(x) where
    x is (B,1,nx,ny,nz) and y is (B, V, R, C). Autograd is supported.
    
    Parameters
    ----------
    vol_shape : (nx, ny, nz)
        Volume dimensions
    det_shape : (rows, cols)
        Detector dimensions
    angles_deg : array
        Rotation angles in degrees
    lamino_angle_deg : float
        Laminography angle 
    tilt_angle_deg : float
        Additional tilt parameter
    voxel_size_mm : float
        Voxel size in mm
    det_spacing_mm : float
        Detector pixel spacing in mm
    device : torch.device
        Compute device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    det_rows, det_cols = det_shape
    
    vol_geom, proj_geom = _create_lamino_geometry(
        vol_shape=vol_shape,
        det_shape=det_shape,
        angles_deg=angles_deg,
        lamino_angle_deg=lamino_angle_deg,
        tilt_angle_deg=tilt_angle_deg,
        voxel_size_mm=voxel_size_mm,
        det_spacing_mm=det_spacing_mm,
    )
    
    op = _AstraLaminoOp(vol_geom, proj_geom, vol_shape, det_rows, det_cols, len(angles_deg))

    class LaminoProjectorFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            if x.ndim != 5 or x.shape[1] != 1:
                raise ValueError('Input must have shape (B,1,nx,ny,nz)')
            B = x.shape[0]
            y_out = torch.empty((B, op.n_views, det_rows, det_cols), device=x.device, dtype=torch.float32)
            for i in range(B):
                vol_t = x[i, 0]
                sino_rvc = op.forward(vol_t)  # (R,V,C)
                y_out[i] = sino_rvc.permute(1, 0, 2).contiguous()
            ctx.op = op
            return y_out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            op_local = ctx.op
            B = grad_out.shape[0]
            g_vols = []
            for i in range(B):
                g_y = grad_out[i]  # (V,R,C)
                # reorder to ASTRA (R,V,C)
                g_sino_rvc = g_y.permute(1, 0, 2).contiguous()
                g_vol = torch.zeros(op_local.vol_shape, device=g_y.device, dtype=torch.float32)
                op_local.adjoint(g_sino_rvc, g_vol)
                g_vols.append(g_vol)
            g_stack = torch.stack(g_vols, dim=0).unsqueeze(1)  # (B,1,nx,ny,nz)
            return g_stack

    def layer(x: torch.Tensor) -> torch.Tensor:
        return LaminoProjectorFn.apply(x)

    return layer
