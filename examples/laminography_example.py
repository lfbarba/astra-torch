#!/usr/bin/env python3
"""
Laminography Reconstruction Example
===================================

This example demonstrates laminography reconstruction using filtered backprojection.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from astra_torch import lamino_fbp_reconstruction_masked


def create_laminography_geometry(num_angles=180, tilt_angle=np.pi/6):
    """Create laminography acquisition geometry with tilted rotation axis.
    
    Args:
        num_angles: Number of projection angles
        tilt_angle: Tilt angle of rotation axis from vertical (radians)
        
    Returns:
        vectors: ASTRA laminography vectors (num_angles, 9)
    """
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    vectors = []
    
    for angle in angles:
        # Ray direction (parallel beam geometry)
        ray_dir = np.array([
            np.cos(angle) * np.cos(tilt_angle),
            np.sin(angle) * np.cos(tilt_angle),
            np.sin(tilt_angle)
        ])
        
        # Detector u vector (horizontal in detector plane)
        u_vec = np.array([-np.sin(angle), np.cos(angle), 0])
        
        # Detector v vector (perpendicular to both ray and u)
        v_vec = np.cross(ray_dir, u_vec)
        v_vec = v_vec / np.linalg.norm(v_vec)
        
        # Combine into ASTRA laminography vector
        vectors.append(np.concatenate([ray_dir, u_vec, v_vec]))
    
    return np.array(vectors)


def create_layered_phantom(shape=(64, 64, 64)):
    """Create a layered phantom suitable for laminography.
    
    Laminography is particularly useful for thin, layered samples,
    so we create a phantom with multiple layers and features.
    
    Args:
        shape: Volume dimensions
        
    Returns:
        phantom: 3D numpy array
    """
    phantom = np.zeros(shape)
    
    # Add several layers with different structures
    z_center = shape[2] // 2
    layer_thickness = shape[2] // 8
    
    # Bottom layer - grid pattern
    z_start = z_center - 2 * layer_thickness
    z_end = z_center - layer_thickness
    for z in range(max(0, z_start), min(shape[2], z_end)):
        # Grid pattern
        phantom[::8, :, z] = 0.6
        phantom[:, ::8, z] = 0.6
    
    # Middle layer - circular features  
    x, y, z = np.mgrid[:shape[0], :shape[1], :shape[2]]
    z_start = z_center - layer_thickness//2
    z_end = z_center + layer_thickness//2
    
    layer_mask = (z >= z_start) & (z <= z_end)
    
    # Add circular features in middle layer
    centers = [(shape[0]//3, shape[1]//3), 
               (2*shape[0]//3, shape[1]//3),
               (shape[0]//2, 2*shape[1]//3)]
    
    for cx, cy in centers:
        circle = ((x - cx)**2 + (y - cy)**2) <= (shape[0]//10)**2
        phantom[circle & layer_mask] = 0.8
    
    # Top layer - line features
    z_start = z_center + layer_thickness
    z_end = z_center + 2 * layer_thickness
    for z in range(max(0, z_start), min(shape[2], z_end)):
        # Diagonal lines
        for i in range(0, shape[0], 6):
            if i + 2 < shape[0]:
                phantom[i:i+2, :, z] = 0.4
        for j in range(0, shape[1], 6):
            if j + 2 < shape[1]:
                phantom[:, j:j+2, z] = 0.4
    
    return phantom


def simulate_laminography_projections(phantom, vectors):
    """Simulate laminography projections.
    
    Args:
        phantom: 3D volume
        vectors: Laminography geometry vectors
        
    Returns:
        projections: Simulated projection data
    """
    num_angles = vectors.shape[0]
    detector_shape = (256, 256)
    
    # Simple projection simulation
    projections = np.zeros((num_angles,) + detector_shape)
    
    for i, vector in enumerate(vectors):
        # Ray direction
        ray_dir = vector[:3]
        
        # Simple line integral approximation
        # In practice, this would use proper ray tracing
        projection_sum = np.sum(phantom) * abs(ray_dir[2])  # Z-component weight
        
        # Add some angular variation and noise
        angle_factor = 1 + 0.2 * np.sin(2 * np.pi * i / num_angles)
        base_intensity = projection_sum * angle_factor / (num_angles * 10)
        
        # Generate Poisson noise
        projections[i] = np.random.poisson(
            base_intensity * 1000, 
            size=detector_shape
        ).astype(np.float32) / 1000.0
        
        # Add some detector-specific patterns
        u, v = np.meshgrid(np.linspace(-1, 1, detector_shape[1]),
                          np.linspace(-1, 1, detector_shape[0]))
        projections[i] *= (1 + 0.1 * np.sin(5 * u) * np.cos(3 * v))
    
    return projections


def main():
    """Main laminography example."""
    print("Laminography Reconstruction Example")
    print("=" * 40)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create laminography geometry
    print("Creating laminography geometry...")
    num_angles = 180
    tilt_angle = np.pi/6  # 30 degrees
    vectors = create_laminography_geometry(num_angles, tilt_angle)
    print(f"Created geometry with {num_angles} angles, tilt = {np.degrees(tilt_angle):.1f}°")
    
    # Create layered phantom
    print("Creating layered phantom...")
    phantom = create_layered_phantom((64, 64, 64))
    print(f"Phantom shape: {phantom.shape}")
    
    # Simulate projections
    print("Simulating laminography projections...")
    projections = simulate_laminography_projections(phantom, vectors)
    print(f"Projection data shape: {projections.shape}")
    
    # Convert to PyTorch tensors
    projections_torch = torch.from_numpy(projections).float()
    projections_torch = projections_torch.unsqueeze(0).unsqueeze(0)
    projections_torch = projections_torch.to(device)
    
    # Perform FBP reconstruction
    print("Performing laminography FBP reconstruction...")
    try:
        volume = lamino_fbp_reconstruction_masked(
            projections_torch,
            vectors,
            voxel_per_mm=2.0,
            volume_shape=(64, 64, 64)
        )
        print(f"Reconstruction successful! Volume shape: {volume.shape}")
        
        # Convert to numpy for visualization
        volume_np = volume.squeeze().cpu().numpy()
        
        # Visualize results
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # Show different Z slices of original phantom
        z_slices = [16, 28, 36, 48]
        for i, z_idx in enumerate(z_slices):
            axes[0, i].imshow(phantom[:, :, z_idx], cmap='gray')
            axes[0, i].set_title(f'Original Z={z_idx}')
            axes[0, i].axis('off')
        
        # Show same Z slices of reconstruction
        for i, z_idx in enumerate(z_slices):
            axes[1, i].imshow(volume_np[:, :, z_idx], cmap='gray')
            axes[1, i].set_title(f'Reconstructed Z={z_idx}')
            axes[1, i].axis('off')
        
        # Show some projections
        proj_indices = [0, num_angles//4, num_angles//2, 3*num_angles//4]
        for i, proj_idx in enumerate(proj_indices):
            axes[2, i].imshow(projections[proj_idx], cmap='gray')
            axes[2, i].set_title(f'Projection {proj_idx}')
            axes[2, i].axis('off')
        
        plt.suptitle('Laminography Reconstruction Results', fontsize=14)
        plt.tight_layout()
        plt.savefig('laminography_example.png', dpi=150, bbox_inches='tight')
        print("Results saved to 'laminography_example.png'")
        plt.show()
        
        # Calculate and display some metrics
        print("\nReconstruction Quality Metrics:")
        print(f"Original phantom range: [{phantom.min():.3f}, {phantom.max():.3f}]")
        print(f"Reconstructed range: [{volume_np.min():.3f}, {volume_np.max():.3f}]")
        
        # Focus on central region for comparison
        center_slice = slice(16, 48)
        phantom_center = phantom[center_slice, center_slice, center_slice]
        volume_center = volume_np[center_slice, center_slice, center_slice]
        
        mse = np.mean((phantom_center - volume_center)**2)
        print(f"MSE (center region): {mse:.6f}")
        
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        print("This might be due to missing ASTRA toolbox installation.")
        print("Please install ASTRA: conda install -c astra-toolbox astra-toolbox")


if __name__ == "__main__":
    main()
