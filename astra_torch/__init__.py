"""ASTRA-Torch: GPU-accelerated tomographic reconstruction library.

This library provides PyTorch-compatible implementations of tomographic reconstruction
algorithms using the ASTRA toolbox, including support for CBCT and laminography.

Modules:
--------
cbct : Cone-beam CT reconstruction functions
lamino : Laminography reconstruction functions

Examples:
---------
>>> import astra_torch
>>> from astra_torch import fdk_reconstruction_masked
>>> volume = fdk_reconstruction_masked(projs, vecs, voxel_per_mm=10)
"""

# Import CBCT functions
from .cbct import (
    project_conebeam_masked, 
    build_conebeam_projector, 
    CBCTAcquisition,
    fdk_reconstruction_masked as cbct_fdk_reconstruction_masked,
    gd_reconstruction_masked as cbct_gd_reconstruction_masked
)

# Import laminography functions  
from .lamino import (
    fbp_reconstruction_masked as lamino_fbp_reconstruction_masked,
    sirt_reconstruction_masked as lamino_sirt_reconstruction_masked,
    gd_reconstruction_masked as lamino_gd_reconstruction_masked,
    build_lamino_projector, 
    LaminoAcquisition
)

# For backward compatibility, expose some functions without prefixes
from .cbct import fdk_reconstruction_masked, gd_reconstruction_masked

__version__ = "0.1.0"
__author__ = "CHIP Project"

__all__ = [
    # CBCT functions
    'cbct_fdk_reconstruction_masked',
    'cbct_gd_reconstruction_masked', 
    'fdk_reconstruction_masked',  # Backward compatibility
    'gd_reconstruction_masked',   # Backward compatibility
    'project_conebeam_masked',
    'build_conebeam_projector',
    'CBCTAcquisition',
    
    # Laminography functions
    'lamino_fbp_reconstruction_masked',
    'lamino_sirt_reconstruction_masked',
    'lamino_gd_reconstruction_masked',
    'build_lamino_projector',
    'LaminoAcquisition',
    
    # Version info
    '__version__',
    '__author__'
]
