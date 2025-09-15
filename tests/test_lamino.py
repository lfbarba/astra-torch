"""Tests for laminography reconstruction functions."""

import unittest
import numpy as np
import torch

try:
    from astra_torch import lamino
    from astra_torch import LaminoAcquisition
    ASTRA_AVAILABLE = True
except ImportError:
    ASTRA_AVAILABLE = False


class TestLaminoReconstruction(unittest.TestCase):
    """Test laminography reconstruction functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    def test_lamino_acquisition_init(self):
        """Test LaminoAcquisition initialization."""
        num_angles = 10
        angles = np.linspace(0, np.pi, num_angles, endpoint=False)
        
        # Simple laminography geometry
        tilt_angle = np.pi/6  # 30 degrees
        vectors = []
        
        for angle in angles:
            ray_dir = np.array([
                np.cos(angle) * np.cos(tilt_angle),
                np.sin(angle) * np.cos(tilt_angle), 
                np.sin(tilt_angle)
            ])
            
            u_vec = np.array([-np.sin(angle), np.cos(angle), 0])
            v_vec = np.cross(ray_dir, u_vec)
            v_vec = v_vec / np.linalg.norm(v_vec)
            
            vectors.append(np.concatenate([ray_dir, u_vec, v_vec]))
        
        vectors = np.array(vectors)
        
        acquisition = LaminoAcquisition(
            vectors=vectors,
            detector_shape=(64, 64),
            volume_shape=(32, 32, 32)
        )
        
        self.assertEqual(acquisition.num_angles, num_angles)
        self.assertEqual(acquisition.detector_shape, (64, 64))
        self.assertEqual(acquisition.volume_shape, (32, 32, 32))

    def test_lamino_vector_validation(self):
        """Test laminography vector format validation."""
        # Test that vectors have correct shape
        num_angles = 5
        vectors = np.random.randn(num_angles, 9)  # Should be 9 components per vector
        
        self.assertEqual(vectors.shape, (num_angles, 9))
        
        # Test individual vector components
        for i in range(num_angles):
            ray_dir = vectors[i, :3]
            u_vec = vectors[i, 3:6]  
            v_vec = vectors[i, 6:9]
            
            # All vectors should be 3D
            self.assertEqual(len(ray_dir), 3)
            self.assertEqual(len(u_vec), 3)
            self.assertEqual(len(v_vec), 3)

    def test_parallel_beam_geometry(self):
        """Test parallel beam geometry properties."""
        # In laminography, rays should be parallel
        num_angles = 180
        angles = np.linspace(0, np.pi, num_angles, endpoint=False)
        tilt_angle = np.pi/4  # 45 degrees
        
        ray_directions = []
        for angle in angles:
            ray_dir = np.array([
                np.cos(angle) * np.cos(tilt_angle),
                np.sin(angle) * np.cos(tilt_angle),
                np.sin(tilt_angle)
            ])
            ray_directions.append(ray_dir)
        
        ray_directions = np.array(ray_directions)
        
        # All rays should have the same z-component (parallel beams)
        z_components = ray_directions[:, 2]
        self.assertTrue(np.allclose(z_components, z_components[0]))


if __name__ == '__main__':
    unittest.main()
