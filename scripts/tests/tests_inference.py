"""
Tests for actual inference.py module.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pytest
import torch
from PIL import Image
from omegaconf import OmegaConf


class TestBasicFunctions:
    """Tests for basic inference functions."""
    
    def test_save_slice_as_png(self, tmp_path):
        """Test saving a tensor slice as PNG image."""
        from scripts.inference import save_slice_as_png
        
        # Create a simple tensor
        tensor = torch.rand(3, 64, 64)  # RGB image
        output_path = tmp_path / "test.png"
        
        # Save it
        save_slice_as_png(tensor, str(output_path))
        
        # Check file exists
        assert output_path.exists()
        
        # Load and verify
        img = Image.open(output_path)
        assert img.size == (64, 64)
        print(f"Successfully saved image to {output_path}")
    
    def test_save_grayscale_slice(self, tmp_path):
        """Test saving grayscale slice."""
        from scripts.inference import save_slice_as_png
        
        # Single channel image
        tensor = torch.rand(1, 64, 64)
        output_path = tmp_path / "gray.png"
        
        save_slice_as_png(tensor, str(output_path))
        
        assert output_path.exists()
        print(f"Grayscale image saved")


class TestModelLoading:
    """Tests for model loading."""
    
    @patch('scripts.inference.instantiate_from_config')
    @patch('torch.load')
    def test_load_model_basic(self, mock_load, mock_instantiate):
        """Test basic model loading."""
        from scripts.inference import load_model_from_config
        
        # Setup mocks
        mock_load.return_value = {
            'state_dict': {'layer': torch.randn(3, 3)}
        }
        
        mock_model = Mock()
        mock_model.load_state_dict = Mock(return_value=([], []))
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.children = Mock(return_value=[])
        mock_instantiate.return_value = mock_model
        
        # Create simple config
        config = OmegaConf.create({
            'model': {'target': 'test.Model'}
        })
        
        # Load model
        model = load_model_from_config(
            config, 
            'checkpoint.ckpt',
            device='cpu'
        )
        
        # Check model was loaded
        assert model is not None
        mock_load.assert_called_once()
        print("Model loaded successfully")


class TestVolumeProcessing:
    """Tests for volume processing."""
    
    def test_volume_shape_conversion(self):
        """Test handling different volume shapes."""
        # Different possible input shapes
        test_shapes = [
            (155, 3, 256, 256),  # [slices, channels, H, W]
            (155, 256, 256, 3),  # [slices, H, W, channels]
            (155, 256, 256),     # [slices, H, W]
        ]
        
        for shape in test_shapes:
            volume = np.random.rand(*shape)
            
            # Process volume to get [slices, H, W]
            if volume.ndim == 4:
                if volume.shape[1] == 3:  # Channels at dim 1
                    processed = volume[:, 0, :, :]
                elif volume.shape[-1] == 3:  # Channels at end
                    processed = volume[:, :, :, 0]
            else:
                processed = volume
            
            # Check result
            assert processed.ndim == 3
            assert processed.shape == (155, 256, 256)
            print(f"Processed volume shape {shape} -> {processed.shape}")
    
    def test_nifti_volume_format(self):
        """Test converting to NIfTI format."""
        # Create a volume [slices, H, W]
        volume = np.random.rand(155, 256, 256)
        
        # Convert to NIfTI format [H, W, slices]
        nifti_volume = np.transpose(volume, (1, 2, 0))
        
        assert nifti_volume.shape == (256, 256, 155)
        print(f"NIfTI format: {volume.shape} -> {nifti_volume.shape}")


class TestBatchProcessing:
    """Tests for batch processing."""
    
    def test_create_mock_batch(self):
        """Test creating a mock batch for inference."""
        batch = {
            'image_target': torch.randn(1, 3, 256, 256),
            'image_cond': torch.randn(1, 3, 256, 256), 
            'filename': ['patient001_slice_001'],
            'T': torch.randn(1, 1, 4)
        }
        
        # Check batch structure
        assert 'image_target' in batch
        assert 'image_cond' in batch
        assert batch['image_cond'].shape == (1, 3, 256, 256)
        print("Mock batch created successfully")
    
    def test_normalize_images(self):
        """Test image normalization."""
        # Create image in [-1, 1] range
        image = torch.randn(1, 3, 256, 256)
        
        # Normalize to [0, 1]
        normalized = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        
        # Check range
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        print(f"Image normalized: [{image.min():.2f}, {image.max():.2f}] -> [0, 1]")


class TestSamplerIntegration:
    """Tests for sampler integration."""
    
    def test_create_sampler(self):
        """Test creating DDIM sampler."""
        from ldm.models.diffusion.ddim import DDIMSampler
        
        # Create mock model
        mock_model = Mock()
        mock_model.device = 'cpu'
        
        # Create sampler
        sampler = DDIMSampler(mock_model)
        
        assert sampler is not None
        assert sampler.model == mock_model
        print("DDIM sampler created")
    
    def test_sampler_device_consistency(self):
        """Test ensuring device consistency."""
        # This tests the device fix pattern used in inference
        mock_model = Mock()
        
        def apply_model_wrapper(original_fn):
            def wrapped(x_noisy, t, cond):
                device = x_noisy.device
                if torch.is_tensor(t):
                    t = t.to(device)
                return original_fn(x_noisy, t, cond)
            return wrapped
        
        # Original function
        original_apply = Mock(return_value=torch.randn(1, 4, 32, 32))
        
        # Wrap it
        wrapped_apply = apply_model_wrapper(original_apply)
        
        # Test with tensors
        x = torch.randn(1, 4, 32, 32)
        t = torch.tensor([100])
        cond = {}
        
        result = wrapped_apply(x, t, cond)
        
        assert result.shape == (1, 4, 32, 32)
        original_apply.assert_called_once()
        print("Device consistency wrapper works")


class TestConfiguration:
    """Tests for configuration handling."""
    
    def test_load_config_file(self, tmp_path):
        """Test loading configuration from YAML."""
        # Create a simple config file
        config_path = tmp_path / "config.yaml"
        config_content = """
model:
  target: ldm.models.diffusion.ddpm
  params:
    timesteps: 1000
    
data:
  image_size: 256
  batch_size: 1
"""
        config_path.write_text(config_content)
        
        # Load config
        config = OmegaConf.load(str(config_path))
        
        assert config.model.params.timesteps == 1000
        assert config.data.image_size == 256
        print(f"Config loaded: {config.model.target}")
    
    def test_create_config_object(self):
        """Test creating configuration programmatically."""
        config = OmegaConf.create({
            'model': {
                'target': 'test.Model',
                'params': {'layers': 4}
            },
            'inference': {
                'ddim_steps': 50,
                'guidance_scale': 7.5
            }
        })
        
        assert config.inference.ddim_steps == 50
        assert config.model.params.layers == 4
        print("Config object created")


class TestOutputDirectories:
    """Simple tests for output directory handling."""
    
    def test_create_output_structure(self, tmp_path):
        """Test creating output directory structure."""
        output_dir = tmp_path / "outputs"
        
        # Create subdirectories
        dirs_to_create = [
            'slices_denoised',
            'slices_reconstruction', 
            'slices_input',
            'slices_target'
        ]
        
        for dir_name in dirs_to_create:
            dir_path = output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            assert dir_path.exists()
        
        print(f"Created {len(dirs_to_create)} output directories")
    
    def test_output_paths(self, tmp_path):
        """Test generating output file paths."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        patient_name = "patient001"
        
        # Generate paths
        paths = {
            'denoised': output_dir / f"{patient_name}_denoised_volume.nii.gz",
            'input': output_dir / f"{patient_name}_input_volume.nii.gz",
            'target': output_dir / f"{patient_name}_target_volume.nii.gz",
        }
        
        for name, path in paths.items():
            assert str(path).endswith('.nii.gz')
            print(f"  {name}: {path.name}")
        
        print("Output paths generated correctly")


class TestArgumentParsing:
    """Tests for command-line argument parsing."""
    
    def test_parse_basic_args(self):
        """Test parsing basic command-line arguments."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--ddim_steps', type=int, default=50)
        parser.add_argument('--guidance_scale', type=float, default=7.5)
        parser.add_argument('--device_idx', type=int, default=0)
        
        # Parse test args
        args = parser.parse_args([
            '--ddim_steps', '25',
            '--guidance_scale', '5.0'
        ])
        
        assert args.ddim_steps == 25
        assert args.guidance_scale == 5.0
        assert args.device_idx == 0  # Default value
        print(f"Parsed: steps={args.ddim_steps}, scale={args.guidance_scale}")
    
    def test_boolean_flags(self):
        """Test parsing boolean flags."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--save_individual_slices', 
                          action='store_true', default=True)
        parser.add_argument('--no_save_individual_slices',
                          dest='save_individual_slices',
                          action='store_false')
        
        # Test with flag
        args1 = parser.parse_args(['--no_save_individual_slices'])
        assert args1.save_individual_slices is False
        
        # Test without flag (default)
        args2 = parser.parse_args([])
        assert args2.save_individual_slices is True
        
        print("Boolean flags parsed correctly")


# Quick test runner
def run_simple_tests():
 
    print("Running Inference Tests")
    
    test_classes = [
        TestBasicFunctions,
        TestModelLoading,
        TestVolumeProcessing,
        TestBatchProcessing,
        TestSamplerIntegration,
        TestConfiguration,
        TestOutputDirectories,
        TestArgumentParsing
    ]
    
    for test_class in test_classes:
        print(f"\n {test_class.__name__}")
        print("-" * 40)
        
        # Create instance
        test_obj = test_class()
        
        # Run all test methods
        for method_name in dir(test_obj):
            if method_name.startswith('test_'):
                method = getattr(test_obj, method_name)
                try:
                    # Create temp path if needed
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir)
                        
                        # Check if method needs tmp_path
                        import inspect
                        sig = inspect.signature(method)
                        if 'tmp_path' in sig.parameters:
                            method(tmp_path)
                        else:
                            method()
                except Exception as e:
                    print(f" {method_name}: {e}")

    print("Tests completed!")



if __name__ == "__main__":
    # Run the simple tests
    run_simple_tests()
    
    # Or run with pytest
    # pytest.main([__file__, "-v", "--tb=short"])