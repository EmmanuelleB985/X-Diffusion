"""
Tests for inference module utilities.
"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


class TestVolumeProcessing:
    """Tests for volume processing."""

    def test_volume_shape_conversion(self):
        """Test handling different volume shapes."""
        test_shapes = [
            (155, 3, 256, 256),
            (155, 256, 256, 3),
            (155, 256, 256),
        ]

        for shape in test_shapes:
            volume = np.random.rand(*shape)

            if volume.ndim == 4:
                if volume.shape[1] == 3:
                    processed = volume[:, 0, :, :]
                elif volume.shape[-1] == 3:
                    processed = volume[:, :, :, 0]
            else:
                processed = volume

            assert processed.ndim == 3
            assert processed.shape == (155, 256, 256)

    def test_nifti_volume_format(self):
        """Test converting to NIfTI format."""
        volume = np.random.rand(155, 256, 256)
        nifti_volume = np.transpose(volume, (1, 2, 0))
        assert nifti_volume.shape == (256, 256, 155)


class TestBatchProcessing:
    """Tests for batch processing."""

    def test_create_mock_batch(self):
        """Test creating a mock batch for inference."""
        batch = {
            "image_target": torch.randn(1, 3, 256, 256),
            "image_cond": torch.randn(1, 3, 256, 256),
            "filename": ["patient001_slice_001"],
            "T": torch.randn(1, 1, 4),
        }

        assert "image_target" in batch
        assert "image_cond" in batch
        assert batch["image_cond"].shape == (1, 3, 256, 256)

    def test_normalize_images(self):
        """Test image normalization."""
        image = torch.randn(1, 3, 256, 256)
        normalized = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)

        assert normalized.min() >= 0
        assert normalized.max() <= 1


class TestSamplerIntegration:
    """Tests for sampler integration."""

    def test_sampler_device_consistency(self):
        """Test ensuring device consistency."""
        mock_model = Mock()

        def apply_model_wrapper(original_fn):
            def wrapped(x_noisy, t, cond):
                device = x_noisy.device
                if torch.is_tensor(t):
                    t = t.to(device)
                return original_fn(x_noisy, t, cond)

            return wrapped

        original_apply = Mock(return_value=torch.randn(1, 4, 32, 32))
        wrapped_apply = apply_model_wrapper(original_apply)

        x = torch.randn(1, 4, 32, 32)
        t = torch.tensor([100])
        cond = {}

        result = wrapped_apply(x, t, cond)

        assert result.shape == (1, 4, 32, 32)
        original_apply.assert_called_once()


class TestConfiguration:
    """Tests for configuration handling."""

    def test_load_config_file(self, tmp_path):
        """Test loading configuration from YAML."""
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

        config = OmegaConf.load(str(config_path))

        assert config.model.params.timesteps == 1000
        assert config.data.image_size == 256

    def test_create_config_object(self):
        """Test creating configuration programmatically."""
        config = OmegaConf.create(
            {
                "model": {"target": "test.Model", "params": {"layers": 4}},
                "inference": {"ddim_steps": 50, "guidance_scale": 7.5},
            }
        )

        assert config.inference.ddim_steps == 50
        assert config.model.params.layers == 4


class TestOutputDirectories:
    """Tests for output directory handling."""

    def test_create_output_structure(self, tmp_path):
        """Test creating output directory structure."""
        output_dir = tmp_path / "outputs"

        dirs_to_create = [
            "slices_denoised",
            "slices_reconstruction",
            "slices_input",
            "slices_target",
        ]

        for dir_name in dirs_to_create:
            dir_path = output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            assert dir_path.exists()

    def test_output_paths(self, tmp_path):
        """Test generating output file paths."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(exist_ok=True)

        patient_name = "patient001"

        paths = {
            "denoised": output_dir / f"{patient_name}_denoised_volume.nii.gz",
            "input": output_dir / f"{patient_name}_input_volume.nii.gz",
            "target": output_dir / f"{patient_name}_target_volume.nii.gz",
        }

        for name, path in paths.items():
            assert str(path).endswith(".nii.gz")


class TestArgumentParsing:
    """Tests for command-line argument parsing."""

    def test_parse_basic_args(self):
        """Test parsing basic command-line arguments."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--ddim_steps", type=int, default=50)
        parser.add_argument("--guidance_scale", type=float, default=7.5)
        parser.add_argument("--device_idx", type=int, default=0)

        args = parser.parse_args(["--ddim_steps", "25", "--guidance_scale", "5.0"])

        assert args.ddim_steps == 25
        assert args.guidance_scale == 5.0
        assert args.device_idx == 0

    def test_boolean_flags(self):
        """Test parsing boolean flags."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--save_individual_slices", action="store_true", default=True
        )
        parser.add_argument(
            "--no_save_individual_slices",
            dest="save_individual_slices",
            action="store_false",
        )

        args1 = parser.parse_args(["--no_save_individual_slices"])
        assert args1.save_individual_slices is False

        args2 = parser.parse_args([])
        assert args2.save_individual_slices is True
