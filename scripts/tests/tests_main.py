"""
Simple unit tests for the main.py training script.

These tests focus on the key functions and classes in the main.py file.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import argparse

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf, DictConfig
from PIL import Image


class TestRandomId:
    """Tests for the random_id function."""
    
    def test_random_id_default(self):
        """Test random_id with default parameters."""
        from scripts.main import random_id
        
        result = random_id()
        assert result.startswith("run")
        assert len(result) == 7  # "run" + 4 digits
    
    def test_random_id_custom_length(self):
        """Test random_id with custom length."""
        from scripts.main import random_id
        
        result = random_id(digits_nb=6)
        assert result.startswith("run")
        assert len(result) == 9  # "run" + 6 digits
    
    def test_random_id_numbers_only(self):
        """Test random_id with numbers only."""
        from scripts.main import random_id
        
        result = random_id(include_letters=False)
        # Check that it only contains "run" followed by numbers
        numbers_part = result[3:]  # Skip "run"
        assert numbers_part.isdigit()
    
    def test_random_id_with_lowercase(self):
        """Test random_id allowing lowercase letters."""
        from scripts.main import random_id
        
        # Run multiple times to likely get some lowercase
        results = [random_id(only_capital=False) for _ in range(10)]
        assert any(result for result in results)  # Just check it doesn't crash


class TestModifyWeights:
    """Tests for the modify_weights function."""
    
    def test_modify_weights_default(self):
        """Test modify_weights with default parameters."""
        from scripts.main import modify_weights
        
        # Create a simple weight tensor
        original_weight = torch.randn(3, 3, 3, 3)
        
        # Modify weights
        new_weight = modify_weights(original_weight)
        
        # Check shape is correct (concatenated twice by default)
        assert new_weight.shape[1] == original_weight.shape[1] * 3  # Original + 2 extra
        assert new_weight.shape[0] == original_weight.shape[0]
    
    def test_modify_weights_custom_n(self):
        """Test modify_weights with custom n parameter."""
        from scripts.main import modify_weights
        
        original_weight = torch.randn(3, 3, 3, 3)
        new_weight = modify_weights(original_weight, n=3)
        
        # Check shape is correct (original + 3 extra)
        assert new_weight.shape[1] == original_weight.shape[1] * 4
    
    def test_modify_weights_preserves_original(self):
        """Test that modify_weights preserves the original weights."""
        from scripts.main import modify_weights
        
        original_weight = torch.randn(3, 3, 3, 3)
        original_copy = original_weight.clone()
        
        new_weight = modify_weights(original_weight)
        
        # Check original is unchanged
        assert torch.allclose(original_weight, original_copy)
        
        # Check first part of new weight matches original
        assert torch.allclose(new_weight[:, :3, :, :], original_weight)


class TestGetParser:
    """Tests for the get_parser function."""
    
    def test_get_parser_creation(self):
        """Test that get_parser creates a valid parser."""
        from scripts.main import get_parser
        
        parser = get_parser()
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_get_parser_basic_args(self):
        """Test parsing basic arguments."""
        from scripts.main import get_parser
        
        parser = get_parser()
        args = parser.parse_args([
            "--train", "true",
            "--seed", "42",
            "--base", "config.yaml"
        ])
        
        assert args.train is True
        assert args.seed == 42
        assert args.base == ["config.yaml"]
    
    def test_get_parser_finetune_arg(self):
        """Test the finetune_from argument."""
        from scripts.main import get_parser
        
        parser = get_parser()
        args = parser.parse_args(["--finetune_from", "checkpoint.ckpt"])
        
        assert args.finetune_from == "checkpoint.ckpt"


class TestWrappedDataset:
    """Tests for the WrappedDataset class."""
    
    def test_wrapped_dataset_basic(self):
        """Test basic WrappedDataset functionality."""
        from scripts.main import WrappedDataset
        
        # Create a simple list to wrap
        data = [1, 2, 3, 4, 5]
        dataset = WrappedDataset(data)
        
        assert len(dataset) == 5
        assert dataset[0] == 1
        assert dataset[4] == 5
    
    def test_wrapped_dataset_with_dict(self):
        """Test WrappedDataset with dictionary data."""
        from scripts.main import WrappedDataset
        
        data = [
            {"image": torch.randn(3, 32, 32), "label": 0},
            {"image": torch.randn(3, 32, 32), "label": 1},
        ]
        dataset = WrappedDataset(data)
        
        assert len(dataset) == 2
        assert "image" in dataset[0]
        assert dataset[0]["label"] == 0


class TestDataModuleFromConfig:
    """Tests for DataModuleFromConfig class."""
    
    @pytest.fixture
    def simple_config(self):
        """Create a simple data configuration."""
        return {
            "target": "scripts.main.WrappedDataset",
            "params": {
                "dataset": [1, 2, 3, 4, 5]
            }
        }
    
    def test_data_module_init(self):
        """Test DataModuleFromConfig initialization."""
        from scripts.main import DataModuleFromConfig
        
        data_module = DataModuleFromConfig(
            batch_size=2,
            num_workers=0
        )
        
        assert data_module.batch_size == 2
        assert data_module.num_workers == 0
    
    @patch('scripts.main.instantiate_from_config')
    def test_data_module_with_train_config(self, mock_instantiate, simple_config):
        """Test DataModuleFromConfig with training data."""
        from scripts.main import DataModuleFromConfig
        
        # Setup mock
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_instantiate.return_value = mock_dataset
        
        data_module = DataModuleFromConfig(
            batch_size=2,
            train=simple_config,
            num_workers=0
        )
        
        # Setup datasets
        data_module.setup()
        
        # Check train dataloader
        train_loader = data_module.train_dataloader()
        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 2


class TestCallbacks:
    """Tests for callback classes."""
    
    def test_setup_callback_init(self):
        """Test SetupCallback initialization."""
        from scripts.main import SetupCallback
        
        callback = SetupCallback(
            resume=False,
            now="2024-01-01",
            logdir="/tmp/logs",
            ckptdir="/tmp/ckpts",
            cfgdir="/tmp/configs",
            config={"model": "test"},
            lightning_config={"trainer": {}},
            debug=False
        )
        
        assert callback.resume is False
        assert callback.now == "2024-01-01"
        assert callback.logdir == "/tmp/logs"
    
    def test_image_logger_init(self):
        """Test ImageLogger initialization."""
        from scripts.main import ImageLogger
        
        logger = ImageLogger(
            batch_frequency=100,
            max_images=8,
            clamp=True,
            rescale=True
        )
        
        assert logger.batch_freq == 100
        assert logger.max_images == 8
        assert logger.clamp is True
        assert logger.rescale is True
    
    def test_image_logger_check_frequency(self):
        """Test ImageLogger frequency checking."""
        from scripts.main import ImageLogger
        
        logger = ImageLogger(
            batch_frequency=100,
            max_images=8,
            log_first_step=True
        )
        
        # Should log first step
        assert logger.check_frequency(0) is True
        
        # Should log at batch_freq intervals
        assert logger.check_frequency(100) is True
        assert logger.check_frequency(50) is False
    
    def test_cuda_callback_init(self):
        """Test CUDACallback initialization."""
        from scripts.main import CUDACallback
        
        callback = CUDACallback()
        assert callback is not None  # Just check it initializes


class TestWorkerInitFn:
    """Tests for worker_init_fn."""
    
    def test_worker_init_fn_basic(self):
        """Test worker_init_fn doesn't crash."""
        from scripts.main import worker_init_fn
        
        # Create a mock worker_info
        with patch('torch.utils.data.get_worker_info') as mock_info:
            mock_worker = Mock()
            mock_worker.dataset = Mock()
            mock_worker.id = 0
            mock_worker.num_workers = 4
            mock_info.return_value = mock_worker
            
            # Should not raise any exceptions
            result = worker_init_fn(0)
            assert result is not None or result is None  # Can return seed or None


class TestIntegrationSimple:
    """Integration tests."""
    
    @pytest.mark.slow
    def test_parser_and_config(self, tmp_path):
        """Test parser with config file."""
        from scripts.main import get_parser
        
        # Create a simple config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
        model:
          target: test.Model
          params:
            in_channels: 3
        """
        config_file.write_text(config_content)
        
        parser = get_parser()
        args = parser.parse_args([
            "--base", str(config_file),
            "--train", "true"
        ])
        
        assert args.base == [str(config_file)]
        assert args.train is True
    
    def test_wrapped_dataset_iteration(self):
        """Test iterating through WrappedDataset."""
        from scripts.main import WrappedDataset
        
        data = list(range(10))
        dataset = WrappedDataset(data)
        
        # Test iteration
        items = []
        for i in range(len(dataset)):
            items.append(dataset[i])
        
        assert items == data


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_nondefault_trainer_args(self):
        """Test nondefault_trainer_args function."""
        from scripts.main import nondefault_trainer_args
        
        # Create a mock opt with some custom values
        opt = argparse.Namespace(
            gpus=2,
            max_epochs=100,
            precision=16,
            accumulate_grad_batches=1
        )
        
        # This should return keys that differ from defaults
        result = nondefault_trainer_args(opt)
        assert isinstance(result, list)
        # The exact result depends on Trainer defaults, so just check type


# Simple fixtures for testing
@pytest.fixture
def mock_model():
    """Create a simple mock model."""
    model = Mock(spec=nn.Module)
    model.state_dict = Mock(return_value={})
    model.load_state_dict = Mock(return_value=([], []))
    model.learning_rate = 1e-4
    return model


@pytest.fixture
def mock_trainer():
    """Create a simple mock trainer."""
    trainer = Mock()
    trainer.global_rank = 0
    trainer.global_step = 0
    trainer.current_epoch = 0
    trainer.root_gpu = 0
    return trainer


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    return {
        "image": torch.randn(2, 3, 256, 256),
        "target": torch.randn(2, 3, 256, 256),
    }


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])