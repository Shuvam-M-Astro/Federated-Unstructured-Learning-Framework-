"""
Tests for configuration loading and validation.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from utils.config_loader import ConfigLoader
from utils.constants import ERROR_MESSAGES


class TestConfigLoader:
    """Test configuration loader functionality."""
    
    @pytest.fixture
    def valid_config(self):
        """Create valid configuration for testing."""
        return {
            'server': {
                'host': 'localhost',
                'port': 8000,
                'max_clients': 100,
                'timeout': 300
            },
            'training': {
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.001,
                'momentum': 0.9,
                'weight_decay': 0.0001
            },
            'privacy': {
                'differential_privacy': True,
                'epsilon': 0.5,
                'delta': 1e-5,
                'noise_multiplier': 1.1,
                'max_grad_norm': 1.0
            },
            'model': {
                'type': 'cnn',
                'input_size': [3, 224, 224],
                'num_classes': 10,
                'hidden_size': 128
            },
            'data_processing': {
                'supported_formats': ['text', 'image', 'tabular'],
                'max_file_size': 100,
                'preprocessing': {
                    'normalize': True,
                    'augment': True
                }
            }
        }
    
    @pytest.fixture
    def temp_config_file(self, valid_config):
        """Create temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_load_valid_config(self, temp_config_file):
        """Test loading valid configuration."""
        config_loader = ConfigLoader(temp_config_file)
        assert config_loader.config is not None
        assert 'server' in config_loader.config
        assert 'training' in config_loader.config
    
    def test_load_nonexistent_config(self):
        """Test loading nonexistent configuration file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            ConfigLoader('/nonexistent/path/config.yaml')
        
        assert 'Configuration file not found' in str(exc_info.value)
    
    def test_load_invalid_yaml(self):
        """Test loading invalid YAML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ConfigLoader(temp_path)
            assert 'Invalid YAML configuration' in str(exc_info.value)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validate_config(self, temp_config_file):
        """Test configuration validation."""
        config_loader = ConfigLoader(temp_config_file)
        assert config_loader.validate_config() is True
    
    def test_get_config_sections(self, temp_config_file):
        """Test getting configuration sections."""
        config_loader = ConfigLoader(temp_config_file)
        
        server_config = config_loader.get_server_config()
        assert 'host' in server_config
        assert server_config['port'] == 8000
        
        training_config = config_loader.get_training_config()
        assert 'epochs' in training_config
        assert training_config['batch_size'] == 32
    
    def test_get_nested_config(self, temp_config_file):
        """Test getting nested configuration values."""
        config_loader = ConfigLoader(temp_config_file)
        
        # Test dot notation
        port = config_loader.get('server.port')
        assert port == 8000
        
        # Test default value
        nonexistent = config_loader.get('nonexistent.key', default='default_value')
        assert nonexistent == 'default_value' 