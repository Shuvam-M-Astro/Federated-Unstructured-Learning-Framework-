"""
Configuration loader for federated learning system.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Configuration dictionary.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration.
        
        Returns:
            Server configuration dictionary
        """
        return self.config.get('server', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration.
        
        Returns:
            Training configuration dictionary
        """
        return self.config.get('training', {})
    
    def get_privacy_config(self) -> Dict[str, Any]:
        """Get privacy configuration.
        
        Returns:
            Privacy configuration dictionary
        """
        return self.config.get('privacy', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Model configuration dictionary
        """
        return self.config.get('model', {})
    
    def get_data_processing_config(self) -> Dict[str, Any]:
        """Get data processing configuration.
        
        Returns:
            Data processing configuration dictionary
        """
        return self.config.get('data_processing', {})
    
    def validate_config(self) -> bool:
        """Validate configuration values.
        
        Returns:
            True if configuration is valid
        """
        required_sections = ['server', 'training', 'privacy', 'model', 'data_processing']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate server config
        server_config = self.config['server']
        if not isinstance(server_config.get('port'), int):
            raise ValueError("Server port must be an integer")
        
        # Validate training config
        training_config = self.config['training']
        if training_config.get('epochs', 0) <= 0:
            raise ValueError("Training epochs must be positive")
        if training_config.get('batch_size', 0) <= 0:
            raise ValueError("Batch size must be positive")
        
        # Validate privacy config
        privacy_config = self.config['privacy']
        if privacy_config.get('epsilon', 0) <= 0:
            raise ValueError("Privacy epsilon must be positive")
        
        logger.info("Configuration validation passed")
        return True
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
        logger.info("Configuration updated")
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses original path.
        """
        save_path = Path(path) if path else self.config_path
        
        try:
            with open(save_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            raise IOError(f"Failed to save configuration: {e}")


# Global configuration instance
config = ConfigLoader() 