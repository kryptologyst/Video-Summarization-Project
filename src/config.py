"""
Configuration management for the video summarization project.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the video summarization project."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path("config/default.yaml")
        self._config = self._load_default_config()
        
        if self.config_path.exists():
            self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "video_summarizer": {
                "scene_threshold": 0.5,
                "max_frames": 20,
                "method": "hybrid",
                "device": "auto"
            },
            "output": {
                "directory": "output",
                "video_fps": 2,
                "image_format": "jpg"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "deep_learning": {
                "model_name": "google/vit-base-patch16-224",
                "batch_size": 1,
                "max_length": 512
            }
        }
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
            elif self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {self.config_path.suffix}")
                return
            
            self._merge_config(file_config)
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def _merge_config(self, file_config: Dict[str, Any]) -> None:
        """Merge file configuration with default configuration."""
        def merge_dicts(default: Dict, override: Dict) -> Dict:
            for key, value in override.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_dicts(default[key], value)
                else:
                    default[key] = value
            return default
        
        self._config = merge_dicts(self._config, file_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration (uses current config_path if None)
        """
        save_path = path or self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if save_path.suffix.lower() in ['.yaml', '.yml']:
                with open(save_path, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
            elif save_path.suffix.lower() == '.json':
                with open(save_path, 'w') as f:
                    json.dump(self._config, f, indent=2)
            else:
                logger.warning(f"Unsupported save format: {save_path.suffix}")
                return
            
            logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
