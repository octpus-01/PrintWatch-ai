"""Configuration loader for PrintWatch AI.

Loads YAML configuration based on PW_ENV environment variable.
Supports environment-specific configs (cpu_dev, gpu_train, etc.)
and allows PW_DEVICE to override the device setting.
"""

import os
import yaml
from pathlib import Path


def load(env=None):
    """Load configuration from YAML file.
    
    Args:
        env: Environment name (e.g., 'cpu_dev', 'gpu_train'). 
             Defaults to PW_ENV environment variable or 'cpu_dev'.
    
    Returns:
        dict: Configuration dictionary with all settings.
    """
    env = env or os.getenv("PW_ENV", "cpu_dev")
    
    config_path = Path(__file__).parent.parent.parent / "configs" / f"{env}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    cfg["device"] = os.getenv("PW_DEVICE", cfg.get("device", "cpu"))
    
    return cfg


def get_device(cfg):
    """Get device string from config.
    
    Args:
        cfg: Configuration dictionary.
    
    Returns:
        str: Device string ('cpu' or 'cuda').
    """
    return cfg.get("device", "cpu")


def is_smoke_mode(cfg):
    """Check if smoke mode is enabled.
    
    Args:
        cfg: Configuration dictionary.
    
    Returns:
        bool: True if smoke mode is enabled.
    """
    return cfg.get("train", {}).get("smoke", False) or cfg.get("search", {}).get("smoke", False)
