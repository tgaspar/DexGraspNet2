"""
Configuration utilities for DexGraspNet2.

Provides helper functions for loading and manipulating configurations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class DotDict(dict):
    """
    Dictionary with attribute-style access.

    Allows accessing dictionary keys as attributes:
        config.key instead of config['key']

    Example:
        >>> d = DotDict({'a': 1, 'b': {'c': 2}})
        >>> d.a
        1
        >>> d.b.c
        2
    """

    def __getattr__(self, item: str) -> Any:
        """Get item as attribute."""
        if item in self.keys():
            return self[item]
        return None

    def __setattr__(self, key: str, value: Any):
        """Set item as attribute."""
        self[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get with default value."""
        if key in self.keys():
            return self[key]
        return default


def to_dot_dict(d: Dict) -> DotDict:
    """
    Recursively convert dictionary to DotDict.

    Args:
        d: Dictionary to convert.

    Returns:
        DotDict with nested DotDicts.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = to_dot_dict(v)
    return DotDict(d)


def to_dict(d: Dict) -> Dict:
    """
    Recursively convert DotDict back to regular dict.

    Args:
        d: DotDict or dict to convert.

    Returns:
        Regular dictionary.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = to_dict(v)
        else:
            result[k] = v
    return result


def load_yaml(yaml_path: Union[str, Path]) -> Dict:
    """
    Load YAML configuration file.

    Args:
        yaml_path: Path to YAML file.

    Returns:
        Dictionary with configuration.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(config: Dict, yaml_path: Union[str, Path]):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary.
        yaml_path: Output path.
    """
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert DotDict to regular dict
    if isinstance(config, DotDict):
        config = to_dict(config)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two configuration dictionaries.

    Values in override take precedence over base.

    Args:
        base: Base configuration.
        override: Override configuration.

    Returns:
        Merged configuration.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Random seed set to {seed}")
