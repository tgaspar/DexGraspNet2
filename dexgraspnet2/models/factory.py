"""
Model factory for DexGraspNet2.

Provides functions to create models from configurations.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from dexgraspnet2.configs.model_config import ModelConfig, load_legacy_model_config
from dexgraspnet2.models.graspness_model import GraspnessModel

logger = logging.getLogger(__name__)


def create_model(
    config: Union[ModelConfig, Dict, str, Path],
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Create a model from configuration.

    Args:
        config: Model configuration (ModelConfig, dict, or path to YAML).
        device: Device to load model on.

    Returns:
        Initialized model.

    Example:
        >>> from dexgraspnet2.models import create_model
        >>> model = create_model("configs/network/train_dex_ours.yaml")
        >>> model = create_model(ModelConfig(type="graspness_diffusion"))
    """
    # Load config
    if isinstance(config, (str, Path)):
        config = ModelConfig.from_yaml(config)
    elif isinstance(config, dict):
        config = ModelConfig.from_dict(config)

    # Create model
    model = GraspnessModel(config.to_legacy_dict())

    # Move to device
    if device is not None:
        model.to(device)

    logger.info(f"Created model: {config.type}")
    return model


def load_model(
    checkpoint_path: Union[str, Path],
    config: Optional[Union[ModelConfig, Dict, str, Path]] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load a model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: Model configuration (optional if checkpoint contains it).
        device: Device to load model on.
        strict: Whether to strictly enforce state dict matching.

    Returns:
        Loaded model.

    Example:
        >>> model = load_model("pretrained/model.pth")
        >>> model = load_model("checkpoint.pth", config=ModelConfig())
    """
    checkpoint_path = Path(checkpoint_path)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Get config
    if config is None:
        # Try to load from checkpoint or experiment directory
        exp_dir = checkpoint_path.parent.parent
        config_path = exp_dir / "model_config.yaml"

        if config_path.exists():
            config = ModelConfig.from_yaml(config_path)
        elif "config" in ckpt:
            config = ModelConfig.from_dict(ckpt["config"])
        else:
            # Try legacy config path
            legacy_config_path = exp_dir / "config.yaml"
            if legacy_config_path.exists():
                config = load_legacy_model_config(legacy_config_path)
            else:
                raise ValueError(
                    f"No config found for checkpoint. "
                    f"Please provide config argument."
                )

    # Create model
    model = create_model(config, device=None)

    # Load weights
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=strict)

    # Move to device
    if device is not None:
        model.to(device)

    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def get_model(config: Dict) -> nn.Module:
    """
    Create model from legacy config dict.

    This function provides backward compatibility with the
    original src/network/model.py interface.

    Args:
        config: Legacy model configuration dictionary.

    Returns:
        Initialized model.
    """
    return GraspnessModel(config)
