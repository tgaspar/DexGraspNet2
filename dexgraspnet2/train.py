"""
Training entry point for DexGraspNet2.

This script provides a command-line interface for training
grasp prediction models.

Usage:
    python -m dexgraspnet2.train --config configs/network/train_dex_ours.yaml
    python -m dexgraspnet2.train --exp_name my_experiment --max_iter 10000
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from dexgraspnet2.configs.model_config import ModelConfig
from dexgraspnet2.configs.training_config import TrainingConfig, load_legacy_config
from dexgraspnet2.training.trainer import Trainer
from dexgraspnet2.training.callbacks import (
    CheckpointCallback,
    LoggingCallback,
    WandbCallback,
)
from dexgraspnet2.utils.config_utils import set_seed
from dexgraspnet2.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DexGraspNet2 grasp prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )

    # Experiment settings (override config)
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=None,
        help="Maximum training iterations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (scenes per batch)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )

    # Model settings
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["graspness_diffusion", "graspness_isa", "graspness_cvae"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["sparseconv", "sparse_glob_conv"],
        help="Backbone network type",
    )

    # Data settings
    parser.add_argument(
        "--train_split",
        type=str,
        default=None,
        help="Training data split",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="Robot hand type",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        choices=["realsense", "kinect"],
        help="Camera type",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory for training data",
    )

    # Resume training
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path to resume training",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (e.g., 'cuda:0', 'cpu')",
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> tuple:
    """
    Build training and model configs from arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (TrainingConfig, ModelConfig).
    """
    # Load base config if provided
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Try new format first, then legacy
        try:
            training_config = TrainingConfig.from_yaml(config_path)
            model_config = ModelConfig.from_yaml(config_path)
        except Exception:
            training_config = load_legacy_config(config_path)
            from dexgraspnet2.configs.model_config import load_legacy_model_config
            model_config = load_legacy_model_config(config_path)
    else:
        # Use defaults
        training_config = TrainingConfig()
        model_config = ModelConfig()

    # Override with command-line arguments
    if args.exp_name is not None:
        training_config.exp_name = args.exp_name
    if args.max_iter is not None:
        training_config.max_iter = args.max_iter
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    if args.lr is not None:
        training_config.lr = args.lr
    if args.train_split is not None:
        training_config.train_split = args.train_split
    if args.ckpt is not None:
        training_config.ckpt = args.ckpt

    # Data config overrides
    if args.robot is not None:
        training_config.data.robot = args.robot
    if args.camera is not None:
        training_config.data.camera = args.camera

    # Model config overrides
    if args.model_type is not None:
        model_config.type = args.model_type
    if args.backbone is not None:
        model_config.backbone.name = args.backbone

    return training_config, model_config


def build_callbacks(args: argparse.Namespace) -> List:
    """
    Build training callbacks.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of callback instances.
    """
    callbacks = [
        LoggingCallback(log_every=100),
        CheckpointCallback(keep_last=5),
    ]

    # W&B logging
    if args.wandb and not args.no_wandb:
        callbacks.append(WandbCallback(disabled=False))
    elif not args.no_wandb:
        # Default: enable for non-temp experiments
        callbacks.append(WandbCallback(disabled=False))

    return callbacks


def main():
    """Main training entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    logger.info("DexGraspNet2 Training")
    logger.info(f"Arguments: {args}")

    try:
        # Build configs
        training_config, model_config = build_configs(args)

        logger.info(f"Experiment: {training_config.exp_name}")
        logger.info(f"Model type: {model_config.type}")

        # Set random seed
        set_seed(training_config.seed)

        # Build callbacks
        callbacks = build_callbacks(args)

        # Parse device
        import torch
        if args.device is not None:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create trainer
        trainer = Trainer(
            training_config=training_config,
            model_config=model_config,
            callbacks=callbacks,
            device=device,
            data_root=args.data_root,
        )

        # Train
        trainer.train()

        logger.info("Training completed successfully")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
