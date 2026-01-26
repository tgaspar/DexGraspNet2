"""
Training pipeline for DexGraspNet2.

This module provides a clean, production-ready training interface
for the grasp prediction model.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

from dexgraspnet2.configs.model_config import ModelConfig, load_legacy_model_config
from dexgraspnet2.configs.training_config import (
    TrainingConfig,
    load_legacy_config,
)
from dexgraspnet2.data.dataset import (
    GraspNetDataset,
    InfiniteLoader,
    create_data_loaders,
    minkowski_collate_fn,
)
from dexgraspnet2.models.graspness_model import GraspnessModel
from dexgraspnet2.training.callbacks import Callback, CallbackList
from dexgraspnet2.training.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator for DexGraspNet2 models.

    Handles the full training loop including:
    - Model and optimizer setup
    - Data loading
    - Training and validation loops
    - Checkpointing
    - Logging via callbacks

    Args:
        training_config: Training configuration.
        model_config: Model architecture configuration.
        callbacks: List of training callbacks.
        device: Device to train on (default: auto-detect).
        data_root: Root directory for training data.

    Example:
        >>> from dexgraspnet2.training import Trainer
        >>> from dexgraspnet2.configs import TrainingConfig, ModelConfig
        >>>
        >>> training_config = TrainingConfig(exp_name="my_exp", max_iter=10000)
        >>> model_config = ModelConfig(type="graspness_diffusion")
        >>>
        >>> trainer = Trainer(training_config, model_config)
        >>> trainer.train()
    """

    def __init__(
        self,
        training_config: Union[TrainingConfig, Dict, str, Path],
        model_config: Optional[Union[ModelConfig, Dict, str, Path]] = None,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
        data_root: str = "data",
    ):
        """Initialize the trainer."""
        # Load configs
        self._training_config = self._load_training_config(training_config)
        self._model_config = self._load_model_config(model_config, training_config)

        # Setup device
        self._device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self._data_root = data_root

        # Initialize components
        self._setup_experiment_dir()
        self._setup_model()
        self._setup_optimizer()
        self._setup_data()

        # Callbacks
        self._callbacks = CallbackList(callbacks or [])

        # Training state
        self._current_iter = 0
        self._metrics = MetricsTracker()

        logger.info(
            f"Initialized Trainer: exp={self._training_config.exp_name}, "
            f"device={self._device}, max_iter={self._training_config.max_iter}"
        )

    def _load_training_config(
        self, config: Union[TrainingConfig, Dict, str, Path]
    ) -> TrainingConfig:
        """Load and validate training configuration."""
        if isinstance(config, TrainingConfig):
            return config
        elif isinstance(config, dict):
            return TrainingConfig.from_dict(config)
        elif isinstance(config, (str, Path)):
            # Try new format first, fall back to legacy
            try:
                return TrainingConfig.from_yaml(config)
            except Exception:
                return load_legacy_config(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

    def _load_model_config(
        self,
        model_config: Optional[Union[ModelConfig, Dict, str, Path]],
        training_config: Union[TrainingConfig, Dict, str, Path],
    ) -> ModelConfig:
        """Load and validate model configuration."""
        if model_config is None:
            # Try to extract from training config if it's a file
            if isinstance(training_config, (str, Path)):
                try:
                    return load_legacy_model_config(training_config)
                except Exception:
                    pass
            return ModelConfig()

        if isinstance(model_config, ModelConfig):
            return model_config
        elif isinstance(model_config, dict):
            return ModelConfig.from_dict(model_config)
        elif isinstance(model_config, (str, Path)):
            return ModelConfig.from_yaml(model_config)
        else:
            raise TypeError(f"Unsupported config type: {type(model_config)}")

    def _setup_experiment_dir(self):
        """Create experiment directory structure."""
        exp_name = self._training_config.exp_name
        self._exp_dir = Path("experiments") / exp_name
        self._log_dir = self._exp_dir / "log"
        self._ckpt_dir = self._exp_dir / "ckpt"

        self._exp_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(exist_ok=True)
        self._ckpt_dir.mkdir(exist_ok=True)

        # Save configs
        self._training_config.to_yaml(self._exp_dir / "training_config.yaml")
        self._model_config.to_yaml(self._exp_dir / "model_config.yaml")

        logger.info(f"Experiment directory: {self._exp_dir}")

    def _setup_model(self):
        """Initialize the model."""
        # Ensure voxel_size is set
        model_dict = self._model_config.to_legacy_dict()
        model_dict["voxel_size"] = self._training_config.data.voxel_size

        # Add loss weights
        model_dict["weight"] = {
            "objectness": self._training_config.weight.objectness,
            "graspness": self._training_config.weight.graspness,
            "diffusion": self._training_config.weight.diffusion,
            "joint": self._training_config.weight.joint,
            "euc": self._training_config.weight.euc,
            "quat": self._training_config.weight.quat,
        }

        self._model = GraspnessModel(model_dict)
        self._model.to(self._device)

        logger.info(
            f"Model initialized: {sum(p.numel() for p in self._model.parameters()):,} parameters"
        )

    def _setup_optimizer(self):
        """Initialize optimizer and scheduler."""
        self._optimizer = Adam(
            self._model.parameters(),
            lr=self._training_config.lr,
        )
        self._scheduler = CosineAnnealingLR(
            self._optimizer,
            T_max=self._training_config.max_iter,
            eta_min=self._training_config.lr_min,
        )

    def _setup_data(self):
        """Initialize data loaders."""
        self._train_loader, self._val_loaders = create_data_loaders(
            self._training_config,
            data_root=self._data_root,
        )

        logger.info(
            f"Data loaders initialized: train={len(self._train_loader.dataset.views)} views, "
            f"val_splits={self._training_config.val_split}"
        )

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        load_optimizer: bool = True,
    ) -> int:
        """
        Load model and optimizer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            load_optimizer: Whether to restore optimizer state.

        Returns:
            Iteration number from checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Load model weights
        self._model.load_state_dict(ckpt["model"])

        # Load optimizer state
        if load_optimizer and "optimizer" in ckpt:
            self._optimizer.load_state_dict(ckpt["optimizer"])

        # Restore iteration counter
        self._current_iter = ckpt.get("iter", 0)

        # Step scheduler to correct position
        for _ in range(self._current_iter):
            self._scheduler.step()

        logger.info(f"Loaded checkpoint at iteration {self._current_iter}")
        return self._current_iter

    def save_checkpoint(self, iteration: Optional[int] = None):
        """
        Save model and optimizer checkpoint.

        Args:
            iteration: Iteration number (default: current iteration).
        """
        iteration = iteration or self._current_iter

        ckpt_path = self._ckpt_dir / f"ckpt_{iteration}.pth"
        torch.save(
            {
                "model": self._model.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "iter": iteration,
            },
            ckpt_path,
        )
        logger.debug(f"Saved checkpoint to {ckpt_path}")

    def _train_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Execute a single training step.

        Args:
            batch: Dictionary of batch data.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        self._optimizer.zero_grad()

        # Move data to device
        batch = {k: v.to(self._device) for k, v in batch.items()}

        # Forward pass
        loss, result_dict = self._model(batch)

        # Backward pass
        loss.backward()

        # Handle NaN gradients
        for p in self._model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                logger.warning("NaN gradient detected, zeroing")
                p.grad.zero_()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self._model.parameters(),
            self._training_config.grad_clip,
        )

        # Optimizer step
        self._optimizer.step()
        self._scheduler.step()

        return loss, result_dict

    def _validate(self) -> Dict[str, Dict[str, float]]:
        """
        Run validation on all validation splits.

        Returns:
            Dictionary mapping split names to metrics.
        """
        val_results = {}

        self._model.eval()
        with torch.no_grad():
            for split, loader in zip(
                self._training_config.val_split, self._val_loaders
            ):
                result_dicts = []

                for _ in range(self._training_config.val_num):
                    batch = loader.get()
                    batch = {k: v.to(self._device) for k, v in batch.items()}
                    _, result_dict = self._model(batch)
                    result_dicts.append(result_dict)

                # Aggregate results
                val_results[split] = {
                    k: torch.cat(
                        [
                            d[k] if len(d[k].shape) else d[k][None]
                            for d in result_dicts
                        ]
                    ).mean().item()
                    for k in result_dicts[0].keys()
                }

        self._model.train()
        return val_results

    def train(self):
        """
        Run the full training loop.

        This method runs training from the current iteration to max_iter,
        handling logging, validation, and checkpointing.
        """
        config = self._training_config

        # Load checkpoint if specified
        if config.ckpt is not None:
            self.load_checkpoint(config.ckpt)

        # Notify callbacks
        self._callbacks.on_train_begin(self)

        self._model.train()

        logger.info(f"Starting training from iteration {self._current_iter}")

        try:
            for it in trange(self._current_iter, config.max_iter, desc="Training"):
                self._current_iter = it

                # Get batch and train
                batch = self._train_loader.get()
                loss, result_dict = self._train_step(batch)

                # Update metrics
                metrics = {k: v.mean().item() for k, v in result_dict.items()}
                self._metrics.update(metrics)

                # Callbacks
                self._callbacks.on_batch_end(self, metrics)

                # Logging
                if it % config.log_every == 0:
                    self._callbacks.on_log(self, metrics, "train", it)

                # Checkpointing
                if (it + 1) % config.save_every == 0:
                    self.save_checkpoint(it + 1)
                    self._callbacks.on_checkpoint(self, it + 1)

                # Validation
                if it % config.val_every == 0:
                    val_results = self._validate()
                    for split, split_metrics in val_results.items():
                        self._callbacks.on_log(self, split_metrics, split, it)
                    self._callbacks.on_validation_end(self, val_results)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        finally:
            # Final checkpoint
            self.save_checkpoint()
            self._callbacks.on_train_end(self)

        logger.info("Training completed")

    @property
    def model(self) -> nn.Module:
        """Get the model."""
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer."""
        return self._optimizer

    @property
    def current_iter(self) -> int:
        """Get current iteration."""
        return self._current_iter

    @property
    def device(self) -> torch.device:
        """Get training device."""
        return self._device

    @property
    def config(self) -> TrainingConfig:
        """Get training configuration."""
        return self._training_config

    @property
    def model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self._model_config
