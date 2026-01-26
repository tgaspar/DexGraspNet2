"""
Training callbacks for DexGraspNet2.

Callbacks provide hooks into the training loop for logging,
checkpointing, and custom behavior.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from dexgraspnet2.training.trainer import Trainer

logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Base class for training callbacks.

    Callbacks are called at various points during training to perform
    logging, checkpointing, early stopping, etc.
    """

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass

    def on_batch_end(
        self,
        trainer: "Trainer",
        metrics: Dict[str, float],
    ) -> None:
        """Called after each training batch."""
        pass

    def on_validation_end(
        self,
        trainer: "Trainer",
        val_results: Dict[str, Dict[str, float]],
    ) -> None:
        """Called after validation."""
        pass

    def on_log(
        self,
        trainer: "Trainer",
        metrics: Dict[str, float],
        split: str,
        iteration: int,
    ) -> None:
        """Called when metrics should be logged."""
        pass

    def on_checkpoint(
        self,
        trainer: "Trainer",
        iteration: int,
    ) -> None:
        """Called when a checkpoint is saved."""
        pass


class CallbackList:
    """
    Container for multiple callbacks.

    Dispatches callback events to all registered callbacks.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """Initialize with list of callbacks."""
        self._callbacks = callbacks or []

    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self._callbacks.append(callback)

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Dispatch to all callbacks."""
        for cb in self._callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer: "Trainer") -> None:
        """Dispatch to all callbacks."""
        for cb in self._callbacks:
            cb.on_train_end(trainer)

    def on_batch_end(
        self,
        trainer: "Trainer",
        metrics: Dict[str, float],
    ) -> None:
        """Dispatch to all callbacks."""
        for cb in self._callbacks:
            cb.on_batch_end(trainer, metrics)

    def on_validation_end(
        self,
        trainer: "Trainer",
        val_results: Dict[str, Dict[str, float]],
    ) -> None:
        """Dispatch to all callbacks."""
        for cb in self._callbacks:
            cb.on_validation_end(trainer, val_results)

    def on_log(
        self,
        trainer: "Trainer",
        metrics: Dict[str, float],
        split: str,
        iteration: int,
    ) -> None:
        """Dispatch to all callbacks."""
        for cb in self._callbacks:
            cb.on_log(trainer, metrics, split, iteration)

    def on_checkpoint(
        self,
        trainer: "Trainer",
        iteration: int,
    ) -> None:
        """Dispatch to all callbacks."""
        for cb in self._callbacks:
            cb.on_checkpoint(trainer, iteration)


class LoggingCallback(Callback):
    """
    Callback for logging metrics to console.

    Args:
        log_every: Log to console every N iterations.
    """

    def __init__(self, log_every: int = 100):
        """Initialize the callback."""
        self._log_every = log_every
        self._last_logged = -1

    def on_log(
        self,
        trainer: "Trainer",
        metrics: Dict[str, float],
        split: str,
        iteration: int,
    ) -> None:
        """Log metrics to console."""
        if split == "train" and iteration - self._last_logged < self._log_every:
            return

        self._last_logged = iteration

        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if not k.startswith("_")
        )
        logger.info(f"[{split}] iter={iteration}: {metrics_str}")


class CheckpointCallback(Callback):
    """
    Callback for managing checkpoints.

    Args:
        keep_last: Number of recent checkpoints to keep.
    """

    def __init__(self, keep_last: int = 5):
        """Initialize the callback."""
        self._keep_last = keep_last
        self._checkpoints: List[Path] = []

    def on_checkpoint(
        self,
        trainer: "Trainer",
        iteration: int,
    ) -> None:
        """Track and clean up checkpoints."""
        ckpt_path = trainer._ckpt_dir / f"ckpt_{iteration}.pth"
        self._checkpoints.append(ckpt_path)

        # Remove old checkpoints
        while len(self._checkpoints) > self._keep_last:
            old_ckpt = self._checkpoints.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()
                logger.debug(f"Removed old checkpoint: {old_ckpt}")


class WandbCallback(Callback):
    """
    Callback for logging to Weights & Biases.

    Args:
        project: W&B project name.
        entity: W&B entity (username or team).
        config: Configuration to log.
        log_code: Whether to log source code.
        disabled: Whether to disable W&B logging.
    """

    def __init__(
        self,
        project: str = "DexGraspNet2",
        entity: Optional[str] = None,
        config: Optional[Dict] = None,
        log_code: bool = True,
        disabled: bool = False,
    ):
        """Initialize the callback."""
        self._project = project
        self._entity = entity
        self._config = config
        self._log_code = log_code
        self._disabled = disabled
        self._run = None

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Initialize W&B run."""
        if self._disabled or trainer.config.exp_name == "temp":
            return

        try:
            import wandb

            # Build config
            config = self._config or {}
            config.update(trainer.config.to_dict())
            config.update({"model": trainer.model_config.to_dict()})

            self._run = wandb.init(
                project=self._project,
                entity=self._entity,
                name=trainer.config.exp_name,
                config=config,
                resume="allow",
            )

            if self._log_code:
                wandb.run.log_code(root="./dexgraspnet2")

            logger.info(f"W&B run initialized: {wandb.run.url}")

        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
            self._disabled = True
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self._disabled = True

    def on_train_end(self, trainer: "Trainer") -> None:
        """Finish W&B run."""
        if self._disabled or self._run is None:
            return

        try:
            import wandb
            wandb.finish()
        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {e}")

    def on_log(
        self,
        trainer: "Trainer",
        metrics: Dict[str, float],
        split: str,
        iteration: int,
    ) -> None:
        """Log metrics to W&B."""
        if self._disabled or self._run is None:
            return

        try:
            import wandb

            # Prefix metrics with split
            prefixed = {f"{split}/{k}": v for k, v in metrics.items()}
            wandb.log(prefixed, step=iteration)

        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on validation metric.

    Args:
        monitor: Metric name to monitor.
        patience: Number of validations to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' or 'max' - whether to minimize or maximize metric.
    """

    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """Initialize the callback."""
        self._monitor = monitor
        self._patience = patience
        self._min_delta = min_delta
        self._mode = mode
        self._best_value = float("inf") if mode == "min" else float("-inf")
        self._counter = 0

    def on_validation_end(
        self,
        trainer: "Trainer",
        val_results: Dict[str, Dict[str, float]],
    ) -> None:
        """Check for improvement."""
        # Get the first validation split
        first_split = list(val_results.keys())[0]
        current_value = val_results[first_split].get(self._monitor)

        if current_value is None:
            return

        improved = False
        if self._mode == "min":
            improved = current_value < self._best_value - self._min_delta
        else:
            improved = current_value > self._best_value + self._min_delta

        if improved:
            self._best_value = current_value
            self._counter = 0
            logger.info(
                f"Validation {self._monitor} improved to {current_value:.4f}"
            )
        else:
            self._counter += 1
            if self._counter >= self._patience:
                logger.info(
                    f"Early stopping triggered after {self._counter} validations "
                    f"without improvement"
                )
                raise KeyboardInterrupt("Early stopping")
