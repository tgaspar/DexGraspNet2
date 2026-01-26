"""
Training configuration module for DexGraspNet2.

Provides dataclasses for configuring the training pipeline.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """
    Configuration for training data loading.

    Attributes:
        num_points: Number of points to sample from point cloud.
        voxel_size: Voxel size for sparse convolution (meters).
        camera: Camera type used for depth images ('realsense' or 'kinect').
        robot: Robot hand type ('leap_hand', 'gripper', etc.).
        graspness_data: Path prefix for graspness annotation data.
        fraction: Fraction of grasp data to use (1 = all).
        scene_fraction: Scene subsampling factor (1 = all scenes).
        sample_total: Total grasp samples to consider per scene.
        k: Number of grasps to select per batch.
        max_point_dis: Maximum distance from grasp point to nearest cloud point.
        resample: Whether to resample grasps per object.
        render: Whether to use rendered depth images.
    """

    num_points: int = 40000
    voxel_size: float = 0.005
    camera: str = "realsense"
    robot: str = "leap_hand"
    graspness_data: str = "dex_graspness_new"
    fraction: int = 1
    scene_fraction: int = 1
    sample_total: int = 256
    k: int = 64
    max_point_dis: float = 0.02
    resample: bool = True
    render: bool = True  # Use depth_gt/ instead of depth/

    def __post_init__(self):
        """Validate data configuration."""
        if self.num_points <= 0:
            raise ValueError(f"num_points must be positive, got {self.num_points}")
        if self.voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {self.voxel_size}")
        if self.k > self.sample_total:
            raise ValueError(f"k ({self.k}) must be <= sample_total ({self.sample_total})")


@dataclass
class LossWeights:
    """
    Loss weight configuration.

    The total loss is computed as:
        L = lambda_o * L_objectness + lambda_g * L_graspness
            + lambda_d * L_diffusion + lambda_t * L_joint

    Attributes:
        objectness: Weight for objectness classification loss.
        graspness: Weight for graspness regression loss.
        diffusion: Weight for diffusion model loss.
        joint: Weight for joint angle prediction loss.
        euc: Weight for euclidean (ISA model) loss.
        quat: Weight for quaternion (ISA model) loss.
    """

    objectness: float = 1.0
    graspness: float = 1.0
    diffusion: float = 10.0
    joint: float = 1.0
    euc: float = 1.0
    quat: float = 1.0


@dataclass
class TrainingConfig:
    """
    Configuration for training pipeline.

    Attributes:
        exp_name: Experiment name for logging and checkpointing.
        seed: Random seed for reproducibility.
        max_iter: Maximum training iterations.
        batch_size: Number of scenes per batch.
        num_workers: Number of data loading workers.
        lr: Initial learning rate.
        lr_min: Minimum learning rate for cosine annealing.
        grad_clip: Gradient clipping threshold.
        log_every: Logging frequency (iterations).
        save_every: Checkpoint saving frequency (iterations).
        val_every: Validation frequency (iterations).
        val_num: Number of validation batches per evaluation.
        train_split: Training data split name.
        val_split: List of validation data split names.
        ckpt: Path to checkpoint for resuming training.
        data: Data loading configuration.
        weight: Loss weight configuration.
    """

    exp_name: str = "experiment"
    seed: int = 42
    max_iter: int = 50000
    batch_size: int = 8
    num_workers: int = 8
    lr: float = 1e-3
    lr_min: float = 1e-7
    grad_clip: float = 10.0
    log_every: int = 100
    save_every: int = 2000
    val_every: int = 1000
    val_num: int = 10
    train_split: str = "train-part"
    val_split: List[str] = field(default_factory=lambda: ["val"])
    ckpt: Optional[str] = None
    data: DataConfig = field(default_factory=DataConfig)
    weight: LossWeights = field(default_factory=LossWeights)

    def __post_init__(self):
        """Validate training configuration."""
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive, got {self.grad_clip}")

        # Convert nested dicts to dataclasses if needed
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.weight, dict):
            self.weight = LossWeights(**self.weight)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "TrainingConfig":
        """
        Load training configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            TrainingConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TrainingConfig":
        """
        Create TrainingConfig from a dictionary.

        Handles nested configs and legacy config formats.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            TrainingConfig instance.
        """
        # Extract data config
        data_dict = config_dict.pop("data", {})
        data_config = DataConfig(**data_dict) if data_dict else DataConfig()

        # Extract weight config (may be nested under different keys)
        weight_dict = config_dict.pop("weight", {})
        if not weight_dict and "model" in config_dict:
            weight_dict = config_dict.get("model", {}).get("weight", {})
        weight_config = LossWeights(**weight_dict) if weight_dict else LossWeights()

        # Filter out model-specific configs that don't belong here
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        return cls(data=data_config, weight=weight_config, **filtered_dict)

    def to_yaml(self, config_path: Union[str, Path]) -> None:
        """
        Save training configuration to a YAML file.

        Args:
            config_path: Path where to save the configuration.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)

    def to_dict(self) -> Dict:
        """
        Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "exp_name": self.exp_name,
            "seed": self.seed,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "lr": self.lr,
            "lr_min": self.lr_min,
            "grad_clip": self.grad_clip,
            "log_every": self.log_every,
            "save_every": self.save_every,
            "val_every": self.val_every,
            "val_num": self.val_num,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "ckpt": self.ckpt,
            "data": {
                "num_points": self.data.num_points,
                "voxel_size": self.data.voxel_size,
                "camera": self.data.camera,
                "robot": self.data.robot,
                "graspness_data": self.data.graspness_data,
                "fraction": self.data.fraction,
                "scene_fraction": self.data.scene_fraction,
                "sample_total": self.data.sample_total,
                "k": self.data.k,
                "max_point_dis": self.data.max_point_dis,
                "resample": self.data.resample,
                "render": self.data.render,
            },
            "weight": {
                "objectness": self.weight.objectness,
                "graspness": self.weight.graspness,
                "diffusion": self.weight.diffusion,
                "joint": self.weight.joint,
            },
        }


def load_legacy_config(yaml_file: Union[str, Path]) -> TrainingConfig:
    """
    Load configuration from a legacy format YAML file.

    This handles the original DexGraspNet2 config format and converts
    it to the new TrainingConfig dataclass.

    Args:
        yaml_file: Path to legacy YAML configuration.

    Returns:
        TrainingConfig instance.
    """
    yaml_file = Path(yaml_file)
    with open(yaml_file, "r") as f:
        raw_config = yaml.safe_load(f)

    # Map legacy keys to new structure
    training_dict = {
        "exp_name": raw_config.get("exp_name", "experiment"),
        "seed": raw_config.get("seed", 42),
        "max_iter": raw_config.get("max_iter", 50000),
        "batch_size": raw_config.get("batch_size", 8),
        "num_workers": raw_config.get("num_workers", 8),
        "lr": raw_config.get("lr", 1e-3),
        "lr_min": raw_config.get("lr_min", 1e-7),
        "grad_clip": raw_config.get("grad_clip", 10.0),
        "log_every": raw_config.get("log_every", 100),
        "save_every": raw_config.get("save_every", 2000),
        "val_every": raw_config.get("val_every", 1000),
        "val_num": raw_config.get("val_num", 10),
        "train_split": raw_config.get("train_split", "train-part"),
        "val_split": raw_config.get("val_split", ["val"]),
        "ckpt": raw_config.get("ckpt"),
    }

    # Extract data config from legacy format
    data_config = raw_config.get("data", {})
    training_dict["data"] = DataConfig(
        num_points=data_config.get("num_points", 40000),
        voxel_size=data_config.get("voxel_size", 0.005),
        camera=raw_config.get("camera", "realsense"),
        robot=data_config.get("robot", "leap_hand"),
        graspness_data=data_config.get("graspness_data", "dex_graspness"),
        fraction=data_config.get("fraction", 1),
        scene_fraction=data_config.get("scene_fraction", 1),
        sample_total=data_config.get("sample_total", 256),
        k=data_config.get("k", 64),
        max_point_dis=data_config.get("max_point_dis", 0.02),
        resample=data_config.get("resample", True),
        render=data_config.get("render", False),
    )

    # Extract weight config from legacy format (nested under model)
    model_config = raw_config.get("model", {})
    weight_config = model_config.get("weight", {})
    training_dict["weight"] = LossWeights(
        objectness=weight_config.get("objectness", 1.0),
        graspness=weight_config.get("graspness", 1.0),
        diffusion=weight_config.get("diffusion", 10.0),
        joint=weight_config.get("joint", 1.0),
        euc=weight_config.get("euc", 1.0),
        quat=weight_config.get("quat", 1.0),
    )

    return TrainingConfig(**training_dict)
