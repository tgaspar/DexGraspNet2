"""
Model configuration module for DexGraspNet2.

Provides dataclasses for configuring neural network architectures.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DiffusionConfig:
    """
    Configuration for the diffusion model.

    Based on DDPM (Denoising Diffusion Probabilistic Models) with
    velocity prediction for improved sample quality.

    Attributes:
        scheduler_type: Scheduler class name ('DDPMScheduler').
        num_train_timesteps: Number of diffusion timesteps during training.
        num_inference_timesteps: Number of steps during inference (can be less).
        beta_start: Starting value for noise schedule beta.
        beta_end: Ending value for noise schedule beta.
        beta_schedule: Type of beta schedule ('linear', 'scaled_linear', 'squaredcos_cap_v2').
        prediction_type: What the model predicts ('epsilon', 'v_prediction').
        clip_sample: Whether to clip samples to [-1, 1].
        ode: Whether to use ODE (deterministic) sampling instead of SDE.
        log_prob_type: Type of log probability estimation ('estimate', 'accurate_cont', None).
        rot_type: Rotation representation type ('svd', 'sixd', 'quat', 'aa', 'euler').
    """

    scheduler_type: str = "DDPMScheduler"
    num_train_timesteps: int = 1000
    num_inference_timesteps: int = 200
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "v_prediction"
    clip_sample: bool = True
    ode: bool = False
    log_prob_type: Optional[str] = None
    rot_type: str = "svd"

    def __post_init__(self):
        """Validate diffusion configuration."""
        valid_schedulers = ["DDPMScheduler"]
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(
                f"scheduler_type must be one of {valid_schedulers}, "
                f"got {self.scheduler_type}"
            )

        valid_pred_types = ["epsilon", "v_prediction", "sample"]
        if self.prediction_type not in valid_pred_types:
            raise ValueError(
                f"prediction_type must be one of {valid_pred_types}, "
                f"got {self.prediction_type}"
            )

        valid_rot_types = ["svd", "sixd", "quat", "aa", "euler"]
        if self.rot_type not in valid_rot_types:
            raise ValueError(
                f"rot_type must be one of {valid_rot_types}, got {self.rot_type}"
            )

    def to_scheduler_dict(self) -> Dict:
        """
        Convert to scheduler initialization kwargs.

        Returns:
            Dictionary for initializing a diffusers scheduler.
        """
        return {
            "num_train_timesteps": self.num_train_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            "prediction_type": self.prediction_type,
            "clip_sample": self.clip_sample,
        }


@dataclass
class BackboneConfig:
    """
    Configuration for the point cloud backbone network.

    The default backbone is a sparse convolutional UNet (MinkUNet14D)
    based on MinkowskiEngine.

    Attributes:
        name: Backbone type ('sparseconv', 'sparse_glob_conv').
        in_channels: Number of input channels (typically 3 for XYZ or 1 for occupancy).
        out_channels: Feature dimension output by the backbone.
    """

    name: str = "sparseconv"
    in_channels: int = 3
    out_channels: int = 512

    def __post_init__(self):
        """Validate backbone configuration."""
        valid_backbones = ["sparseconv", "sparse_glob_conv"]
        if self.name not in valid_backbones:
            raise ValueError(
                f"backbone name must be one of {valid_backbones}, got {self.name}"
            )


@dataclass
class MLPConfig:
    """
    Configuration for MLP networks used in the model.

    Attributes:
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function ('relu', 'leaky_relu', 'mish', 'elu', 'tanh').
        use_layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    activation: str = "mish"
    use_layer_norm: bool = False

    def __post_init__(self):
        """Validate MLP configuration."""
        valid_activations = ["relu", "leaky_relu", "mish", "elu", "tanh"]
        if self.activation not in valid_activations:
            raise ValueError(
                f"activation must be one of {valid_activations}, "
                f"got {self.activation}"
            )


@dataclass
class ModelConfig:
    """
    Configuration for the full grasp prediction model.

    The DexGraspNet 2.0 model consists of:
    1. Backbone: Extracts point-wise features from sparse point cloud
    2. Graspness head: Predicts objectness and graspness scores
    3. Diffusion model: Generates (translation, rotation) conditioned on features
    4. Joint MLP: Predicts joint angles (either via diffusion or separate MLP)

    Attributes:
        type: Model architecture type ('graspness_diffusion', 'graspness_isa', 'graspness_cvae').
        feature_dim: Dimension of point features from backbone.
        joint_num: Number of robot hand joints (DoF).
        trans_scale: Scale factor for translation predictions.
        joint_scale: Scale factor for joint angle predictions.
        dist_joint: Whether to predict joints via diffusion (1) or separate MLP (0).
        voxel_size: Voxel size for sparse convolution (should match data config).
        backbone: Backbone network configuration.
        diffusion: Diffusion model configuration.
        policy_mlp: MLP configuration for diffusion policy network.
    """

    type: str = "graspness_diffusion"
    feature_dim: int = 512
    joint_num: int = 16
    trans_scale: float = 25.0  # ktrans from paper Table 11
    joint_scale: float = 1.0  # Match OURS checkpoint (scale=1)
    dist_joint: int = 0  # 0 = separate joint MLP, 1 = joints in diffusion
    voxel_size: float = 0.005
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    policy_mlp: MLPConfig = field(default_factory=MLPConfig)

    def __post_init__(self):
        """Validate model configuration."""
        valid_types = ["graspness_diffusion", "graspness_isa", "graspness_cvae"]
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got {self.type}")

        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        if self.joint_num <= 0:
            raise ValueError(f"joint_num must be positive, got {self.joint_num}")

        # Convert nested dicts to dataclasses if needed
        if isinstance(self.backbone, dict):
            self.backbone = BackboneConfig(**self.backbone)
        if isinstance(self.diffusion, dict):
            self.diffusion = DiffusionConfig(**self.diffusion)
        if isinstance(self.policy_mlp, dict):
            self.policy_mlp = MLPConfig(**self.policy_mlp)

        # Ensure backbone output matches feature_dim
        if self.backbone.out_channels != self.feature_dim:
            logger.warning(
                f"backbone.out_channels ({self.backbone.out_channels}) doesn't match "
                f"feature_dim ({self.feature_dim}), updating backbone config"
            )
            self.backbone.out_channels = self.feature_dim

    @property
    def rot_dim(self) -> int:
        """Get rotation representation dimension based on rot_type."""
        rot_dims = {
            "svd": 9,
            "sixd": 6,
            "quat": 4,
            "aa": 3,
            "euler": 3,
        }
        return rot_dims[self.diffusion.rot_type]

    @property
    def diffusion_output_dim(self) -> int:
        """Get total diffusion output dimension (rot + trans + optional joints)."""
        output_dim = self.rot_dim + 3  # rotation + translation
        if self.dist_joint:
            output_dim += self.joint_num  # joint angles
        return output_dim

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "ModelConfig":
        """
        Load model configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            ModelConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Extract model-specific config if nested
        if "model" in config_dict:
            config_dict = config_dict["model"]

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ModelConfig":
        """
        Create ModelConfig from a dictionary.

        Handles nested configs and legacy config formats.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            ModelConfig instance.
        """
        config_dict = config_dict.copy()

        # Handle backbone config
        backbone_dict = config_dict.pop("backbone", None)
        if backbone_dict is None:
            backbone_name = config_dict.pop("backbone", "sparseconv")
            backbone_dict = {"name": backbone_name}
        if isinstance(backbone_dict, str):
            backbone_dict = {"name": backbone_dict}
        backbone_config = BackboneConfig(**backbone_dict)

        # Handle diffusion config
        diffusion_dict = config_dict.pop("diffusion", {})
        # Handle legacy scheduler format
        if "scheduler" in diffusion_dict:
            scheduler_dict = diffusion_dict.pop("scheduler")
            diffusion_dict.update(scheduler_dict)
        if "scheduler_type" not in diffusion_dict and "scheduler" not in diffusion_dict:
            diffusion_dict["scheduler_type"] = "DDPMScheduler"
        diffusion_config = DiffusionConfig(**diffusion_dict)

        # Handle MLP config
        mlp_dict = config_dict.pop("policy_mlp", {})
        if "hidden_layers_dim" in mlp_dict:
            mlp_dict["hidden_dims"] = mlp_dict.pop("hidden_layers_dim")
        if "act" in mlp_dict:
            mlp_dict["activation"] = mlp_dict.pop("act")
        mlp_config = MLPConfig(**mlp_dict) if mlp_dict else MLPConfig()

        # Remove backbone_parameters if present (handled differently now)
        config_dict.pop("backbone_parameters", None)
        config_dict.pop("weight", None)  # Handled in training config

        # Filter to valid fields only
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        return cls(
            backbone=backbone_config,
            diffusion=diffusion_config,
            policy_mlp=mlp_config,
            **filtered_dict,
        )

    def to_yaml(self, config_path: Union[str, Path]) -> None:
        """
        Save model configuration to a YAML file.

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
            "type": self.type,
            "feature_dim": self.feature_dim,
            "joint_num": self.joint_num,
            "trans_scale": self.trans_scale,
            "joint_scale": self.joint_scale,
            "dist_joint": self.dist_joint,
            "voxel_size": self.voxel_size,
            "backbone": {
                "name": self.backbone.name,
                "in_channels": self.backbone.in_channels,
                "out_channels": self.backbone.out_channels,
            },
            "diffusion": {
                "scheduler_type": self.diffusion.scheduler_type,
                "num_train_timesteps": self.diffusion.num_train_timesteps,
                "num_inference_timesteps": self.diffusion.num_inference_timesteps,
                "beta_start": self.diffusion.beta_start,
                "beta_end": self.diffusion.beta_end,
                "beta_schedule": self.diffusion.beta_schedule,
                "prediction_type": self.diffusion.prediction_type,
                "clip_sample": self.diffusion.clip_sample,
                "ode": self.diffusion.ode,
                "log_prob_type": self.diffusion.log_prob_type,
                "rot_type": self.diffusion.rot_type,
            },
            "policy_mlp": {
                "hidden_dims": self.policy_mlp.hidden_dims,
                "activation": self.policy_mlp.activation,
                "use_layer_norm": self.policy_mlp.use_layer_norm,
            },
        }

    def to_legacy_dict(self) -> Dict:
        """
        Convert to legacy format for compatibility with old checkpoints.

        Returns:
            Dictionary in legacy format.
        """
        return {
            "type": self.type,
            "backbone": self.backbone.name,
            "feature_dim": self.feature_dim,
            "joint_num": self.joint_num,
            "trans_scale": self.trans_scale,
            "joint_scale": self.joint_scale,
            "dist_joint": self.dist_joint,
            "voxel_size": self.voxel_size,
            "backbone_parameters": {
                self.backbone.name: {}
            },
            "diffusion": {
                "scheduler_type": self.diffusion.scheduler_type,
                "scheduler": {
                    "num_train_timesteps": self.diffusion.num_train_timesteps,
                    "beta_start": self.diffusion.beta_start,
                    "beta_end": self.diffusion.beta_end,
                    "beta_schedule": self.diffusion.beta_schedule,
                    "prediction_type": self.diffusion.prediction_type,
                    "clip_sample": self.diffusion.clip_sample,
                },
                "num_inference_timesteps": self.diffusion.num_inference_timesteps,
                "ode": self.diffusion.ode,
                "log_prob_type": self.diffusion.log_prob_type,
                "rot_type": self.diffusion.rot_type,
            },
        }


def load_legacy_model_config(yaml_file: Union[str, Path]) -> ModelConfig:
    """
    Load model configuration from a legacy format YAML file.

    This handles the original DexGraspNet2 config format and converts
    it to the new ModelConfig dataclass.

    Args:
        yaml_file: Path to legacy YAML configuration.

    Returns:
        ModelConfig instance.
    """
    yaml_file = Path(yaml_file)
    with open(yaml_file, "r") as f:
        raw_config = yaml.safe_load(f)

    model_dict = raw_config.get("model", raw_config)
    return ModelConfig.from_dict(model_dict)
