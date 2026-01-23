"""
Production-grade grasp prediction pipeline for DexGraspNet2.

This module provides a clean, hand-agnostic interface for predicting
dexterous grasp poses using the DexGraspNet 2.0 method.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from dexgraspnet2.configs.hand_config import HandConfig
from dexgraspnet2.data.grasp_result import GraspPose, GraspResult
from dexgraspnet2.utils.point_cloud import voxelize_point_cloud, to_voxel_center

logger = logging.getLogger(__name__)


class DotDict(dict):
    """Dictionary that allows attribute-style access."""

    def __getattr__(self, item):
        if item in self.keys():
            return self[item]
        return None

    def __setattr__(self, key, value):
        self[key] = value


def to_dot_dict(dic: dict) -> DotDict:
    """Recursively convert dict to DotDict."""
    for k in dic.keys():
        if isinstance(dic[k], dict):
            dic[k] = to_dot_dict(dic[k])
    return DotDict(dic)


class GraspPredictor:
    """
    Production-grade grasp pose predictor for dexterous hands.

    This class implements the two-stage DexGraspNet 2.0 method:
    1. Seed Point Proposal: Identify graspable points in the scene
    2. Grasp Pose Generation: Sample grasp poses using diffusion model

    The predictor is hand-agnostic - configure for different hands by
    providing the appropriate HandConfig and checkpoint.

    Example:
        >>> hand_config = HandConfig.leap_hand()
        >>> predictor = GraspPredictor(
        ...     checkpoint_path="checkpoints/leap_hand.pth",
        ...     hand_config=hand_config,
        ...     device="cuda:0"
        ... )
        >>> result = predictor.predict(point_cloud, segmentation)
        >>> best_grasp = result.best()
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        hand_config: Optional[HandConfig] = None,
        device: str = "cuda:0",
        voxel_size: float = 0.005,
    ):
        """
        Initialize the grasp predictor.

        Args:
            checkpoint_path: Path to the trained model checkpoint (.pth file).
            hand_config: Hand configuration. If None, will attempt to infer
                from checkpoint or use LEAP hand defaults.
            device: Device to run inference on (e.g., 'cuda:0', 'cpu').
            voxel_size: Voxel size for point cloud processing.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.voxel_size = voxel_size

        # Load configuration from checkpoint directory
        self.config = self._load_config()

        # Set up hand configuration
        if hand_config is not None:
            self.hand_config = hand_config
        else:
            # Default to LEAP hand if config indicates it
            robot_name = self.config.data.get("robot", "leap_hand")
            if robot_name == "leap_hand":
                self.hand_config = HandConfig.leap_hand()
            else:
                raise ValueError(
                    f"Unknown robot '{robot_name}' in config. "
                    "Please provide hand_config explicitly."
                )

        # Validate DoF matches
        model_dof = self.config.model.joint_num
        if model_dof != self.hand_config.num_dofs:
            raise ValueError(
                f"Model expects {model_dof} DoF but hand has {self.hand_config.num_dofs}"
            )

        # Load model
        self.model = self._load_model()

        logger.info(
            f"GraspPredictor initialized for {self.hand_config.name} "
            f"({self.hand_config.num_dofs} DoF) on {self.device}"
        )

    def _load_config(self) -> DotDict:
        """Load model configuration from checkpoint directory."""
        config_path = self.checkpoint_path.parent.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                "Expected checkpoint structure: exp_dir/ckpt/checkpoint.pth"
            )

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return to_dot_dict(config_dict)

    def _load_model(self) -> torch.nn.Module:
        """Load and initialize the model from checkpoint."""
        # Import here to avoid circular imports
        from src.network.model import get_model

        # Create model from config
        model = get_model(self.config.model)
        model.config.voxel_size = self.voxel_size

        # Load weights
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)

        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        return model

    @torch.no_grad()
    def predict(
        self,
        point_cloud: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        num_grasps: int = 1024,
        graspness_scale: float = 5.0,
        use_category_sampling: bool = False,
        edge_mask: Optional[np.ndarray] = None,
        graspness_ratio: float = 0.01,
    ) -> GraspResult:
        """
        Predict grasp poses for the given point cloud.

        Args:
            point_cloud: Input point cloud (N, 3) in camera frame.
            segmentation: Object segmentation mask (N,). Each unique non-zero
                value represents a different object. If None, all points
                are treated as one object.
            num_grasps: Number of grasp proposals to generate.
            graspness_scale: Weight for graspness in final score computation.
                Higher values prefer points with high graspness.
            use_category_sampling: If True, sample grasps uniformly across
                object categories. If False, sample globally.
            edge_mask: Binary mask (N,) marking edge points to avoid.
            graspness_ratio: Ratio of top graspness points to consider.

        Returns:
            GraspResult containing predicted grasps sorted by score.
        """
        # Validate input
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError(f"Expected point_cloud shape (N, 3), got {point_cloud.shape}")

        if segmentation is None:
            segmentation = np.ones(len(point_cloud), dtype=np.int64)

        # Prepare input data
        data = self._prepare_input(point_cloud, segmentation, edge_mask)

        # Run inference
        outputs = self._run_inference(
            data,
            num_grasps=num_grasps,
            graspness_scale=graspness_scale,
            use_category_sampling=use_category_sampling,
            graspness_ratio=graspness_ratio,
        )

        # Convert to GraspResult
        result = self._create_result(
            outputs,
            point_cloud,
            segmentation,
        )

        return result

    def _prepare_input(
        self,
        point_cloud: np.ndarray,
        segmentation: np.ndarray,
        edge_mask: Optional[np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """Prepare input data for the model."""
        # Voxelize point cloud
        data = voxelize_point_cloud(point_cloud, self.voxel_size)

        # Add segmentation
        data["seg"] = torch.tensor(segmentation, dtype=torch.long).unsqueeze(0)

        # Add edge mask if provided
        if edge_mask is not None:
            data["edge"] = torch.tensor(edge_mask, dtype=torch.long).unsqueeze(0)

        # Move to device
        data = {k: v.to(self.device) for k, v in data.items()}

        return data

    def _run_inference(
        self,
        data: Dict[str, torch.Tensor],
        num_grasps: int,
        graspness_scale: float,
        use_category_sampling: bool,
        graspness_ratio: float,
    ) -> Dict[str, torch.Tensor]:
        """Run model inference."""
        # Get edge tensor if available
        edge = data.pop("edge", None)

        # Run model sample method
        outputs = self.model.sample(
            data,
            k=num_grasps,
            cate=use_category_sampling,
            graspness_scale=graspness_scale,
            allow_fail=True,
            with_point=True,
            with_score_parts=True,
            edge=edge,
            ratio=graspness_ratio,
        )

        # Unpack outputs
        rotation, translation, joints, score, obj_indices, graspness, log_prob, seed_points = outputs

        return {
            "rotation": rotation.cpu(),
            "translation": translation.cpu(),
            "joints": joints.cpu(),
            "score": score.cpu(),
            "obj_indices": obj_indices.cpu(),
            "graspness": graspness.cpu(),
            "log_prob": log_prob.cpu(),
            "seed_points": seed_points.cpu(),
        }

    def _create_result(
        self,
        outputs: Dict[str, torch.Tensor],
        point_cloud: np.ndarray,
        segmentation: np.ndarray,
    ) -> GraspResult:
        """Convert model outputs to GraspResult."""
        # Extract tensors (batch dim 0)
        rotations = outputs["rotation"][0].numpy()
        translations = outputs["translation"][0].numpy()
        joints = outputs["joints"][0].numpy()
        scores = outputs["score"][0].numpy()
        graspness = outputs["graspness"][0].numpy()
        log_probs = outputs["log_prob"][0].numpy()
        obj_indices = outputs["obj_indices"][0].numpy()
        seed_points = outputs["seed_points"].numpy()

        # Create GraspPose objects
        grasps = []
        num_grasps = len(rotations)

        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]

        for idx in sorted_indices:
            grasp = GraspPose(
                translation=translations[idx],
                rotation=rotations[idx],
                joint_angles=joints[idx],
                score=float(scores[idx]),
                graspness=float(graspness[idx]),
                log_prob=float(log_probs[idx]),
                seed_point=seed_points[idx] if idx < len(seed_points) else None,
                object_id=int(obj_indices[idx]),
            )
            grasps.append(grasp)

        # Create result
        result = GraspResult(
            grasps=grasps,
            point_cloud=point_cloud,
            metadata={
                "hand_name": self.hand_config.name,
                "num_dofs": self.hand_config.num_dofs,
                "checkpoint": str(self.checkpoint_path),
                "num_requested": num_grasps,
            },
        )

        return result

    @torch.no_grad()
    def get_graspness_map(
        self,
        point_cloud: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get per-point graspness and objectness predictions.

        This is useful for visualization and debugging.

        Args:
            point_cloud: Input point cloud (N, 3).

        Returns:
            Tuple of (objectness, graspness) arrays, each (N,).
            objectness: Binary classification (0=background, 1=object).
            graspness: Continuous graspability score (higher=more graspable).
        """
        # Prepare input
        data = voxelize_point_cloud(point_cloud, self.voxel_size)
        data = {k: v.to(self.device) for k, v in data.items()}

        # Extract features
        feature = self.model.get_feature(data)

        # Predict scores
        objectness, graspness = self.model.pred_score(feature)

        # Convert to numpy
        objectness = objectness[0].argmax(dim=-1).cpu().numpy()
        graspness = graspness[0].cpu().numpy()

        return objectness, graspness

    def get_joint_names(self) -> list:
        """Get ordered list of joint names."""
        return self.hand_config.joint_names.copy()

    def clamp_joints(self, joints: np.ndarray) -> np.ndarray:
        """
        Clamp joint values to valid range.

        Args:
            joints: Joint values (..., num_dofs).

        Returns:
            Clamped joint values.
        """
        lower = np.array(self.hand_config.joint_lower_limits)
        upper = np.array(self.hand_config.joint_upper_limits)
        return np.clip(joints, lower, upper)


def load_predictor(
    checkpoint_path: str,
    hand_name: str = "leap_hand",
    device: str = "cuda:0",
) -> GraspPredictor:
    """
    Convenience function to load a GraspPredictor.

    Args:
        checkpoint_path: Path to model checkpoint.
        hand_name: Name of hand configuration to use.
        device: Device for inference.

    Returns:
        Configured GraspPredictor instance.
    """
    # Get hand config
    if hand_name == "leap_hand":
        hand_config = HandConfig.leap_hand()
    else:
        raise ValueError(f"Unknown hand name: {hand_name}")

    return GraspPredictor(
        checkpoint_path=checkpoint_path,
        hand_config=hand_config,
        device=device,
    )
