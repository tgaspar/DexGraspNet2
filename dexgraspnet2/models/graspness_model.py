"""
Graspness-based grasp prediction model for DexGraspNet2.

This module implements the main grasp prediction model that combines:
1. Sparse convolutional backbone for point-wise feature extraction
2. Graspness/objectness prediction heads
3. Diffusion model for (translation, rotation) generation
4. Optional MLP for joint angle prediction
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

try:
    from pytorch3d import transforms as pttf
    from pytorch3d.ops import sample_farthest_points, ball_query
except ImportError:
    pttf = None

from dexgraspnet2.configs.model_config import DiffusionConfig, ModelConfig
from dexgraspnet2.models.backbones import get_backbone, get_feature
from dexgraspnet2.models.diffusion import GaussianDiffusion1D, MLPDenoiser

logger = logging.getLogger(__name__)


def proper_svd(rot: torch.Tensor) -> torch.Tensor:
    """
    Orthogonalize rotation matrix using SVD.

    Ensures proper rotation matrix (det = +1) by correcting
    reflections when necessary.

    Args:
        rot: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3, 3) orthogonalized rotation matrices.
    """
    u, _, vh = torch.linalg.svd(rot)
    rot_ortho = u @ vh
    # Correct reflections (ensure det = +1)
    det = torch.linalg.det(rot_ortho)
    correction = torch.eye(3, device=rot.device, dtype=rot.dtype)
    correction = correction.unsqueeze(0).expand(*rot.shape[:-2], 3, 3).clone()
    correction[..., 2, 2] = det.sign()
    return u @ correction @ vh


def to_voxel_center(pc: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """
    Calculate the center of voxel corresponding to each point.

    Args:
        pc: (..., 3) point coordinates.
        voxel_size: Size of voxels in meters.

    Returns:
        (..., 3) voxel center coordinates.
    """
    return (torch.floor(pc / voxel_size) + 0.5) * voxel_size


# Use the nflows ResidualNet directly to ensure exact compatibility with the
# original implementation. The original src/network/condition.py imports from nflows.
# Our previous custom implementation had architectural differences (extra ReLUs,
# post-activation instead of pre-activation) that caused wrong joint predictions.
from nflows.nn.nets.resnet import ResidualNet


class ConditionalTransform(nn.Module):
    """
    Conditional transformation using nflows ResidualNet.

    A MLP with 2 residual blocks for predicting joint angles or other outputs.
    Uses the exact same architecture as src/network/condition.py.

    Architecture (from nflows):
        input -> Linear -> [ResidualBlock x num_blocks] -> Linear -> output

    Where each ResidualBlock uses pre-activation:
        x -> ReLU -> Linear -> ReLU -> Dropout -> Linear -> + x
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
    ):
        """Initialize the transformation."""
        super().__init__()
        self.net = ResidualNet(
            in_features=input_dim,
            out_features=output_dim,
            hidden_features=hidden_dim,
            num_blocks=num_blocks,
            dropout_probability=0.0,
            use_batch_norm=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformation."""
        return self.net(x)


class GraspnessModel(nn.Module):
    """
    Main grasp prediction model combining graspness and diffusion.

    This model implements the DexGraspNet 2.0 architecture:
    1. Backbone extracts point-wise features from sparse point cloud
    2. Graspness head predicts objectness (binary) and graspness (continuous)
    3. Diffusion model generates (translation, rotation) conditioned on features
    4. Joint MLP (or diffusion) predicts hand joint angles

    Args:
        config: Model configuration (ModelConfig or dict).

    Attributes:
        backbone: Point cloud feature extraction network.
        graspable: Linear head for objectness/graspness prediction.
        diffusion: Diffusion model for pose generation.
        rot_type: Rotation representation type.
    """

    # Rotation representation dimensions
    ROT_DIMS = {
        "svd": 9,
        "euler": 3,
        "quat": 4,
        "aa": 3,
        "sixd": 6,
    }

    def __init__(self, config: Union[ModelConfig, Dict]):
        """Initialize the grasp prediction model."""
        super().__init__()

        # Handle config types
        if isinstance(config, dict):
            self.config = config
            # Convert to namespace-like for attribute access
            config = _DictConfig(config)
        else:
            self.config = config

        # Store key parameters
        self._feature_dim = getattr(config, "feature_dim", 512)
        self._joint_num = getattr(config, "joint_num", 16)
        self._model_type = getattr(config, "type", "graspness_diffusion")
        self._dist_joint = getattr(config, "dist_joint", 1)
        self._trans_scale = getattr(config, "trans_scale", 10.0)
        self._joint_scale = getattr(config, "joint_scale", 10.0)
        self._voxel_size = getattr(config, "voxel_size", 0.005)

        # Initialize backbone
        backbone_name = getattr(config, "backbone", "sparseconv")
        if isinstance(backbone_name, str):
            backbone_config = getattr(config, "backbone_parameters", {})
            if backbone_config and backbone_name in backbone_config:
                backbone_config = backbone_config[backbone_name]
            else:
                backbone_config = {}
        else:
            backbone_name = backbone_name.name
            backbone_config = {}

        self._backbone_name = backbone_name
        self.backbone = get_backbone(
            backbone_name=backbone_name,
            feature_dim=self._feature_dim,
            backbone_config=backbone_config,
        )

        # Graspness/objectness prediction head
        # Output: [objectness_0, objectness_1, graspness]
        self.graspable = nn.Linear(self._feature_dim, 3)

        # Initialize model-specific components
        if self._model_type == "graspness_diffusion":
            self._init_diffusion_model(config)
        elif self._model_type == "graspness_isa":
            self._init_isa_model()
        elif self._model_type == "graspness_cvae":
            self._init_cvae_model(config)
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

        # Joint MLP for non-distributed joint prediction
        if not self._dist_joint:
            self.joint_mlp = ConditionalTransform(
                self._feature_dim + 9 + 3, self._joint_num
            )

        # Loss functions
        self.objectness_loss = nn.CrossEntropyLoss(reduction="none")
        self.graspness_loss = nn.SmoothL1Loss(reduction="none")
        self.joint_loss = nn.SmoothL1Loss(reduction="none")

        # Warn about voxel_size
        if not hasattr(config, "voxel_size"):
            warnings.warn("voxel_size not set, using 0.005 as default")

        # Register normalization buffers (for potential future use)
        self.register_buffer("mean", torch.zeros(9 + 3 + self._joint_num))
        self.register_buffer("std", torch.ones(9 + 3 + self._joint_num))

        logger.info(
            f"Initialized GraspnessModel: type={self._model_type}, "
            f"backbone={backbone_name}, feature_dim={self._feature_dim}, "
            f"joint_num={self._joint_num}"
        )

    def _init_diffusion_model(self, config):
        """Initialize diffusion-based grasp generation."""
        # Get diffusion config
        diff_config = getattr(config, "diffusion", {})

        # Handle different config types
        if isinstance(diff_config, DiffusionConfig):
            self.rot_type = diff_config.rot_type
        elif isinstance(diff_config, dict):
            self.rot_type = diff_config.get("rot_type", "svd")
            diff_config = DiffusionConfig(**diff_config)
        elif isinstance(diff_config, _DictConfig):
            # Convert _DictConfig to dict then to DiffusionConfig
            self.rot_type = getattr(diff_config, "rot_type", "svd")
            diff_dict = {k: getattr(diff_config, k) for k in diff_config}
            # Handle nested scheduler config (legacy format)
            if "scheduler" in diff_dict and isinstance(
                diff_dict["scheduler"], _DictConfig
            ):
                scheduler_dict = {
                    k: getattr(diff_dict["scheduler"], k)
                    for k in diff_dict["scheduler"]
                }
                diff_dict.update(scheduler_dict)
                del diff_dict["scheduler"]
            diff_config = DiffusionConfig(
                **{
                    k: v
                    for k, v in diff_dict.items()
                    if k in DiffusionConfig.__dataclass_fields__
                }
            )
        else:
            self.rot_type = getattr(diff_config, "rot_type", "svd")

        self.rot_dim = self.ROT_DIMS[self.rot_type]

        # Output dimension: rotation + translation + (optional) joints
        output_dim = self.rot_dim + 3
        if self._dist_joint:
            output_dim += self._joint_num

        # Create denoiser MLP
        hidden_layers_dim = [512, 256]
        self.policy = MLPDenoiser(
            channels=output_dim,
            feature_dim=self._feature_dim,
            hidden_layers_dim=hidden_layers_dim,
            output_dim=output_dim,
            act="mish",
        )

        # Create diffusion model
        self.diffusion = GaussianDiffusion1D(self.policy, diff_config)

    def _init_isa_model(self):
        """Initialize ISA (direct prediction) model."""
        assert self._dist_joint, "ISA model requires dist_joint=1"
        self.joint_mlp = ConditionalTransform(
            self._feature_dim,
            self._joint_num + 4 + 3,  # quat + trans + joints
        )

    def _init_cvae_model(self, config):
        """Initialize CVAE model (placeholder)."""
        assert self._dist_joint, "CVAE model requires dist_joint=1"
        # CVAE implementation would go here
        raise NotImplementedError(
            "CVAE model not yet implemented in refactored version"
        )

    def get_feature(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract point-wise features from point cloud.

        Args:
            data: Dictionary containing:
                - point_clouds: (B, N, 3) point coordinates
                - coors: Sparse coordinates for MinkowskiEngine
                - feats: Sparse features
                - quantize2original: Index mapping

        Returns:
            (B, N, C) point-wise features.
        """
        return get_feature(
            backbone_name=self._backbone_name,
            backbone=self.backbone,
            data=data,
        )

    def pred_score(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict objectness and graspness scores.

        Args:
            feature: (B, N, C) point-wise features.

        Returns:
            Tuple of:
                - objectness: (B, N, 2) binary classification logits
                - graspness: (B, N) continuous graspness scores
        """
        graspable = self.graspable(feature)
        objectness = graspable[..., :2]
        graspness = graspable[..., 2]
        return objectness, graspness

    def to_voxel_center(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Calculate voxel centers for points.

        Args:
            pc: (..., 3) point coordinates.

        Returns:
            (..., 3) voxel center coordinates.
        """
        return to_voxel_center(pc, self._voxel_size)

    def get_euc_rot(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract euclidean and rotation targets from data.

        Args:
            data: Dictionary with point_clouds, trans, rot, qpos, centers.

        Returns:
            Tuple of (euc, rot) targets for diffusion.
        """
        batch_size = data["point_clouds"].shape[0]
        centers = data["trans"]

        # Get point indices
        arange = repeat(
            torch.arange(batch_size, device=centers.device),
            "n -> (n k)",
            k=centers.shape[1],
        )
        indices = data["centers"].reshape(-1).long()
        points = data["point_clouds"][arange, indices]

        # Flatten rotation and compute relative translation
        rot = rearrange(data["rot"], "n k a b -> (n k) a b")
        trans = (
            rearrange(centers, "n k d -> (n k) d") - self.to_voxel_center(points)
        ) * self._trans_scale

        gt_joints = data["qpos"]
        euc = trans

        # Optionally include joints in euclidean part
        if self._dist_joint:
            euc = torch.cat(
                [euc, gt_joints.reshape(-1, gt_joints.shape[-1]) * self._joint_scale],
                dim=-1,
            )

        return euc, rot

    def sample_grasp(
        self,
        feature: torch.Tensor,
        seed_points: torch.Tensor,
        sample_num: int = 1,
        allow_fail: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample grasp poses from learned distribution.

        Args:
            feature: (B, C) point features at seed locations.
            seed_points: (B, 3) seed point coordinates.
            sample_num: Number of samples per seed point.
            allow_fail: Whether to allow failures (unused).

        Returns:
            Tuple of:
                - rot: (B, N, 3, 3) rotation matrices
                - trans: (B, N, 3) translations
                - joints: (B, N, J) joint angles
                - log_prob: (B, N) log probabilities
        """
        batch_size = feature.shape[0]

        if self._model_type == "graspness_diffusion":
            # Expand features for multiple samples
            feature_expanded = repeat(feature, "b c -> (b n) c", n=sample_num)

            # Sample from diffusion
            samples, log_prob = self.diffusion.sample(cond=feature_expanded)

            # Parse output
            rot_flat = samples[..., : self.rot_dim]
            euc = samples[..., self.rot_dim :].reshape(batch_size, sample_num, -1)

            # Convert rotation representation to matrix
            rot = self._rot_to_matrix(rot_flat, batch_size, sample_num)
            log_prob = log_prob.reshape(batch_size, sample_num)

        elif self._model_type == "graspness_isa":
            assert sample_num == 1, "ISA model only supports single sample"
            est = self.joint_mlp(feature)
            quat, euc = est[:, :4], est[:, 4:]
            rot = pttf.quaternion_to_matrix(quat)
            rot, euc = rot[:, None], euc[:, None]
            log_prob = torch.zeros_like(euc[..., 0])

        # Extract translation and joints
        if self._dist_joint:
            delta_trans = euc[..., :3]
            joints = euc[..., 3:] / self._joint_scale
        else:
            delta_trans = euc
            # Predict joints separately
            feature_for_joint = repeat(feature, "b c -> (b n) c", n=sample_num)
            rot_flat = rot.reshape(-1, 9)
            delta_flat = delta_trans.reshape(-1, 3)
            joints = self.joint_mlp(
                torch.cat([feature_for_joint, rot_flat, delta_flat], dim=-1)
            )
            joints = rearrange(joints, "(b n) c -> b n c", n=sample_num)
            joints = joints / self._joint_scale

        # Convert delta translation to absolute
        trans = (
            self.to_voxel_center(seed_points[:, None]) + delta_trans / self._trans_scale
        )

        return rot, trans, joints, log_prob

    def _rot_to_matrix(
        self, rot_flat: torch.Tensor, batch_size: int, sample_num: int
    ) -> torch.Tensor:
        """Convert rotation representation to matrix."""
        if self.rot_type == "svd":
            rot = proper_svd(rot_flat.reshape(-1, 3, 3))
        elif self.rot_type == "sixd":
            rot = pttf.rotation_6d_to_matrix(rot_flat)
        elif self.rot_type == "quat":
            rot = pttf.quaternion_to_matrix(rot_flat)
        elif self.rot_type == "aa":
            rot = pttf.axis_angle_to_matrix(rot_flat)
        elif self.rot_type == "euler":
            rot = pttf.euler_angles_to_matrix(rot_flat, "XYZ")
        return rot.reshape(batch_size, sample_num, 3, 3)

    def _matrix_to_rot(self, rot: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to configured representation."""
        if self.rot_type == "svd":
            return rot.reshape(-1, 9)
        elif self.rot_type == "sixd":
            return pttf.matrix_to_rotation_6d(rot)
        elif self.rot_type == "quat":
            return pttf.matrix_to_quaternion(rot)
        elif self.rot_type == "aa":
            return pttf.matrix_to_axis_angle(rot)
        elif self.rot_type == "euler":
            return pttf.matrix_to_euler_angles(rot, "XYZ")

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute training loss.

        Args:
            data: Dictionary containing:
                - point_clouds: (B, N, 3)
                - coors, feats, quantize2original: Sparse tensor data
                - objectness: (B, N) ground truth
                - graspness: (B, N) ground truth
                - trans: (B, K, 3) grasp translations
                - rot: (B, K, 3, 3) grasp rotations
                - qpos: (B, K, J) joint positions
                - centers: (B, K) grasp center indices
                - has_graspness: (B, 1) graspness data availability

        Returns:
            Tuple of:
                - loss: Scalar training loss
                - result_dict: Dictionary of individual loss components
        """
        batch_size, point_num, _ = data["point_clouds"].shape

        # Extract features
        feature = self.get_feature(data)

        # Predict objectness and graspness
        objectness, graspness = self.pred_score(feature)
        gt_objectness = data["objectness"]
        gt_graspness = data["graspness"]

        # Objectness loss (cross-entropy)
        loss_objectness = (
            self.objectness_loss(objectness.reshape(-1, 2), gt_objectness.reshape(-1))
            .reshape(batch_size, point_num)
            .mean(dim=1)
        )

        # Graspness loss (only on object points)
        loss_graspness = self.graspness_loss(
            graspness * gt_objectness, gt_graspness * gt_objectness
        ).sum(dim=1) / (gt_objectness.sum(dim=1) + 1e-6)
        loss_graspness = loss_graspness * data["has_graspness"].reshape(
            *loss_graspness.shape
        )

        # Metrics
        acc_objectness = (
            (objectness.argmax(dim=-1) == gt_objectness).float().mean(dim=-1)
        )
        abs_graspness = torch.abs(
            graspness * gt_objectness - gt_graspness * gt_objectness
        ).sum(dim=-1) / (gt_objectness.sum(dim=1) + 1e-6)

        # Get features at grasp center points
        centers = data["trans"]
        arange = repeat(
            torch.arange(batch_size, device=centers.device),
            "n -> (n k)",
            k=centers.shape[1],
        )
        indices = data["centers"].reshape(-1).long()
        sel_point_feature = feature[arange, indices]

        # Get targets
        euc, rot = self.get_euc_rot(data)

        # Model-specific loss
        if self._model_type == "graspness_diffusion":
            rot_rep = self._matrix_to_rot(rot)
            gt_goal = torch.cat([rot_rep, euc], dim=-1)
            loss_diffusion = self.diffusion(gt_goal, sel_point_feature)
        elif self._model_type == "graspness_isa":
            est = self.joint_mlp(sel_point_feature)
            est_quat, est_euc = est[:, :4], est[:, 4:]
            est_rot = pttf.quaternion_to_matrix(est_quat)
            loss_euc = (
                (est_euc - euc).abs().mean(dim=-1).reshape(batch_size, -1).mean(dim=-1)
            )
            loss_quat = (
                pttf.so3_relative_angle(est_rot, rot, eps=1e-2)
                .reshape(batch_size, -1)
                .mean(dim=-1)
            )

        # Joint loss (if not predicted via diffusion)
        gt_joints = data["qpos"]
        if not self._dist_joint:
            points = data["point_clouds"][arange, indices]
            trans = (
                rearrange(centers, "n k d -> (n k) d") - self.to_voxel_center(points)
            ) * self._trans_scale
            est_joints = self.joint_mlp(
                torch.cat([sel_point_feature, rot.reshape(-1, 9), trans], dim=-1)
            ).reshape(*gt_joints.shape)
            loss_joint = (
                self.joint_loss(est_joints, gt_joints * self._joint_scale)
                .mean(dim=-1)
                .mean(dim=-1)
            )
            abs_dis_joint = torch.abs(
                est_joints[..., 0] / self._joint_scale - gt_joints[..., 0]
            ).mean(dim=-1)

        # Compute total loss
        weight = (
            self.config.get("weight", {})
            if isinstance(self.config, dict)
            else self.config
        )
        w_obj = (
            getattr(weight, "objectness", 1.0) if hasattr(weight, "objectness") else 1.0
        )
        w_grasp = (
            getattr(weight, "graspness", 1.0) if hasattr(weight, "graspness") else 1.0
        )

        loss = w_obj * loss_objectness + w_grasp * loss_graspness

        # Build result dict
        result_dict = {
            "loss_objectness": loss_objectness,
            "loss_graspness": loss_graspness,
            "acc_objectness": acc_objectness,
            "abs_graspness": abs_graspness,
        }

        if not self._dist_joint:
            w_joint = getattr(weight, "joint", 1.0) if hasattr(weight, "joint") else 1.0
            loss = loss + w_joint * loss_joint
            result_dict["loss_joint"] = loss_joint
            result_dict["abs_dis_joint"] = abs_dis_joint

        if self._model_type == "graspness_diffusion":
            w_diff = (
                getattr(weight, "diffusion", 10.0)
                if hasattr(weight, "diffusion")
                else 10.0
            )
            loss = loss + w_diff * loss_diffusion
            result_dict["loss_diffusion"] = loss_diffusion
        elif self._model_type == "graspness_isa":
            w_euc = getattr(weight, "euc", 1.0) if hasattr(weight, "euc") else 1.0
            w_quat = getattr(weight, "quat", 1.0) if hasattr(weight, "quat") else 1.0
            loss = loss + w_euc * loss_euc + w_quat * loss_quat
            result_dict["loss_euc"] = loss_euc
            result_dict["loss_quat"] = loss_quat

        result_dict["loss"] = loss

        return loss.mean(), result_dict

    def sample_points(
        self,
        pc: torch.Tensor,
        graspable: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample seed points using farthest point sampling.

        Args:
            pc: (N, 3) point cloud.
            graspable: (N,) boolean mask of graspable points.
            k: Number of points to sample.

        Returns:
            Tuple of (seed_points, indices).
        """
        if graspable.sum() == 0:
            raise ValueError("No graspable points")
        elif graspable.sum() <= k:
            indices = torch.randint(0, graspable.sum(), (k,), device=graspable.device)[
                None
            ]
            seed_point = pc[graspable][indices[0]][None]
        else:
            seed_point, indices = sample_farthest_points(
                pc[graspable][None].contiguous(), K=k, random_start_point=True
            )
        return seed_point, indices

    @torch.no_grad()
    def sample(
        self,
        data: Dict[str, torch.Tensor],
        k: int,
        cate: bool = True,
        ratio: float = 0.005,
        graspness_scale: float = 1.0,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Sample grasp poses from the model.

        Args:
            data: Dictionary with point_clouds, coors, feats, quantize2original, seg.
            k: Number of grasps to sample.
            cate: Whether to sample uniformly across object categories.
            ratio: Top ratio of graspness scores to consider.
            graspness_scale: Weight for graspness in scoring.
            **kwargs: Additional arguments (for compatibility).

        Returns:
            List of [rot, trans, joints, score, obj_indices].
        """
        pc_cuda = data["point_clouds"]
        batch_size = pc_cuda.shape[0]

        # Extract features and predict scores
        feature = self.get_feature(data)
        objectness, graspness = self.pred_score(feature)

        # Mask graspness by objectness
        graspness = torch.where(
            objectness.argmax(dim=-1) == 1,
            graspness,
            torch.full_like(graspness, np.log(1e-3)),
        )

        features_list = []
        seed_points_list = []
        graspnesses_list = []
        obj_indices = []

        for i in range(batch_size):
            obj_indices.append([])

            if cate and "seg" in data:
                # Sample uniformly from each object
                seg = data["seg"][i]
                obj_ids = [idx for idx in torch.unique(seg).tolist() if idx != 0]
                obj_num = len(obj_ids)
                obj_k = [k // obj_num for _ in range(obj_num)]
                for j in range(k % obj_num):
                    obj_k[np.random.randint(obj_num)] += 1

                for j, obj_id in enumerate(obj_ids):
                    graspable = seg == obj_id
                    graspness_obj = graspness[i, graspable].sort(descending=True).values
                    threshold = graspness_obj[int(graspness_obj.size(0) * 0.05)]
                    graspable = (seg == obj_id) & (graspness[i] >= threshold)

                    seed_point, indices = self.sample_points(
                        pc_cuda[i], graspable, obj_k[j]
                    )
                    features_list.append(feature[i, graspable][indices][0])
                    seed_points_list.append(seed_point[0])
                    graspnesses_list.append(graspness[i, graspable][indices][0])
                    obj_indices[-1] += [obj_id] * obj_k[j]
            else:
                # Sample from top graspness points
                threshold = (
                    graspness[i]
                    .sort(descending=True)
                    .values[int(graspness[i].size(0) * ratio)]
                )
                graspable = graspness[i] > np.log(1e-2)
                if graspable.sum() == 0:
                    graspable = graspness[i] >= threshold

                seed_point, indices = self.sample_points(pc_cuda[i], graspable, k)
                features_list.append(feature[i, graspable][indices][0])
                seed_points_list.append(seed_point[0])
                graspnesses_list.append(graspness[i, graspable][indices][0])
                obj_indices[-1] += [-1] * k

        # Concatenate and sample
        features_cat = torch.cat(features_list)
        seed_points_cat = torch.cat(seed_points_list)

        rot, trans, joints, log_prob = self.sample_grasp(
            features_cat, seed_points_cat, sample_num=1
        )

        # Reshape to batch
        rot = rot.reshape(batch_size, k, 3, 3)
        trans = trans.reshape(batch_size, k, 3)
        joints = joints.reshape(batch_size, k, -1)
        log_prob = log_prob.reshape(batch_size, k)
        graspnesses_cat = torch.cat(graspnesses_list, dim=0).reshape(batch_size, k)

        # Compute score
        score = log_prob + graspnesses_cat * graspness_scale
        score = score.nan_to_num(nan=-1e6)

        obj_indices = torch.tensor(obj_indices).to(rot.device)

        return [rot, trans, joints, score, obj_indices]


class _DictConfig:
    """Helper class to access dict as attributes."""

    def __init__(self, d: Dict):
        self._keys = list(d.keys())
        for k, v in d.items():
            if isinstance(v, dict):
                v = _DictConfig(v)
            setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def __iter__(self):
        return iter(self._keys)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


# Alias for backward compatibility
GraspnessSample = GraspnessModel
