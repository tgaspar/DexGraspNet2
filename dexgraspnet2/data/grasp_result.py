"""
Data structures for representing grasp predictions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class GraspPose:
    """
    Represents a single grasp pose for a dexterous hand.

    Attributes:
        translation: Wrist position in world/camera frame (3,).
        rotation: Wrist orientation as a 3x3 rotation matrix (3, 3).
        joint_angles: Joint angles in radians, ordered by HandConfig.joint_names.
        score: Combined confidence score (graspness + log_prob).
        graspness: Predicted graspability score at the seed point.
        log_prob: Log probability from the diffusion model.
        seed_point: The 3D point used as the conditioning seed.
        object_id: Object instance ID if available (-1 for unknown).
    """

    translation: np.ndarray
    rotation: np.ndarray
    joint_angles: np.ndarray
    score: float = 0.0
    graspness: float = 0.0
    log_prob: float = 0.0
    seed_point: Optional[np.ndarray] = None
    object_id: int = -1

    def __post_init__(self):
        """Validate shapes after initialization."""
        if self.translation.shape != (3,):
            raise ValueError(
                f"Translation must be (3,), got {self.translation.shape}"
            )
        if self.rotation.shape != (3, 3):
            raise ValueError(
                f"Rotation must be (3, 3), got {self.rotation.shape}"
            )
        if self.seed_point is not None and self.seed_point.shape != (3,):
            raise ValueError(
                f"Seed point must be (3,), got {self.seed_point.shape}"
            )

    def to_dict(self) -> Dict:
        """
        Convert to dictionary format.

        Returns:
            Dictionary with grasp pose data.
        """
        result = {
            "translation": self.translation.tolist(),
            "rotation": self.rotation.tolist(),
            "joint_angles": self.joint_angles.tolist(),
            "score": self.score,
            "graspness": self.graspness,
            "log_prob": self.log_prob,
            "object_id": self.object_id,
        }
        if self.seed_point is not None:
            result["seed_point"] = self.seed_point.tolist()
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "GraspPose":
        """
        Create GraspPose from dictionary.

        Args:
            data: Dictionary with grasp pose data.

        Returns:
            GraspPose instance.
        """
        return cls(
            translation=np.array(data["translation"]),
            rotation=np.array(data["rotation"]),
            joint_angles=np.array(data["joint_angles"]),
            score=data.get("score", 0.0),
            graspness=data.get("graspness", 0.0),
            log_prob=data.get("log_prob", 0.0),
            seed_point=np.array(data["seed_point"]) if "seed_point" in data else None,
            object_id=data.get("object_id", -1),
        )

    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> "GraspPose":
        """
        Apply a rigid transformation to the grasp pose.

        Args:
            rotation: Rotation matrix (3, 3).
            translation: Translation vector (3,).

        Returns:
            Transformed GraspPose.
        """
        new_translation = rotation @ self.translation + translation
        new_rotation = rotation @ self.rotation
        new_seed = None
        if self.seed_point is not None:
            new_seed = rotation @ self.seed_point + translation

        return GraspPose(
            translation=new_translation,
            rotation=new_rotation,
            joint_angles=self.joint_angles.copy(),
            score=self.score,
            graspness=self.graspness,
            log_prob=self.log_prob,
            seed_point=new_seed,
            object_id=self.object_id,
        )


@dataclass
class GraspResult:
    """
    Collection of grasp predictions for a scene.

    Attributes:
        grasps: List of GraspPose predictions, sorted by score (best first).
        point_cloud: The input point cloud (N, 3).
        graspness_map: Per-point graspness scores (N,).
        objectness_map: Per-point objectness predictions (N,).
        metadata: Additional metadata (hand name, checkpoint, etc.).
    """

    grasps: List[GraspPose]
    point_cloud: Optional[np.ndarray] = None
    graspness_map: Optional[np.ndarray] = None
    objectness_map: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of grasps."""
        return len(self.grasps)

    def __getitem__(self, idx: int) -> GraspPose:
        """Get grasp by index."""
        return self.grasps[idx]

    def __iter__(self):
        """Iterate over grasps."""
        return iter(self.grasps)

    def top_k(self, k: int) -> "GraspResult":
        """
        Get top-k grasps by score.

        Args:
            k: Number of grasps to return.

        Returns:
            New GraspResult with only top-k grasps.
        """
        return GraspResult(
            grasps=self.grasps[:k],
            point_cloud=self.point_cloud,
            graspness_map=self.graspness_map,
            objectness_map=self.objectness_map,
            metadata=self.metadata.copy(),
        )

    def filter_by_object(self, object_id: int) -> "GraspResult":
        """
        Filter grasps by object ID.

        Args:
            object_id: Object ID to filter by.

        Returns:
            New GraspResult with only grasps for the specified object.
        """
        filtered = [g for g in self.grasps if g.object_id == object_id]
        return GraspResult(
            grasps=filtered,
            point_cloud=self.point_cloud,
            graspness_map=self.graspness_map,
            objectness_map=self.objectness_map,
            metadata=self.metadata.copy(),
        )

    def best(self) -> Optional[GraspPose]:
        """
        Get the best grasp by score.

        Returns:
            Best GraspPose or None if no grasps.
        """
        return self.grasps[0] if self.grasps else None

    def transform_to_world(
        self, extrinsics: np.ndarray
    ) -> "GraspResult":
        """
        Transform all grasps from camera frame to world frame.

        Args:
            extrinsics: Camera extrinsics matrix (4, 4).

        Returns:
            New GraspResult with grasps in world frame.
        """
        rotation = extrinsics[:3, :3]
        translation = extrinsics[:3, 3]

        transformed_grasps = [
            g.transform(rotation, translation) for g in self.grasps
        ]

        # Transform point cloud if available
        new_pc = None
        if self.point_cloud is not None:
            new_pc = (rotation @ self.point_cloud.T).T + translation

        return GraspResult(
            grasps=transformed_grasps,
            point_cloud=new_pc,
            graspness_map=self.graspness_map,
            objectness_map=self.objectness_map,
            metadata=self.metadata.copy(),
        )

    def save(self, path: str) -> None:
        """
        Save grasp result to numpy file.

        Args:
            path: Output file path (.npz).
        """
        # Convert grasps to arrays
        if len(self.grasps) > 0:
            translations = np.stack([g.translation for g in self.grasps])
            rotations = np.stack([g.rotation for g in self.grasps])
            joint_angles = np.stack([g.joint_angles for g in self.grasps])
            scores = np.array([g.score for g in self.grasps])
            graspness_scores = np.array([g.graspness for g in self.grasps])
            log_probs = np.array([g.log_prob for g in self.grasps])
            object_ids = np.array([g.object_id for g in self.grasps])
        else:
            translations = np.zeros((0, 3))
            rotations = np.zeros((0, 3, 3))
            joint_angles = np.zeros((0, 0))
            scores = np.zeros(0)
            graspness_scores = np.zeros(0)
            log_probs = np.zeros(0)
            object_ids = np.zeros(0, dtype=int)

        save_dict = {
            "translations": translations,
            "rotations": rotations,
            "joint_angles": joint_angles,
            "scores": scores,
            "graspness_scores": graspness_scores,
            "log_probs": log_probs,
            "object_ids": object_ids,
        }

        if self.point_cloud is not None:
            save_dict["point_cloud"] = self.point_cloud
        if self.graspness_map is not None:
            save_dict["graspness_map"] = self.graspness_map
        if self.objectness_map is not None:
            save_dict["objectness_map"] = self.objectness_map

        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path: str) -> "GraspResult":
        """
        Load grasp result from numpy file.

        Args:
            path: Input file path (.npz).

        Returns:
            GraspResult instance.
        """
        data = np.load(path)

        grasps = []
        num_grasps = len(data["translations"])
        for i in range(num_grasps):
            grasps.append(
                GraspPose(
                    translation=data["translations"][i],
                    rotation=data["rotations"][i],
                    joint_angles=data["joint_angles"][i],
                    score=float(data["scores"][i]),
                    graspness=float(data["graspness_scores"][i]) if "graspness_scores" in data else 0.0,
                    log_prob=float(data["log_probs"][i]) if "log_probs" in data else 0.0,
                    object_id=int(data["object_ids"][i]) if "object_ids" in data else -1,
                )
            )

        return cls(
            grasps=grasps,
            point_cloud=data.get("point_cloud"),
            graspness_map=data.get("graspness_map"),
            objectness_map=data.get("objectness_map"),
        )
