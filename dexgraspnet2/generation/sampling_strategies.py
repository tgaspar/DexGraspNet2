import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from dexgraspnet2.configs.hand_config import HandConfig

logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    translation: np.ndarray
    rotation: np.ndarray
    point: np.ndarray  # Seed point (surface or volume)


class SamplingStrategy(ABC):
    """Abstract base class for grasp sampling strategies."""

    def __init__(self, hand_config: HandConfig):
        self.hand_config = hand_config

    @abstractmethod
    def sample(
        self,
        mesh: trimesh.Trimesh,
        num_candidates: int,
        seed: Optional[int] = None,
    ) -> List[SamplingResult]:
        """
        Sample grasp poses relative to the object mesh.

        Args:
            mesh: Object mesh.
            num_candidates: Number of poses to generate.
            seed: Random seed.

        Returns:
            List of SamplingResult objects containing world-frame hand poses.
        """
        pass

    def _get_hand_tcp_transform(self) -> np.ndarray:
        """Get the transform from hand root to TCP."""
        if self.hand_config.tcp_position and self.hand_config.tcp_rotation_rpy:
            pos = self.hand_config.tcp_position
            rpy = self.hand_config.tcp_rotation_rpy
            R = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = pos
            return T
        return np.eye(4)

    def _solve_hand_pose(
        self, T_world_tcp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for hand root pose given TCP pose in world frame.
        T_world_hand = T_world_tcp @ inv(T_hand_tcp)
        """
        T_hand_tcp = self._get_hand_tcp_transform()
        T_tcp_hand = np.linalg.inv(T_hand_tcp)
        T_world_hand = T_world_tcp @ T_tcp_hand

        return T_world_hand[:3, 3], T_world_hand[:3, :3]


class SurfaceNormalStrategy(SamplingStrategy):
    """
    Samples grasps by aligning TCP Z-axis with object surface normals.
    Good for precision grasps on surface patches.
    """

    def sample(
        self,
        mesh: trimesh.Trimesh,
        num_candidates: int,
        seed: Optional[int] = None,
    ) -> List[SamplingResult]:
        if seed is not None:
            np.random.seed(seed)

        # Params
        params = self.hand_config.sampling_params or {}
        standoff_min = params.get("standoff_min", 0.08)
        standoff_max = params.get("standoff_max", 0.12)
        jitter_xy = params.get("jitter_xy", 0.0)
        jitter_rot_x = params.get("jitter_rot_x", 0.0)
        jitter_rot_y = params.get("jitter_rot_y", 0.0)
        jitter_rot_z = params.get("jitter_rot_z", np.pi)
        # Reject candidates whose outward surface normal is within this angle
        # of horizontal (i.e. pointing sideways or toward the ground). Only
        # normals tilted at least `min_normal_angle_deg` above horizontal pass.
        min_normal_angle_deg = params.get("min_normal_angle_deg", 0.0)
        min_nz = float(np.sin(np.deg2rad(min_normal_angle_deg)))

        # Oversample from the mesh surface so we can drop rejected normals and
        # still return the requested count. If the object's upper-facing area
        # is small, the caller ends up with fewer candidates — we warn rather
        # than loop forever.
        oversample_factor = 4 if min_normal_angle_deg > 0.0 else 1
        sample_n = max(num_candidates * oversample_factor, num_candidates)
        points, face_indices = trimesh.sample.sample_surface(mesh, sample_n)
        normals = mesh.face_normals[face_indices]

        results = []
        for i in range(sample_n):
            if len(results) >= num_candidates:
                break
            point = points[i]
            normal = normals[i]

            # Normal must tilt at least `min_normal_angle_deg` above the
            # horizontal plane. Since normals are unit-length, that's just a
            # z-component threshold.
            if normal[2] < min_nz:
                continue

            # Target: Align TCP Z (out of palm) with -normal (into object)
            target_z = -normal

            # Orthogonal basis
            tmp = (
                np.array([1, 0, 0])
                if np.abs(target_z[0]) < 0.9
                else np.array([0, 1, 0])
            )
            x_axis = np.cross(tmp, target_z)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(target_z, x_axis)
            R_base = np.column_stack([x_axis, y_axis, target_z])

            # Per-axis jitter expressed in the TCP frame. Rx/Ry tilt the
            # approach axis away from -normal; Rz rotates around the approach
            # axis (pure yaw, doesn't change where the hand is pointing, just
            # around which finger ends up where).
            rx = np.random.uniform(-jitter_rot_x, jitter_rot_x)
            ry = np.random.uniform(-jitter_rot_y, jitter_rot_y)
            rz = np.random.uniform(-jitter_rot_z, jitter_rot_z)
            R_jitter = Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()
            rotation_tcp = R_base @ R_jitter

            # Position
            standoff = np.random.uniform(standoff_min, standoff_max)
            dx = np.random.uniform(-jitter_xy, jitter_xy)
            dy = np.random.uniform(-jitter_xy, jitter_xy)

            # P_tcp = Point + Normal * Standoff + Jitter_in_plane
            pos_jitter = rotation_tcp @ np.array([dx, dy, 0])
            translation_tcp = point + normal * standoff + pos_jitter

            # Solve hand pose
            T_world_tcp = np.eye(4)
            T_world_tcp[:3, :3] = rotation_tcp
            T_world_tcp[:3, 3] = translation_tcp

            trans, rot = self._solve_hand_pose(T_world_tcp)
            results.append(SamplingResult(translation=trans, rotation=rot, point=point))

        if min_normal_angle_deg > 0.0 and len(results) < num_candidates:
            logger.warning(
                f"SurfaceNormalStrategy: only {len(results)}/{num_candidates} "
                f"candidates survived the min_normal_angle_deg="
                f"{min_normal_angle_deg}° filter after oversampling "
                f"{oversample_factor}x. Consider lowering the threshold or "
                f"increasing the oversample factor if the object's upper-facing "
                f"surface is small."
            )

        return results


class DomeSamplingStrategy(SamplingStrategy):
    """
    Samples grasps on a 'dome' or sphere surrounding the object.
    TCP Z-axis points towards the object center.
    Good for power grasps or larger objects.
    """

    def sample(
        self,
        mesh: trimesh.Trimesh,
        num_candidates: int,
        seed: Optional[int] = None,
    ) -> List[SamplingResult]:
        if seed is not None:
            np.random.seed(seed)

        # Object center and size
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        # Radius of bounding sphere (approx)
        extent = np.linalg.norm(bounds[1] - bounds[0]) / 2.0

        # Params
        params = self.hand_config.sampling_params or {}
        # Distance from center = radius + standoff
        # Making it relative to object extent for robustness
        standoff_min = params.get("standoff_min", 0.05)
        standoff_max = params.get("standoff_max", 0.10)

        jitter_rot_x = params.get("jitter_rot_x", 0.0)
        jitter_rot_y = params.get("jitter_rot_y", 0.0)
        jitter_rot_z = params.get("jitter_rot_z", np.pi)

        # Sample points on unit sphere
        # Gaussian sampling for uniform sphere distribution
        vecs = np.random.normal(size=(num_candidates, 3))
        vecs /= np.linalg.norm(vecs, axis=1)[:, np.newaxis]

        # Filter for "dome" (upper hemisphere + some side angles)
        # Assuming Z is up. Filter vecs where z < -0.2 (to avoid approach from below)
        # Note: Canonical mesh might not have Z as up, but we assume it for 'dome' logic
        mask = vecs[:, 2] > -0.3
        vecs = vecs[mask]
        # Resample missing if needed, or just proceed with fewer
        actual_num = len(vecs)

        results = []
        for i in range(actual_num):
            vec = vecs[i]
            # Direction FROM center TO hand = vec
            # TCP Z points TO center = -vec

            target_z = -vec

            # Basis
            tmp = (
                np.array([1, 0, 0])
                if np.abs(target_z[0]) < 0.9
                else np.array([0, 1, 0])
            )
            x_axis = np.cross(tmp, target_z)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(target_z, x_axis)
            R_base = np.column_stack([x_axis, y_axis, target_z])

            # Per-axis jitter in TCP frame (same semantics as
            # SurfaceNormalStrategy).
            rx = np.random.uniform(-jitter_rot_x, jitter_rot_x)
            ry = np.random.uniform(-jitter_rot_y, jitter_rot_y)
            rz = np.random.uniform(-jitter_rot_z, jitter_rot_z)
            R_jitter = Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()
            rotation_tcp = R_base @ R_jitter

            # Position
            standoff = np.random.uniform(standoff_min, standoff_max)
            dist = extent + standoff
            translation_tcp = center + vec * dist

            # Solve hand pose
            T_world_tcp = np.eye(4)
            T_world_tcp[:3, :3] = rotation_tcp
            T_world_tcp[:3, 3] = translation_tcp

            trans, rot = self._solve_hand_pose(T_world_tcp)

            # Use center as a reference seed point
            results.append(
                SamplingResult(translation=trans, rotation=rot, point=center)
            )

        return results
