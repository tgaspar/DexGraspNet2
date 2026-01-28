"""
Grasp label generation using Isaac Lab simulation.

This module provides the GraspGenerator class for generating grasp labels
by simulating grasp execution and filtering stable grasps.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dexgraspnet2.configs.hand_config import HandConfig
from dexgraspnet2.generation.grasp_simulator import GraspSimulator, SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class GraspCandidate:
    """
    Represents a candidate grasp pose for evaluation.

    Attributes:
        translation: Wrist position (3,).
        rotation: Wrist orientation as 3x3 rotation matrix (3, 3).
        joint_angles: Joint angles in radians, ordered by HandConfig.joint_names.
        object_id: Object instance ID this grasp is for.
    """

    translation: np.ndarray
    rotation: np.ndarray
    joint_angles: np.ndarray
    object_id: int = 0


@dataclass
class GraspLabel:
    """
    Represents a validated grasp label.

    Attributes:
        translation: Wrist position (3,).
        rotation: Wrist orientation as 3x3 rotation matrix (3, 3).
        joint_angles: Joint angles in radians.
        is_stable: Whether the grasp successfully lifted the object.
        lift_height: Height the object was lifted (meters).
        contact_points: Points where hand contacts object (N, 3).
    """

    translation: np.ndarray
    rotation: np.ndarray
    joint_angles: np.ndarray
    is_stable: bool = False
    lift_height: float = 0.0
    contact_points: Optional[np.ndarray] = None


class GraspGenerator:
    """
    Generates and validates grasp labels using Isaac Lab simulation.

    This class implements the grasp label generation pipeline from DexGraspNet 2.0:
    1. Initialize grasp candidates (from reference grasps or sampling)
    2. Optimize grasps using force-closure metric
    3. Validate grasps via physics simulation
    4. Export stable grasps as training labels

    Example:
        >>> generator = GraspGenerator(
        ...     hand_config=HandConfig.inspire_hand(),
        ...     device="cuda:0",
        ...     headless=True
        ... )
        >>> generator.setup_scene(object_meshes=[bowl_mesh])
        >>> labels = generator.generate(num_grasps=1000)
        >>> generator.save_labels("grasps.npz")

    Note:
        This class requires Isaac Lab to be installed. The full implementation
        will be completed once Isaac Lab environment is set up.
    """

    def __init__(
        self,
        hand_config: HandConfig,
        device: str = "cuda:0",
        headless: bool = True,
        simulation_hz: float = 60.0,
        friction: float = 0.5,
    ):
        """
        Initialize the grasp generator.

        Args:
            hand_config: Hand configuration (URDF, joints, limits).
            device: Device for simulation (e.g., 'cuda:0').
            headless: Run without visualization if True.
            simulation_hz: Simulation frequency in Hz.
            friction: Friction coefficient for contacts.
        """
        self.hand_config = hand_config
        self.device = device
        self.headless = headless
        self.simulation_hz = simulation_hz
        self.friction = friction

        self._scene_initialized = False
        self._object_meshes: List[Path] = []
        self._object_poses: List[np.ndarray] = []
        self._labels: List[GraspLabel] = []

        self.simulator = GraspSimulator(
            hand_config=hand_config,
            device=device,
            headless=headless,
            config=SimulationConfig(friction=friction),
        )
        self._batch_size = 64  # Default batch size for simulation

        logger.info(
            f"GraspGenerator initialized for {hand_config.name} "
            f"({hand_config.num_dofs} DoF)"
        )

    def setup_scene(
        self,
        object_meshes: List[Path],
        object_poses: Optional[List[np.ndarray]] = None,
        table_height: float = 0.0,
    ) -> None:
        """
        Set up the simulation scene with objects.

        Args:
            object_meshes: List of paths to object mesh files (.obj, .urdf).
            object_poses: Initial poses for each object (4, 4) matrices.
                If None, objects are placed on the table.
            table_height: Height of the table surface.

        Raises:
            ImportError: If Isaac Lab is not available.
            FileNotFoundError: If mesh files don't exist.
        """
        # Validate inputs
        for mesh_path in object_meshes:
            mesh_path = Path(mesh_path)
            if not mesh_path.exists():
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        self._object_meshes = [Path(p) for p in object_meshes]

        if object_poses is None:
            # Place objects on table with default spacing
            self._object_poses = [np.eye(4) for _ in object_meshes]
            for i, pose in enumerate(self._object_poses):
                pose[2, 3] = table_height + 0.05  # 5cm above table
        else:
            self._object_poses = [np.array(p) for p in object_poses]

        self._table_height = table_height
        self._scene_initialized = True

        # Initialize simulator with first object mesh
        # TODO: Support multiple objects in GraspSimulator if needed for clutter generation
        if self._object_meshes:
            self.simulator.setup(
                object_mesh_path=self._object_meshes[0], num_envs=self._batch_size
            )

        logger.info(f"Scene set up with {len(object_meshes)} object(s)")

    def sample_grasp_candidates(
        self,
        num_candidates: int = 1000,
        seed: Optional[int] = None,
    ) -> List[GraspCandidate]:
        """
        Sample initial grasp candidates.

        Uses approach vectors sampled from object surface and random
        joint configurations within limits.

        Args:
            num_candidates: Number of candidates to generate.
            seed: Random seed for reproducibility.

        Returns:
            List of GraspCandidate objects.
        """
        if seed is not None:
            np.random.seed(seed)

        candidates = []

        for _ in range(num_candidates):
            # Sample random joint angles within limits
            lower = np.array(self.hand_config.joint_lower_limits)
            upper = np.array(self.hand_config.joint_upper_limits)
            joints = np.random.uniform(lower, upper)

            # Sample random position (will be refined)
            translation = np.array([0.0, 0.0, 0.2])

            # Sample random rotation
            rotation = self._sample_random_rotation()

            candidates.append(
                GraspCandidate(
                    translation=translation,
                    rotation=rotation,
                    joint_angles=joints,
                    object_id=0,
                )
            )

        return candidates

    def _sample_random_rotation(self) -> np.ndarray:
        """Sample a random rotation matrix."""
        # Use quaternion uniform sampling
        u = np.random.uniform(size=3)
        q = np.array(
            [
                np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
                np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
                np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
                np.sqrt(u[0]) * np.cos(2 * np.pi * u[2]),
            ]
        )
        # Convert to rotation matrix
        return self._quaternion_to_matrix(q)

    @staticmethod
    def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion (x,y,z,w) to rotation matrix."""
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def optimize_grasps(
        self,
        candidates: List[GraspCandidate],
        use_force_closure: bool = True,
        max_iterations: int = 100,
    ) -> List[GraspCandidate]:
        """
        Optimize grasp candidates using force-closure optimization.

        Args:
            candidates: Initial grasp candidates.
            use_force_closure: Use force-closure metric for optimization.
            max_iterations: Maximum optimization iterations.

        Returns:
            Optimized grasp candidates.
        """
        # TODO: Implement force-closure optimization
        # For now, return candidates unchanged
        logger.warning("Force-closure optimization not yet implemented")
        return candidates

    def validate_grasps(
        self,
        candidates: List[GraspCandidate],
        batch_size: int = 100,
    ) -> List[GraspLabel]:
        """
        Validate grasp candidates via physics simulation.

        Simulates grasp execution: approach -> close -> lift.
        Returns labels for grasps that successfully lift the object.

        Args:
            candidates: Grasp candidates to validate.
            batch_size: Number of grasps to simulate in parallel.

        Returns:
            List of GraspLabel objects with validation results.
        """
        if not self._scene_initialized:
            raise RuntimeError("Scene not initialized. Call setup_scene() first.")

        if not candidates:
            return []

        logger.info(f"Validating {len(candidates)} candidates via simulation...")

        # Convert candidates to arrays for batch processing
        translations = np.stack([c.translation for c in candidates])
        rotations = np.stack([c.rotation for c in candidates])
        joint_angles = np.stack([c.joint_angles for c in candidates])

        num_candidates = len(candidates)
        all_is_stable = []
        all_lift_heights = []

        # Process in batches
        # Ensure we use the configured batch size
        sim_batch_size = min(batch_size, self._batch_size)

        for i in range(0, num_candidates, sim_batch_size):
            batch_trans = translations[i : i + sim_batch_size]
            batch_rots = rotations[i : i + sim_batch_size]
            batch_joints = joint_angles[i : i + sim_batch_size]

            is_stable, lift_heights = self.simulator.validate_batch(
                batch_trans, batch_rots, batch_joints
            )

            all_is_stable.append(is_stable)
            all_lift_heights.append(lift_heights)

            logger.info(
                f"Validated batch {i // sim_batch_size + 1}: "
                f"{np.sum(is_stable)}/{len(is_stable)} stable"
            )

        # Concatenate results
        is_stable = np.concatenate(all_is_stable)
        lift_heights = np.concatenate(all_lift_heights)

        labels = []
        for i, candidate in enumerate(candidates):
            labels.append(
                GraspLabel(
                    translation=candidate.translation,
                    rotation=candidate.rotation,
                    joint_angles=candidate.joint_angles,
                    is_stable=bool(is_stable[i]),
                    lift_height=float(lift_heights[i]),
                )
            )

        return labels

    def generate(
        self,
        num_grasps: int = 1000,
        seed: Optional[int] = None,
        optimize: bool = True,
    ) -> List[GraspLabel]:
        """
        Generate grasp labels for the current scene.

        Full pipeline: sample -> optimize -> validate.

        Args:
            num_grasps: Target number of stable grasps.
            seed: Random seed for reproducibility.
            optimize: Apply force-closure optimization.

        Returns:
            List of validated GraspLabel objects.
        """
        if not self._scene_initialized:
            raise RuntimeError("Scene not initialized. Call setup_scene() first.")

        # Sample candidates (oversample to account for failures)
        num_candidates = num_grasps * 10
        candidates = self.sample_grasp_candidates(num_candidates, seed)

        # Optimize
        if optimize:
            candidates = self.optimize_grasps(candidates)

        # Validate
        labels = self.validate_grasps(candidates)

        # Filter stable grasps
        stable_labels = [l for l in labels if l.is_stable]

        self._labels = stable_labels[:num_grasps]
        logger.info(
            f"Generated {len(self._labels)} stable grasps "
            f"from {num_candidates} candidates"
        )

        return self._labels

    def save_labels(
        self,
        output_path: str,
        include_unstable: bool = False,
    ) -> None:
        """
        Save grasp labels to file.

        Args:
            output_path: Output file path (.npz).
            include_unstable: Include unstable grasps in output.
        """
        if not self._labels:
            raise RuntimeError("No labels to save. Run generate() first.")

        labels_to_save = self._labels
        if not include_unstable:
            labels_to_save = [l for l in labels_to_save if l.is_stable]

        # Convert to arrays
        translations = np.stack([l.translation for l in labels_to_save])
        rotations = np.stack([l.rotation for l in labels_to_save])
        joint_angles = np.stack([l.joint_angles for l in labels_to_save])
        is_stable = np.array([l.is_stable for l in labels_to_save])
        lift_heights = np.array([l.lift_height for l in labels_to_save])

        # Save with joint names for reference
        np.savez(
            output_path,
            translations=translations,
            rotations=rotations,
            joint_angles=joint_angles,
            is_stable=is_stable,
            lift_heights=lift_heights,
            joint_names=self.hand_config.joint_names,
            hand_name=self.hand_config.name,
        )

        logger.info(f"Saved {len(labels_to_save)} labels to {output_path}")

    @classmethod
    def load_labels(cls, path: str) -> Tuple[List[GraspLabel], Dict]:
        """
        Load grasp labels from file.

        Args:
            path: Path to saved labels (.npz).

        Returns:
            Tuple of (labels, metadata).
        """
        data = np.load(path)

        labels = []
        for i in range(len(data["translations"])):
            labels.append(
                GraspLabel(
                    translation=data["translations"][i],
                    rotation=data["rotations"][i],
                    joint_angles=data["joint_angles"][i],
                    is_stable=bool(data["is_stable"][i]),
                    lift_height=float(data["lift_heights"][i]),
                )
            )

        metadata = {
            "joint_names": list(data["joint_names"]),
            "hand_name": str(data["hand_name"]),
        }

        return labels, metadata
