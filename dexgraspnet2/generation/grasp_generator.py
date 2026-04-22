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
import trimesh
from scipy.spatial.transform import Rotation

from dexgraspnet2.configs.hand_config import HandConfig
from dexgraspnet2.generation.grasp_simulator import GraspSimulator, SimulationConfig
from dexgraspnet2.generation.sampling_strategies import (
    DomeSamplingStrategy,
    SurfaceNormalStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class GraspCandidate:
    """
    Represents a candidate grasp pose for evaluation.

    Attributes:
        translation: Wrist position (3,).
        rotation: Wrist orientation as 3x3 rotation matrix (3, 3).
        joint_angles: Joint angles in radians, ordered by HandConfig.joint_names.
        point: Target contact point on object surface (3,).
        object_id: Object instance ID this grasp is for.
        pregrasp_joint_angles: Joint angles to hold at pregrasp waypoint.
        preshape_name: Name of the preshape this candidate was sampled from
            (e.g. "power", "pinch_index"), or None if no preshape was used.
    """

    translation: np.ndarray
    rotation: np.ndarray
    joint_angles: np.ndarray
    point: Optional[np.ndarray] = None
    object_id: int = 0
    pregrasp_joint_angles: Optional[np.ndarray] = None
    preshape_name: Optional[str] = None


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
        point: Target contact point on object surface (3,).
        contact_points: Points where hand contacts object (N, 3).
        trajectory: Optional per-candidate debug trace — includes the 5
            milestone snapshots from the simulator plus the preshape name
            and sampling strategy used. Populated by validate_grasps.
    """

    translation: np.ndarray
    rotation: np.ndarray
    joint_angles: np.ndarray
    is_stable: bool = False
    lift_height: float = 0.0
    point: Optional[np.ndarray] = None
    contact_points: Optional[np.ndarray] = None
    trajectory: Optional[Dict] = None


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
        self._all_labels: List[GraspLabel] = []

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

        # Load meshes for sampling
        self._trimesh_objects = []
        for p in self._object_meshes:
            try:
                self._trimesh_objects.append(trimesh.load(p, force="mesh"))
            except Exception as e:
                logger.warning(f"Failed to load mesh {p} for sampling: {e}")
                self._trimesh_objects.append(None)

        if object_poses is None:
            # Place each object so its mesh's lowest vertex sits exactly
            # `_spawn_margin` above the table. Without this, spawning every
            # object at a fixed z=0.05 causes meshes whose local origin is
            # offset from the geometric bottom to either (a) start
            # penetrating the ground plane — PhysX explodes the object
            # outward to resolve the intersection, sometimes hurling it
            # multiple metres — or (b) start floating way above the ground
            # and freefall with non-trivial kinetic energy into the settle
            # step. A per-mesh z avoids both.
            # Note: margin must exceed PhysX's `contact_offset` (default 2 mm
            # in SimulationConfig) or the mesh spawns *inside* the contact
            # buffer zone — the solver applies a separation impulse on step 0
            # that destabilises (and sometimes segfaults) the sim. 5 mm is
            # comfortably above the contact buffer.
            spawn_margin = 0.005
            self._object_poses = []
            for i, mesh in enumerate(self._trimesh_objects):
                pose = np.eye(4)
                if mesh is not None and hasattr(mesh, "bounds"):
                    z_min_local = float(mesh.bounds[0][2])
                    spawn_z = table_height - z_min_local + spawn_margin
                    logger.info(
                        f"Mesh {self._object_meshes[i].name}: z_min_local="
                        f"{z_min_local:+.4f} → spawn_z={spawn_z:.4f}"
                    )
                else:
                    spawn_z = table_height + 0.05  # legacy fallback
                    logger.warning(
                        f"Mesh for {self._object_meshes[i]} unavailable; "
                        f"using fallback spawn_z={spawn_z:.4f}"
                    )
                pose[2, 3] = spawn_z
                self._object_poses.append(pose)
        else:
            self._object_poses = [np.array(p) for p in object_poses]

        self._table_height = table_height
        self._scene_initialized = True

        # Initialize simulator with first object mesh. The simulator settles
        # the object under gravity during setup(), so the authoritative world
        # pose of the object after setup is the *settled* pose — overwrite our
        # stored pose with it so `sample_grasp_candidates` targets the
        # resting mesh, not the arbitrary initial spawn.
        # TODO: Support multiple objects in GraspSimulator if needed for clutter generation
        if self._object_meshes:
            self.simulator.setup(
                object_mesh_path=self._object_meshes[0],
                num_envs=self._batch_size,
                object_pose=self._object_poses[0],
            )
            settled_pose = self.simulator.get_initial_object_pose()
            logger.info(
                f"Using settled object pose for sampling: "
                f"translation {settled_pose[:3, 3].tolist()}"
            )
            self._object_poses[0] = settled_pose

        logger.info(f"Scene set up with {len(object_meshes)} object(s)")

    def sample_grasp_candidates(
        self,
        num_candidates: int = 1000,
        seed: Optional[int] = None,
    ) -> List[GraspCandidate]:
        """
        Sample initial grasp candidates using the configured strategy.

        Delegates to SamplingStrategy implementation defined in HandConfig.

        Args:
            num_candidates: Number of candidates to generate.
            seed: Random seed for reproducibility.

        Returns:
            List of GraspCandidate objects.
        """
        if seed is not None:
            np.random.seed(seed)

        candidates = []

        if not self._trimesh_objects or self._trimesh_objects[0] is None:
            logger.warning("No valid mesh for sampling, using fallback")
            return self._sample_random_candidates(num_candidates)

        # Transform the mesh by the object's world-frame spawn pose so the
        # sampler produces candidates in the same frame the simulator places
        # the object in. Without this, targets are in mesh frame while the
        # sim spawns at `self._object_poses[0]`, offsetting every candidate.
        mesh = self._trimesh_objects[0].copy()
        mesh.apply_transform(self._object_poses[0])

        # Select strategy
        strategy_name = getattr(self.hand_config, "sampling_strategy", "surface_normal")
        if strategy_name == "dome":
            strategy = DomeSamplingStrategy(self.hand_config)
        else:
            strategy = SurfaceNormalStrategy(self.hand_config)

        results = strategy.sample(mesh, num_candidates, seed)

        # Preshapes logic
        preshapes = self.hand_config.preshapes
        preshape_keys = list(preshapes.keys()) if preshapes else []

        for res in results:
            # Select preshape
            if preshape_keys:
                key = str(np.random.choice(preshape_keys))
                joints = np.array(preshapes[key]["target"])
                pregrasp_joints = np.array(preshapes[key]["pregrasp"])
            else:
                key = None
                joints = np.array(self.hand_config.get_closed_pose())
                pregrasp_joints = np.array(self.hand_config.get_open_pose())

            candidates.append(
                GraspCandidate(
                    translation=res.translation,
                    rotation=res.rotation,
                    joint_angles=joints,
                    point=res.point,
                    object_id=0,
                    pregrasp_joint_angles=pregrasp_joints,
                    preshape_name=key,
                )
            )

        return candidates

    def _sample_random_candidates(self, num_candidates: int) -> List[GraspCandidate]:
        """Fallback sampling without mesh."""
        candidates = []

        preshapes = self.hand_config.preshapes
        preshape_keys = list(preshapes.keys()) if preshapes else []

        for _ in range(num_candidates):
            if preshape_keys:
                key = str(np.random.choice(preshape_keys))
                joints = np.array(preshapes[key]["target"])
                pregrasp_joints = np.array(preshapes[key]["pregrasp"])
            else:
                key = None
                joints = np.array(self.hand_config.get_closed_pose())
                pregrasp_joints = np.array(self.hand_config.get_open_pose())

            translation = np.array([0.0, 0.0, 0.2]) + np.random.uniform(-0.05, 0.05, 3)
            rotation = self._sample_random_rotation()
            candidates.append(
                GraspCandidate(
                    translation=translation,
                    rotation=rotation,
                    joint_angles=joints,
                    point=np.zeros(3),
                    object_id=0,
                    pregrasp_joint_angles=pregrasp_joints,
                    preshape_name=key,
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

        # Stack pregrasp joint angles only if every candidate has them set.
        # If any is None, fall back to passing None (simulator opens fingers).
        if all(c.pregrasp_joint_angles is not None for c in candidates):
            pregrasp_stack = np.stack([c.pregrasp_joint_angles for c in candidates])
        else:
            pregrasp_stack = None

        num_candidates = len(candidates)
        all_is_stable = []
        all_lift_heights = []
        all_final_joints = []
        all_traces: List[dict] = []

        # Process in batches
        # Ensure we use the configured batch size
        sim_batch_size = min(batch_size, self._batch_size)

        for i in range(0, num_candidates, sim_batch_size):
            batch_trans = translations[i : i + sim_batch_size]
            batch_rots = rotations[i : i + sim_batch_size]
            batch_joints = joint_angles[i : i + sim_batch_size]
            batch_pregrasp = (
                pregrasp_stack[i : i + sim_batch_size]
                if pregrasp_stack is not None
                else None
            )

            is_stable, lift_heights, final_joints, traces = self.simulator.validate_batch(
                batch_trans,
                batch_rots,
                batch_joints,
                pregrasp_joint_angles=batch_pregrasp,
                visualize=not self.headless,
            )

            all_is_stable.append(is_stable)
            all_lift_heights.append(lift_heights)
            all_final_joints.append(final_joints)
            all_traces.extend(traces)

            logger.info(
                f"Validated batch {i // sim_batch_size + 1}: "
                f"{np.sum(is_stable)}/{len(is_stable)} stable"
            )

        # Concatenate results
        is_stable = np.concatenate(all_is_stable)
        lift_heights = np.concatenate(all_lift_heights)
        final_joints = np.concatenate(all_final_joints)

        strategy_name = getattr(self.hand_config, "sampling_strategy", None)

        labels = []
        for i, candidate in enumerate(candidates):
            # Attach per-candidate debug trace. If the simulator truncated a
            # larger batch (shouldn't happen given sim_batch_size <= _batch_size
            # but defensive), some candidates may lack snapshots — leave None.
            snap = all_traces[i] if i < len(all_traces) else None
            trajectory: Optional[Dict] = None
            if snap is not None:
                trajectory = {
                    "snapshots": snap,
                    "preshape": candidate.preshape_name,
                    "sampling_strategy": strategy_name,
                    "sampled_point": (
                        candidate.point.tolist()
                        if candidate.point is not None
                        else None
                    ),
                    "candidate_index": i,
                }

            labels.append(
                GraspLabel(
                    translation=candidate.translation,
                    rotation=candidate.rotation,
                    joint_angles=final_joints[
                        i
                    ],  # Use the actual closed joints from sim
                    is_stable=bool(is_stable[i]),
                    lift_height=float(lift_heights[i]),
                    point=candidate.point,
                    trajectory=trajectory,
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

        # Keep the full list too — it carries the failed candidates'
        # trajectory traces, which callers (e.g. trajectory JSONL writer)
        # need for post-hoc debugging.
        self._all_labels = labels
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
