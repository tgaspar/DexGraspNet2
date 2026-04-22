"""
Isaac Gym-based grasp simulation for validating grasp candidates.

This module provides the GraspSimulator class that uses Isaac Gym's GPU
physics pipeline to efficiently validate many grasp candidates in parallel.

Uses the "_free" URDF variant with a virtual 6-DOF joint chain for floating
base control, following the approach from DexGraspNet2.0 paper.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# Set environment variables before importing isaacgym
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

from isaacgym import gymapi, gymtorch
import torch

from dexgraspnet2.configs.hand_config import HandConfig
from dexgraspnet2.utils.geometric_conversions import (
    rotation_matrix_to_euler_xyz_intrinsic,
    rotation_matrix_to_quaternion,
)

# Legacy default spawn pose (identity rotation, 5 cm above ground).
_LEGACY_OBJECT_POSE = np.eye(4)
_LEGACY_OBJECT_POSE[2, 3] = 0.05

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for the grasp simulation."""

    dt: float = 1.0 / 60.0
    substeps: int = 2
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    friction: float = 1.0
    contact_offset: float = 0.002
    rest_offset: float = 0.0
    solver_type: int = 1
    num_position_iterations: int = 8
    num_velocity_iterations: int = 2

    # Grasp validation params (following original paper's 5-waypoint system)
    pregrasp_steps: int = 30  # Steps at pregrasp position
    approach_steps: int = 60  # Steps to approach (pregrasp -> cover)
    grasp_steps: int = 60  # Steps to close fingers
    squeeze_steps: int = 30  # Additional squeeze
    lift_steps: int = 90  # Steps to lift
    settle_steps: int = 60  # Steps to let the object settle on the ground at setup

    # Distances
    pregrasp_distance: float = 0.10  # 10cm back from grasp pose
    lift_height: float = 0.10  # Lift 10cm
    settle_hand_safe_z: float = 0.5  # Z height to park the hand during settling
    success_threshold: float = 0.03  # Object must rise 3cm (paper: 3cm above initial)


class GraspSimulator:
    """
    Isaac Gym-based grasp simulator for validating grasp candidates.

    Uses the "_free" URDF variant which adds a virtual 6-DOF joint chain
    (3 prismatic + 3 revolute) before the hand base. This allows controlling
    the hand's position and orientation through DOF targets rather than
    direct rigid body manipulation.

    DOF structure for _free variant:
        [x, y, z, rx, ry, rz, finger_joint_1, ..., finger_joint_n]

    Example:
        >>> sim = GraspSimulator(hand_config, headless=False)
        >>> sim.setup(object_mesh_path="bowl.obj")
        >>> results = sim.validate_grasps(candidates, batch_size=64)
        >>> sim.cleanup()
    """

    # Virtual 6-DOF joint names in order
    VIRTUAL_JOINT_NAMES = [
        "x_joint",
        "y_joint",
        "z_joint",
        "x_rotation_joint",
        "y_rotation_joint",
        "z_rotation_joint",
    ]

    def __init__(
        self,
        hand_config: HandConfig,
        device: str = "cuda:0",
        headless: bool = True,
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize the grasp simulator.

        Args:
            hand_config: Hand configuration with URDF path and joint info.
            device: CUDA device for simulation.
            headless: Run without GUI visualization.
            config: Simulation configuration parameters.
        """
        self.hand_config = hand_config
        self.device = device
        self.headless = headless
        self.config = config or SimulationConfig()

        self._gym = None
        self._sim = None
        self._viewer = None
        self._envs = []
        self._robot_handles = []
        self._object_handles = []

        self._root_states = None
        self._dof_states = None
        self._dof_targets = None

        self._initialized = False

        # Derive the _free URDF path from the hand config
        self._free_urdf_path = self._get_free_urdf_path()

        # Precompute TCP-derived approach axis expressed in the wrist frame.
        # TCP convention (see HandConfig): +Z points toward the object.
        # Pregrasp backoff and approach interpolation use this axis.
        if self.hand_config.tcp_rotation_rpy is not None:
            R_wrist_tcp = Rotation.from_euler(
                "xyz", self.hand_config.tcp_rotation_rpy, degrees=True
            ).as_matrix()
            self._approach_axis_wrist = R_wrist_tcp @ np.array([0.0, 0.0, 1.0])
        else:
            # No TCP rotation configured: assume TCP = wrist, so approach axis
            # is wrist +Z. Set tcp_rotation_rpy in the hand YAML if incorrect.
            self._approach_axis_wrist = np.array([0.0, 0.0, 1.0])
        logger.info(
            f"Approach axis (wrist frame) for {hand_config.name}: "
            f"{np.round(self._approach_axis_wrist, 3).tolist()}"
        )

        # Set by setup(); world-frame spawn pose of the object (4x4).
        self._initial_object_pose: Optional[np.ndarray] = None

        logger.info(f"GraspSimulator created for {hand_config.name}")

    def _get_free_urdf_path(self) -> Path:
        """Get the path to the _free URDF variant."""
        urdf_path = Path(self.hand_config.urdf_path)

        # Check if it's already a _free variant
        if "_free" in urdf_path.stem:
            return urdf_path

        # Try to find the _free variant
        free_name = urdf_path.stem + "_free" + urdf_path.suffix
        free_path = urdf_path.parent / free_name

        # Also check in robot_models/urdf directory
        if not free_path.exists():
            project_root = Path(__file__).parent.parent.parent
            alt_path = project_root / "robot_models" / "urdf" / free_name
            if alt_path.exists():
                free_path = alt_path

        if not free_path.exists():
            raise FileNotFoundError(
                f"Could not find _free URDF variant. Tried:\n"
                f"  {urdf_path.parent / free_name}\n"
                f"  robot_models/urdf/{free_name}\n"
                f"Please create the _free variant with virtual 6-DOF joint chain."
            )

        logger.info(f"Using _free URDF: {free_path}")
        return free_path

    def setup(
        self,
        object_mesh_path: Optional[Path] = None,
        object_scale: float = 1.0,
        num_envs: int = 1,
        object_pose: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set up the simulation environment.

        Args:
            object_mesh_path: Path to object mesh file. If None, uses a box.
            object_scale: Scale factor for the object mesh.
            num_envs: Number of parallel environments.
            object_pose: 4x4 world-frame spawn pose for the object. Defaults to
                identity rotation with a +5 cm Z translation (legacy behavior).
                The reset logic in `validate_batch` / `validate_single_grasp`
                uses this pose; sampled grasp candidates must be in the same
                world frame (see `GraspGenerator.sample_grasp_candidates`).
        """
        logger.info(f"Setting up simulation with {num_envs} environment(s)")

        # If setup() was already called once (e.g. this is the second object
        # in a multi-object loop), tear down the previous sim first. Otherwise
        # the old sim's GPU/PhysX state is orphaned but still resident, and
        # after a few iterations something in PhysX or the Isaac Gym driver
        # segfaults during asset loading / env creation of the new sim.
        if self._initialized:
            logger.info("Tearing down previous sim before creating a new one.")
            self.cleanup()

        # Store object spawn pose for use in _create_envs and reset logic.
        self._initial_object_pose = (
            np.asarray(object_pose, dtype=np.float64)
            if object_pose is not None
            else _LEGACY_OBJECT_POSE.copy()
        )

        # Initialize gym
        self._gym = gymapi.acquire_gym()

        # Simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(*self.config.gravity)
        sim_params.dt = self.config.dt
        sim_params.substeps = self.config.substeps
        sim_params.physx.solver_type = self.config.solver_type
        sim_params.physx.num_position_iterations = self.config.num_position_iterations
        sim_params.physx.num_velocity_iterations = self.config.num_velocity_iterations
        sim_params.physx.contact_offset = self.config.contact_offset
        sim_params.physx.rest_offset = self.config.rest_offset

        # Create simulation
        compute_device = int(self.device.split(":")[-1])
        self._sim = self._gym.create_sim(
            compute_device, compute_device, gymapi.SIM_PHYSX, sim_params
        )

        if self._sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation")

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self._gym.add_ground(self._sim, plane_params)

        # Create viewer if not headless
        if not self.headless:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280
            camera_props.height = 720
            self._viewer = self._gym.create_viewer(self._sim, camera_props)

            if self._viewer is None:
                logger.warning("Failed to create viewer, continuing headless")
            else:
                cam_pos = gymapi.Vec3(0.5, -0.5, 0.5)
                cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
                self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)

        # Load robot asset (using _free variant)
        robot_asset = self._load_robot_asset()

        # Load or create object asset
        object_asset = self._load_object_asset(object_mesh_path, object_scale)

        # Create environments
        self._create_envs(num_envs, robot_asset, object_asset)

        # Prepare simulation
        self._gym.prepare_sim(self._sim)

        # Acquire tensor handles
        self._setup_tensors()

        self._initialized = True

        # Let the object settle on the ground under gravity. The sampling
        # math needs the object's actual rest pose, not the (arbitrary)
        # spawn pose we placed it at. Overwrites `self._initial_object_pose`
        # with the measured settled pose so reset-between-batches puts the
        # object back where the sampler thinks it is.
        if self.config.settle_steps > 0:
            self._settle_object()

        logger.info("Simulation setup complete")

    def _settle_object(self) -> None:
        """
        Park the hand out of the way and let the object fall and settle under
        gravity for `settle_steps` sim steps. Then read back the actual object
        pose and store it as the new `_initial_object_pose`.

        Matches the paper's reference validation pipeline
        (tests/demo_batch_grasp_validation.py:463-488): settle first, sample
        grasps against the settled pose. This replaces the earlier shortcut
        of disabling object gravity during pregrasp/approach/grasp/squeeze.
        """
        safe_z = float(self.config.settle_hand_safe_z)
        num_envs = len(self._envs)

        # Park every env's hand at (0, 0, safe_z) with identity rotation.
        safe_translations = np.zeros((num_envs, 3), dtype=np.float64)
        safe_translations[:, 2] = safe_z
        safe_rotations = np.tile(np.eye(3, dtype=np.float64), (num_envs, 1, 1))

        self._dof_targets.zero_()
        self._set_hand_poses_batch(safe_translations, safe_rotations)

        # Teleport the DOF state so the hand is physically up there from
        # step 0 — otherwise the PD controller takes a few steps to arrive
        # and the object meanwhile settles with the hand passing through.
        dof_state_view = self._dof_states.view(num_envs, self._dofs_per_robot, 2)
        targets_view = self._dof_targets.view(num_envs, self._dofs_per_robot)
        dof_state_view[:, :, 0] = targets_view
        dof_state_view[:, :, 1] = 0.0
        self._gym.set_dof_state_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_states)
        )

        # Step under gravity.
        for _ in range(self.config.settle_steps):
            self.step(render=False)

        # Read back the settled object pose. All envs have the same mesh at
        # the same initial pose, so env 0 is representative.
        self._gym.refresh_actor_root_state_tensor(self._sim)
        obj_idx_0 = int(self._object_indices[0].item())
        settled_state = self._root_states[obj_idx_0, :7].cpu().numpy()
        settled_pos = settled_state[:3]
        settled_quat_xyzw = settled_state[3:7]

        settled_pose = np.eye(4)
        settled_pose[:3, :3] = Rotation.from_quat(settled_quat_xyzw).as_matrix()
        settled_pose[:3, 3] = settled_pos

        logger.info(
            f"Object settled: p {self._initial_object_pose[:3, 3].tolist()} -> "
            f"{settled_pos.tolist()}; saved as initial_object_pose."
        )
        self._initial_object_pose = settled_pose

    def get_initial_object_pose(self) -> np.ndarray:
        """
        Return the 4x4 world-frame pose the simulator will reset the object to
        at the start of each validate_* call. After `setup()` this is the
        *settled* pose (not the user-provided spawn pose).
        """
        if self._initial_object_pose is None:
            raise RuntimeError("setup() must be called before get_initial_object_pose()")
        return self._initial_object_pose.copy()

    def _load_robot_asset(self) -> int:
        """Load the robot hand asset (_free variant)."""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True  # Fix the root of the virtual chain
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.armature = 0.01

        asset_root = str(self._free_urdf_path.parent)
        asset_file = self._free_urdf_path.name

        robot_asset = self._gym.load_asset(
            self._sim, asset_root, asset_file, asset_options
        )

        if robot_asset is None:
            raise RuntimeError(f"Failed to load robot from {self._free_urdf_path}")

        num_dofs = self._gym.get_asset_dof_count(robot_asset)
        logger.info(f"Loaded {self.hand_config.name} (_free) with {num_dofs} DOFs")

        return robot_asset

    def _load_object_asset(
        self,
        mesh_path: Optional[Path],
        scale: float,
    ) -> int:
        """Load or create the object asset."""
        asset_options = gymapi.AssetOptions()
        asset_options.density = 500.0
        asset_options.fix_base_link = False
        # Convex decomposition for collision
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        asset_options.vhacd_params.max_convex_hulls = 10
        asset_options.vhacd_params.max_num_vertices_per_ch = 64

        if mesh_path is None:
            # Create a simple box for testing
            object_asset = self._gym.create_box(
                self._sim, 0.05, 0.05, 0.05, asset_options
            )
            logger.info("Created box object (no mesh provided)")
        else:
            mesh_path = Path(mesh_path)
            if not mesh_path.exists():
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

            # Isaac Gym works best with URDFs.
            # If the mesh path points to an OBJ, check if a URDF exists in the same directory.
            # DexGraspNet dataset usually has nontextured_simplified.urdf alongside simplified.obj

            asset_path = mesh_path

            # Check for standard URDF
            urdf_path = mesh_path.parent / "nontextured_simplified.urdf"
            if urdf_path.exists():
                asset_path = urdf_path
            # Check for same-name URDF
            elif mesh_path.with_suffix(".urdf").exists():
                asset_path = mesh_path.with_suffix(".urdf")

            logger.info(f"Loading object asset from: {asset_path}")

            object_asset = self._gym.load_asset(
                self._sim, str(asset_path.parent), asset_path.name, asset_options
            )

        if object_asset is None:
            raise RuntimeError("Failed to create object asset")

        return object_asset

    def _create_envs(
        self,
        num_envs: int,
        robot_asset: int,
        object_asset: int,
    ) -> None:
        """Create parallel environments."""
        env_lower = gymapi.Vec3(-0.5, -0.5, 0.0)
        env_upper = gymapi.Vec3(0.5, 0.5, 1.0)

        # Spacing between environments
        envs_per_row = int(np.sqrt(num_envs))

        self._envs = []
        self._robot_handles = []
        self._object_handles = []

        for i in range(num_envs):
            env = self._gym.create_env(self._sim, env_lower, env_upper, envs_per_row)
            self._envs.append(env)

            # Add robot at origin - position controlled via virtual 6-DOF joints
            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            robot_pose.r = gymapi.Quat(0, 0, 0, 1)

            robot_handle = self._gym.create_actor(
                env, robot_asset, robot_pose, f"robot_{i}", i, 1
            )
            self._robot_handles.append(robot_handle)

            # Set robot drive properties for ALL DOFs (virtual + finger)
            dof_props = self._gym.get_actor_dof_properties(env, robot_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)

            # Virtual joints need high stiffness/damping for smooth movement
            dof_names = self._gym.get_actor_dof_names(env, robot_handle)
            for j, name in enumerate(dof_names):
                if name in self.VIRTUAL_JOINT_NAMES:
                    dof_props["stiffness"][j] = 5000.0
                    dof_props["damping"][j] = 500.0
                elif name in self.hand_config.joint_names:
                    # Finger joints - use config if available
                    idx = self.hand_config.joint_names.index(name)
                    stiffness = (
                        self.hand_config.joint_stiffness[idx]
                        if self.hand_config.joint_stiffness
                        else 800.0
                    )
                    damping = (
                        self.hand_config.joint_damping[idx]
                        if self.hand_config.joint_damping
                        else 80.0
                    )
                    dof_props["stiffness"][j] = stiffness
                    dof_props["damping"][j] = damping
                else:
                    # Mimic/other joints - default low stiffness
                    dof_props["stiffness"][j] = 800.0
                    dof_props["damping"][j] = 80.0

            self._gym.set_actor_dof_properties(env, robot_handle, dof_props)

            # Add object at the configured spawn pose.
            pose_mat = self._initial_object_pose
            quat_xyzw = rotation_matrix_to_quaternion(pose_mat[:3, :3])
            object_pose = gymapi.Transform()
            object_pose.p = gymapi.Vec3(
                float(pose_mat[0, 3]),
                float(pose_mat[1, 3]),
                float(pose_mat[2, 3]),
            )
            object_pose.r = gymapi.Quat(
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
                float(quat_xyzw[3]),
            )

            object_handle = self._gym.create_actor(
                env, object_asset, object_pose, f"object_{i}", i, 2
            )
            self._object_handles.append(object_handle)

            # Set object friction
            shape_props = self._gym.get_actor_rigid_shape_properties(env, object_handle)
            for prop in shape_props:
                prop.friction = self.config.friction
            self._gym.set_actor_rigid_shape_properties(env, object_handle, shape_props)

            # Set object color
            self._gym.set_rigid_body_color(
                env, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.6, 0.2)
            )

        logger.info(f"Created {num_envs} environments")

    def _setup_tensors(self) -> None:
        """Set up tensor handles for GPU pipeline."""
        # Root state tensor (for objects)
        _root_tensor = self._gym.acquire_actor_root_state_tensor(self._sim)
        self._root_states = gymtorch.wrap_tensor(_root_tensor)

        # DOF state tensor (positions, velocities)
        _dof_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        self._dof_states = gymtorch.wrap_tensor(_dof_tensor)

        # DOF position targets
        total_dofs = self._gym.get_sim_dof_count(self._sim)
        self._dof_targets = torch.zeros(
            total_dofs, dtype=torch.float32, device=self.device
        )

        # Get indices
        self._num_envs = len(self._envs)
        self._robot_indices = torch.tensor(
            [
                self._gym.get_actor_index(
                    env, self._robot_handles[i], gymapi.DOMAIN_SIM
                )
                for i, env in enumerate(self._envs)
            ],
            dtype=torch.int32,
            device=self.device,
        )
        self._object_indices = torch.tensor(
            [
                self._gym.get_actor_index(
                    env, self._object_handles[i], gymapi.DOMAIN_SIM
                )
                for i, env in enumerate(self._envs)
            ],
            dtype=torch.int32,
            device=self.device,
        )

        # Total DOFs per robot (virtual 6-DOF + finger DOFs)
        self._dofs_per_robot = self._gym.get_asset_dof_count(
            self._gym.get_actor_asset(self._envs[0], self._robot_handles[0])
        )

        # Get DOF names and create mappings
        dof_names = self._gym.get_actor_dof_names(self._envs[0], self._robot_handles[0])
        self._dof_names = dof_names
        logger.info(f"DOF names: {dof_names}")

        # Map virtual joint names to indices
        self._virtual_dof_indices = []
        for vj_name in self.VIRTUAL_JOINT_NAMES:
            if vj_name in dof_names:
                self._virtual_dof_indices.append(dof_names.index(vj_name))
            else:
                logger.warning(f"Virtual joint {vj_name} not found in URDF")

        # Map actuated finger joints to DOF indices
        self._finger_dof_indices = []
        for joint_name in self.hand_config.joint_names:
            if joint_name in dof_names:
                self._finger_dof_indices.append(dof_names.index(joint_name))
            else:
                logger.warning(f"Finger joint {joint_name} not found in URDF DOFs")

        logger.info(f"Virtual DOF indices: {self._virtual_dof_indices}")
        logger.info(f"Finger DOF indices: {self._finger_dof_indices}")

        # Rigid-body state tensor — needed to read the wrist link's pose for
        # milestone captures. Layout: (total_bodies, 13) = [pos(3), quat(4), lin_vel(3), ang_vel(3)].
        _rb_tensor = self._gym.acquire_rigid_body_state_tensor(self._sim)
        self._rigid_body_states = gymtorch.wrap_tensor(_rb_tensor)

        # Per-env global body index of the hand's wrist link. Indexed by name
        # because the _free URDF prepends virtual joints — body 0 is not the hand.
        wrist_link = self.hand_config.wrist_link
        self._wrist_body_indices = []
        for i, env in enumerate(self._envs):
            idx = self._gym.find_actor_rigid_body_index(
                env, self._robot_handles[i], wrist_link, gymapi.DOMAIN_SIM
            )
            if idx < 0:
                logger.warning(
                    f"env {i}: wrist link '{wrist_link}' not found on robot"
                )
            self._wrist_body_indices.append(idx)

    def _set_object_gravity(self, enable: bool) -> None:
        """
        Toggle gravity on the object actor per env.

        Used to pin the object at its spawn pose during the approach/grasp/
        squeeze phases (gravity OFF) and then let gravity act during the lift
        phase (gravity ON) so the "is the grasp stable?" check is meaningful.
        Without this, the object free-falls to the ground in the first 6 sim
        steps, long before the hand arrives at its grasp pose.
        """
        flag = gymapi.RIGID_BODY_NONE if enable else gymapi.RIGID_BODY_DISABLE_GRAVITY
        for i, env in enumerate(self._envs):
            body_props = self._gym.get_actor_rigid_body_properties(
                env, self._object_handles[i]
            )
            for prop in body_props:
                prop.flags = flag
            self._gym.set_actor_rigid_body_properties(
                env, self._object_handles[i], body_props
            )

    def _capture_milestone(
        self, milestone: str, active_envs: int
    ) -> List[dict]:
        """
        Snapshot hand + object pose and finger joint state for the first
        `active_envs` envs. Refreshes the underlying tensors before reading.

        Args:
            milestone: Label for this capture (e.g. "spawn", "at_grasp").
            active_envs: Number of envs whose state to record.

        Returns:
            List of per-env dicts with keys hand_p, hand_q, obj_p, obj_q, fingers.
            Caller is expected to merge these into per-candidate records.
        """
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)

        rb = self._rigid_body_states  # (total_bodies, 13)
        roots = self._root_states  # (total_actors, 13)
        dof_view = self._dof_states.view(len(self._envs), self._dofs_per_robot, 2)

        snapshots: List[dict] = []
        for i in range(active_envs):
            wrist_idx = self._wrist_body_indices[i]
            if wrist_idx >= 0:
                hand_pq = rb[wrist_idx, :7].cpu().numpy()
                hand_p = hand_pq[:3].tolist()
                hand_q = hand_pq[3:7].tolist()
            else:
                hand_p = [float("nan")] * 3
                hand_q = [float("nan")] * 4

            obj_idx = int(self._object_indices[i].item())
            obj_pq = roots[obj_idx, :7].cpu().numpy()
            obj_p = obj_pq[:3].tolist()
            obj_q = obj_pq[3:7].tolist()

            fingers = [
                float(dof_view[i, dof_idx, 0].item())
                for dof_idx in self._finger_dof_indices
            ]

            snapshots.append(
                {
                    "milestone": milestone,
                    "hand_p": hand_p,
                    "hand_q": hand_q,
                    "obj_p": obj_p,
                    "obj_q": obj_q,
                    "fingers": fingers,
                }
            )
        return snapshots

    def _set_hand_poses_batch(
        self, translations: np.ndarray, rotations: np.ndarray
    ) -> None:
        """
        Set hand poses for active environments in batch.

        Args:
            translations: (B, 3) XYZ positions.
            rotations: (B, 3, 3) rotation matrices.
        """
        num_inputs = len(translations)
        # Note: We don't check against num_envs here, assuming caller handles batch size

        # Calculate stride (DOFs per env)
        stride = self._dofs_per_robot

        # Vectorized update of DOF targets tensor
        # We need to map [B, values] to the flat _dof_targets tensor

        # Translations [B, 3] -> DOF targets
        for i in range(3):
            if i < len(self._virtual_dof_indices):
                dof_idx = self._virtual_dof_indices[i]
                # Indices in global tensor: dof_idx + env_idx * stride
                indices = (
                    torch.arange(num_inputs, device=self.device) * stride + dof_idx
                )
                values = torch.tensor(
                    translations[:, i], dtype=torch.float, device=self.device
                )
                self._dof_targets[indices] = values

        # Rotations -> Euler angles
        eulers = []
        for rot in rotations:
            eulers.append(rotation_matrix_to_euler_xyz_intrinsic(rot))
        eulers = np.array(eulers)  # (B, 3)

        for i in range(3):
            if i + 3 < len(self._virtual_dof_indices):
                dof_idx = self._virtual_dof_indices[i + 3]
                indices = (
                    torch.arange(num_inputs, device=self.device) * stride + dof_idx
                )
                values = torch.tensor(
                    eulers[:, i], dtype=torch.float, device=self.device
                )
                self._dof_targets[indices] = values

    def _set_finger_joints_batch(self, joint_angles: np.ndarray) -> None:
        """
        Set finger joint targets for active environments in batch.

        Args:
            joint_angles: (B, N_joints) target angles.
        """
        num_inputs = len(joint_angles)

        stride = self._dofs_per_robot

        for i, dof_idx in enumerate(self._finger_dof_indices):
            if i < joint_angles.shape[1]:
                indices = (
                    torch.arange(num_inputs, device=self.device) * stride + dof_idx
                )
                values = torch.tensor(
                    joint_angles[:, i], dtype=torch.float, device=self.device
                )
                self._dof_targets[indices] = values

    def step(self, render: bool = True) -> None:
        """Step the simulation forward."""
        # Apply DOF targets
        self._gym.set_dof_position_target_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_targets)
        )

        self._gym.simulate(self._sim)
        self._gym.fetch_results(self._sim, True)

        if render and self._viewer is not None:
            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(self._viewer, self._sim, True)
            self._gym.sync_frame_time(self._sim)

    def validate_batch(
        self,
        translations: np.ndarray,
        rotations: np.ndarray,
        joint_angles: np.ndarray,
        pregrasp_joint_angles: Optional[np.ndarray] = None,
        visualize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Validate a batch of grasps in parallel.

        Args:
            translations: (B, 3) positions.
            rotations: (B, 3, 3) rotation matrices.
            joint_angles: (B, N_joints) joint angles.
            pregrasp_joint_angles: (B, N_joints) pregrasp joint angles.
            visualize: Whether to render the simulation.

        Returns:
            Tuple of (is_stable, lift_heights, final_joints, milestone_traces).
            is_stable: (B,) boolean array.
            lift_heights: (B,) float array.
            final_joints: (B, N_joints) float array.
            milestone_traces: list of length B, each entry is a dict mapping
                milestone name ("spawn" | "approach_started" | "at_grasp" |
                "grasping" | "retreat") to a per-env snapshot dict with keys
                hand_p, hand_q, obj_p, obj_q, fingers.
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call setup() first.")

        batch_size = len(translations)
        # Ensure we have enough environments
        if batch_size > len(self._envs):
            logger.warning(
                f"Batch size {batch_size} > num_envs {len(self._envs)}. Truncating batch."
            )
            batch_size = len(self._envs)
            translations = translations[:batch_size]
            rotations = rotations[:batch_size]
            joint_angles = joint_angles[:batch_size]
            if pregrasp_joint_angles is not None:
                pregrasp_joint_angles = pregrasp_joint_angles[:batch_size]

        # If batch is smaller than num_envs, we only use the first batch_size envs
        # (The others will just sit there idle)
        active_envs = batch_size

        # Initial object pose (world frame) was stored at setup() time.
        initial_object_pos = self._initial_object_pose[:3, 3]
        initial_object_quat_xyzw = rotation_matrix_to_quaternion(
            self._initial_object_pose[:3, :3]
        )
        initial_object_height = float(initial_object_pos[2])

        # Compute approach direction in world frame.
        # approach_axis_wrist = TCP +Z expressed in the wrist frame; this is
        # the direction the hand moves to reach the object. Pregrasp sits
        # `pregrasp_distance` behind the grasp along this axis.
        approach_dirs = np.einsum(
            "bij,j->bi", rotations, self._approach_axis_wrist
        )

        # Pregrasp positions
        pregrasp_translations = (
            translations - approach_dirs * self.config.pregrasp_distance
        )

        # ==========================================
        # Reset simulation
        # ==========================================
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)

        # Reset object states (active envs)
        # Object indices for active envs
        obj_indices = self._object_indices[:active_envs].long()

        # Reset pos/rot/vel to the configured spawn pose.
        self._root_states[obj_indices, :3] = torch.tensor(
            initial_object_pos, dtype=torch.float32, device=self.device
        )
        self._root_states[obj_indices, 3:7] = torch.tensor(
            initial_object_quat_xyzw, dtype=torch.float32, device=self.device
        )
        self._root_states[obj_indices, 7:] = 0.0  # velocities

        self._gym.set_actor_root_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._object_indices[:active_envs]),
            active_envs,
        )

        # Compute pregrasp DOF targets before touching state so we can teleport
        # the hand straight to the pregrasp pose. Driving from world origin to
        # pregrasp with the PD controller left the first rendered frames with
        # the hand lying on the ground plane at (0,0,0), which made visual
        # debugging impossible.
        self._dof_targets.zero_()
        self._set_hand_poses_batch(pregrasp_translations, rotations)
        if pregrasp_joint_angles is not None:
            self._set_finger_joints_batch(pregrasp_joint_angles)
        else:
            self._set_finger_joints_batch(np.zeros_like(joint_angles))  # Open fingers

        # Mirror the target tensor into the DOF state (positions) so the
        # hand starts physically at the pregrasp pose with zero velocity.
        dof_state_view = self._dof_states.view(len(self._envs), self._dofs_per_robot, 2)
        targets_view = self._dof_targets.view(len(self._envs), self._dofs_per_robot)
        dof_state_view[:active_envs, :, 0] = targets_view[:active_envs]
        dof_state_view[:active_envs, :, 1] = 0.0

        self._gym.set_dof_state_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_states)
        )

        # Gravity stays ON for the whole sequence now — the object has already
        # been settled on the ground in setup(), so it's stable at its rest
        # pose when the hand starts moving. Matches the paper's reference
        # pipeline (tests/demo_batch_grasp_validation.py:463-559).

        # Per-env trace: milestone name -> snapshot dict.
        milestone_traces: List[dict] = [dict() for _ in range(active_envs)]

        def record(name: str) -> None:
            snaps = self._capture_milestone(name, active_envs)
            for env_i, snap in enumerate(snaps):
                milestone_traces[env_i][name] = snap

        # Snapshot 1: spawn — hand teleported to pregrasp, object at spawn pose.
        record("spawn")

        # ==========================================
        # Waypoint 1: Pregrasp (hold & settle)
        # ==========================================
        for _ in range(self.config.pregrasp_steps):
            self.step(render=visualize)

        # Snapshot 2: approach_started — end of pregrasp settle, about to move.
        record("approach_started")

        # ==========================================
        # Waypoint 2: Cover (approach)
        # ==========================================
        for i in range(self.config.approach_steps):
            t = (i + 1) / self.config.approach_steps
            current_pos = pregrasp_translations + t * (
                translations - pregrasp_translations
            )
            self._set_hand_poses_batch(current_pos, rotations)
            self.step(render=visualize)

        # Snapshot 3: at_grasp — wrist reached grasp pose, fingers still pregrasp.
        record("at_grasp")

        # ==========================================
        # Waypoint 3: Grasp (close fingers)
        # ==========================================
        for i in range(self.config.grasp_steps):
            t = (i + 1) / self.config.grasp_steps
            current_joints = t * joint_angles
            self._set_finger_joints_batch(current_joints)
            self.step(render=visualize)

        # ==========================================
        # Waypoint 4: Squeeze
        # ==========================================
        squeeze_joints = joint_angles * 1.1
        squeeze_joints = np.clip(squeeze_joints, 0, 2.0)
        self._set_finger_joints_batch(squeeze_joints)

        for _ in range(self.config.squeeze_steps):
            self.step(render=visualize)

        # Snapshot 4: grasping — fingers closed + squeezed, before lift.
        record("grasping")

        # ==========================================
        # Waypoint 5: Lift (gravity has been on throughout; lift tests whether
        # the grasp holds against the already-acting gravity)
        # ==========================================
        lift_translations = translations.copy()
        lift_translations[:, 2] += self.config.lift_height

        for i in range(self.config.lift_steps):
            t = (i + 1) / self.config.lift_steps
            current_pos = translations + t * (lift_translations - translations)
            self._set_hand_poses_batch(current_pos, rotations)
            self.step(render=visualize)

        # Snapshot 5: retreat — final state at end of lift, before success check.
        record("retreat")

        # ==========================================
        # Check result
        # ==========================================
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)

        final_heights = self._root_states[obj_indices, 2].cpu().numpy()
        height_gained = final_heights - initial_object_height
        is_stable = height_gained > self.config.success_threshold

        # Capture final joint angles for all active environments
        # dof_state_view is (num_envs, dofs_per_robot, 2)
        # We want the positions (index 0) for the finger joints only?
        # Or all DOFs including virtual?
        # Usually we want the finger joints.
        # But let's return all DOFs and let the caller filter.
        # Wait, the caller expects 'joint_angles' which usually matches hand_config.joint_names order.
        # hand_config.joint_names maps to _finger_dof_indices.

        all_dof_pos = dof_state_view[:active_envs, :, 0].cpu().numpy()  # (B, num_dofs)

        # Extract only the finger joints in the correct order
        final_joints = np.zeros((active_envs, len(self._finger_dof_indices)))
        for i, dof_idx in enumerate(self._finger_dof_indices):
            final_joints[:, i] = all_dof_pos[:, dof_idx]

        return is_stable, final_heights, final_joints, milestone_traces

    def validate_single_grasp(
        self,
        translation: np.ndarray,
        rotation: np.ndarray,
        joint_angles: np.ndarray,
        visualize: bool = False,
    ) -> Tuple[bool, float, dict]:
        """
        Validate a single grasp candidate using the 5-waypoint system.

        Waypoints (following DexGraspNet2.0 paper):
        1. Pregrasp: fingers open, hand 10cm back from grasp pose
        2. Cover: move to grasp position, fingers still open
        3. Grasp: close fingers to target angles
        4. Squeeze: additional finger squeeze, enable object gravity
        5. Lift: move hand up, check if object follows

        Args:
            translation: Wrist position at grasp pose (3,).
            rotation: Wrist rotation matrix (3, 3).
            joint_angles: Target joint angles for grasping.
            visualize: Show grasp execution if viewer available.

        Returns:
            Tuple of (is_stable, lift_height, milestone_trace).
            milestone_trace is a dict keyed by milestone name whose values
            are snapshots (hand_p, hand_q, obj_p, obj_q, fingers).
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call setup() first.")

        # Initial object pose (world frame) from setup().
        initial_object_pos = self._initial_object_pose[:3, 3]
        initial_object_quat_xyzw = rotation_matrix_to_quaternion(
            self._initial_object_pose[:3, :3]
        )
        initial_object_height = float(initial_object_pos[2])

        # Compute approach direction in world frame using TCP-derived axis.
        approach_dir = rotation @ self._approach_axis_wrist

        # Pregrasp position: pregrasp_distance back along approach direction.
        pregrasp_translation = (
            translation - approach_dir * self.config.pregrasp_distance
        )

        # ==========================================
        # Reset simulation
        # ==========================================
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)

        # Reset object to the configured spawn pose.
        self._root_states[self._object_indices[0], :3] = torch.tensor(
            initial_object_pos, dtype=torch.float32, device=self.device
        )
        self._root_states[self._object_indices[0], 3:7] = torch.tensor(
            initial_object_quat_xyzw, dtype=torch.float32, device=self.device
        )
        self._root_states[self._object_indices[0], 7:] = 0  # Zero velocities

        self._gym.set_actor_root_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._object_indices[:1]),
            1,
        )

        # Compute pregrasp DOF targets, then teleport state to match so the
        # hand starts physically at the pregrasp pose (no fly-in from origin).
        self._dof_targets.zero_()
        self._set_hand_pose(pregrasp_translation, rotation)
        self._set_finger_joints(np.zeros_like(joint_angles))  # Fingers open

        dof_state = self._dof_states.view(self._num_envs, self._dofs_per_robot, 2)
        targets_view = self._dof_targets.view(self._num_envs, self._dofs_per_robot)
        dof_state[0, :, 0] = targets_view[0]
        dof_state[0, :, 1] = 0
        self._gym.set_dof_state_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_states)
        )

        # Gravity on throughout — object pre-settled in setup().

        # Per-milestone snapshot collection for this single candidate.
        milestone_trace: dict = {}

        def record(name: str) -> None:
            snaps = self._capture_milestone(name, 1)
            milestone_trace[name] = snaps[0]

        # Snapshot 1: spawn.
        record("spawn")

        # ==========================================
        # Waypoint 1: Pregrasp (hold & settle)
        # ==========================================
        for _ in range(self.config.pregrasp_steps):
            self.step(render=visualize)

        # Snapshot 2: approach_started.
        record("approach_started")

        # ==========================================
        # Waypoint 2: Cover (approach to grasp pose)
        # ==========================================
        for i in range(self.config.approach_steps):
            t = (i + 1) / self.config.approach_steps
            # Interpolate position
            current_pos = pregrasp_translation + t * (
                translation - pregrasp_translation
            )
            self._set_hand_pose(current_pos, rotation)
            self.step(render=visualize)

        # Snapshot 3: at_grasp.
        record("at_grasp")

        # ==========================================
        # Waypoint 3: Grasp (close fingers)
        # ==========================================
        for i in range(self.config.grasp_steps):
            t = (i + 1) / self.config.grasp_steps
            current_joints = t * joint_angles
            self._set_finger_joints(current_joints)
            self.step(render=visualize)

        # ==========================================
        # Waypoint 4: Squeeze (additional grip)
        # ==========================================
        squeeze_joints = joint_angles * 1.1  # 10% extra squeeze
        squeeze_joints = np.clip(squeeze_joints, 0, 2.0)  # Clip to reasonable range
        self._set_finger_joints(squeeze_joints)

        for _ in range(self.config.squeeze_steps):
            self.step(render=visualize)

        # Snapshot 4: grasping.
        record("grasping")

        # ==========================================
        # Waypoint 5: Lift (gravity always on; object was pre-settled)
        # ==========================================
        lift_translation = translation.copy()
        lift_translation[2] += self.config.lift_height

        for i in range(self.config.lift_steps):
            t = (i + 1) / self.config.lift_steps
            current_pos = translation + t * (lift_translation - translation)
            self._set_hand_pose(current_pos, rotation)
            self.step(render=visualize)

        # Snapshot 5: retreat.
        record("retreat")

        # ==========================================
        # Check result
        # ==========================================
        self._gym.refresh_actor_root_state_tensor(self._sim)
        final_object_height = float(self._root_states[self._object_indices[0], 2].cpu())

        # Success if object lifted at least 3cm above initial position
        height_gained = final_object_height - initial_object_height
        is_stable = height_gained > self.config.success_threshold

        return is_stable, final_object_height, milestone_trace

    def _set_hand_pose(self, translation: np.ndarray, rotation: np.ndarray) -> None:
        """
        Set hand pose via virtual 6-DOF joint targets.

        Args:
            translation: XYZ position.
            rotation: 3x3 rotation matrix.
        """
        # Set translation DOFs
        if len(self._virtual_dof_indices) >= 3:
            self._dof_targets[self._virtual_dof_indices[0]] = translation[0]
            self._dof_targets[self._virtual_dof_indices[1]] = translation[1]
            self._dof_targets[self._virtual_dof_indices[2]] = translation[2]

        # Convert rotation matrix to Euler angles (intrinsic XYZ for URDF joint chain)
        euler = rotation_matrix_to_euler_xyz_intrinsic(rotation)

        # Set rotation DOFs
        if len(self._virtual_dof_indices) >= 6:
            self._dof_targets[self._virtual_dof_indices[3]] = euler[0]  # rx
            self._dof_targets[self._virtual_dof_indices[4]] = euler[1]  # ry
            self._dof_targets[self._virtual_dof_indices[5]] = euler[2]  # rz

    def _set_finger_joints(self, joint_angles: np.ndarray) -> None:
        """
        Set finger joint targets.

        Args:
            joint_angles: Target angles for actuated finger joints.
        """
        for i, dof_idx in enumerate(self._finger_dof_indices):
            if i < len(joint_angles):
                self._dof_targets[dof_idx] = joint_angles[i]

    def cleanup(self) -> None:
        """Clean up simulation resources."""
        if self._viewer is not None:
            self._gym.destroy_viewer(self._viewer)
            self._viewer = None

        if self._sim is not None:
            self._gym.destroy_sim(self._sim)
            self._sim = None

        # Drop references to destroyed-sim-owned objects so a subsequent
        # setup() call starts from a clean slate. Leaving these around
        # pointing at freed PhysX memory is what caused the segfaults
        # when processing multiple objects in one process.
        self._envs = []
        self._robot_handles = []
        self._object_handles = []
        self._root_states = None
        self._dof_states = None
        self._dof_targets = None
        self._rigid_body_states = None
        self._wrist_body_indices = []
        self._initial_object_pose = None

        self._initialized = False
        logger.info("Simulation cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
