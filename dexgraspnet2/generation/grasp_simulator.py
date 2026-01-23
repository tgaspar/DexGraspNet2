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

# Set environment variables before importing isaacgym
os.environ.setdefault('VK_ICD_FILENAMES', '/etc/vulkan/icd.d/nvidia_icd.json')
os.environ.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')

from isaacgym import gymapi, gymtorch
import torch

from dexgraspnet2.configs.hand_config import HandConfig
from dexgraspnet2.utils.geometric_conversions import (
    rotation_matrix_to_euler_xyz_intrinsic,
    rotation_matrix_to_quaternion,
)

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
    pregrasp_steps: int = 30     # Steps at pregrasp position
    approach_steps: int = 60     # Steps to approach (pregrasp -> cover)
    grasp_steps: int = 60        # Steps to close fingers
    squeeze_steps: int = 30      # Additional squeeze
    lift_steps: int = 90         # Steps to lift
    settle_steps: int = 30       # Steps to let physics settle

    # Distances
    pregrasp_distance: float = 0.10  # 10cm back from grasp pose
    lift_height: float = 0.10        # Lift 10cm
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
        "x_joint", "y_joint", "z_joint",
        "x_rotation_joint", "y_rotation_joint", "z_rotation_joint"
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
    ) -> None:
        """
        Set up the simulation environment.

        Args:
            object_mesh_path: Path to object mesh file. If None, uses a box.
            object_scale: Scale factor for the object mesh.
            num_envs: Number of parallel environments.
        """
        logger.info(f"Setting up simulation with {num_envs} environment(s)")

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
                self._gym.viewer_camera_look_at(
                    self._viewer, None, cam_pos, cam_target
                )

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
        logger.info("Simulation setup complete")

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

        if mesh_path is None:
            # Create a simple box for testing
            object_asset = self._gym.create_box(
                self._sim, 0.05, 0.05, 0.05, asset_options
            )
            logger.info("Created box object (no mesh provided)")
        else:
            mesh_path = Path(mesh_path)
            if mesh_path.suffix.lower() == '.urdf':
                object_asset = self._gym.load_asset(
                    self._sim, str(mesh_path.parent), mesh_path.name, asset_options
                )
            else:
                # For OBJ/other formats, create a URDF wrapper
                # For now, fall back to box
                logger.warning(
                    f"Mesh format {mesh_path.suffix} not directly supported, using box"
                )
                object_asset = self._gym.create_box(
                    self._sim, 0.05, 0.05, 0.05, asset_options
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
                else:
                    # Finger joints
                    dof_props["stiffness"][j] = 800.0
                    dof_props["damping"][j] = 80.0

            self._gym.set_actor_dof_properties(env, robot_handle, dof_props)

            # Add object
            object_pose = gymapi.Transform()
            object_pose.p = gymapi.Vec3(0.0, 0.0, 0.05)
            object_pose.r = gymapi.Quat(0, 0, 0, 1)

            object_handle = self._gym.create_actor(
                env, object_asset, object_pose, f"object_{i}", i, 2
            )
            self._object_handles.append(object_handle)

            # Set object friction
            shape_props = self._gym.get_actor_rigid_shape_properties(
                env, object_handle
            )
            for prop in shape_props:
                prop.friction = self.config.friction
            self._gym.set_actor_rigid_shape_properties(
                env, object_handle, shape_props
            )

            # Set object color
            self._gym.set_rigid_body_color(
                env, object_handle, 0, gymapi.MESH_VISUAL,
                gymapi.Vec3(0.2, 0.6, 0.2)
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
            [self._gym.get_actor_index(env, self._robot_handles[i], gymapi.DOMAIN_SIM)
             for i, env in enumerate(self._envs)],
            dtype=torch.int32, device=self.device
        )
        self._object_indices = torch.tensor(
            [self._gym.get_actor_index(env, self._object_handles[i], gymapi.DOMAIN_SIM)
             for i, env in enumerate(self._envs)],
            dtype=torch.int32, device=self.device
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

    def validate_single_grasp(
        self,
        translation: np.ndarray,
        rotation: np.ndarray,
        joint_angles: np.ndarray,
        visualize: bool = False,
    ) -> Tuple[bool, float]:
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
            Tuple of (is_stable, lift_height).
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call setup() first.")

        # Record initial object height
        initial_object_height = 0.05  # Object starts at z=0.05

        # Compute approach direction (along -Z of hand frame by default)
        # For palm-down grasp, approach from above
        approach_dir = rotation @ np.array([0, 0, -1])

        # Pregrasp position: 10cm back along approach direction
        pregrasp_translation = translation - approach_dir * self.config.pregrasp_distance

        # ==========================================
        # Reset simulation
        # ==========================================
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)

        # Reset object position and disable gravity initially
        self._root_states[self._object_indices[0], :3] = torch.tensor(
            [0.0, 0.0, initial_object_height], device=self.device
        )
        self._root_states[self._object_indices[0], 3:7] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.device
        )
        self._root_states[self._object_indices[0], 7:] = 0  # Zero velocities

        self._gym.set_actor_root_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._object_indices[:1]),
            1
        )

        # Reset all DOFs to zero
        self._dof_targets.zero_()

        # Reset DOF states
        dof_state = self._dof_states.view(self._num_envs, self._dofs_per_robot, 2)
        dof_state[0, :, 0] = 0  # positions
        dof_state[0, :, 1] = 0  # velocities
        self._gym.set_dof_state_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_states)
        )

        # ==========================================
        # Waypoint 1: Pregrasp
        # ==========================================
        self._set_hand_pose(pregrasp_translation, rotation)
        self._set_finger_joints(np.zeros_like(joint_angles))  # Fingers open

        for _ in range(self.config.pregrasp_steps):
            self.step(render=visualize)

        # ==========================================
        # Waypoint 2: Cover (approach to grasp pose)
        # ==========================================
        for i in range(self.config.approach_steps):
            t = (i + 1) / self.config.approach_steps
            # Interpolate position
            current_pos = pregrasp_translation + t * (translation - pregrasp_translation)
            self._set_hand_pose(current_pos, rotation)
            self.step(render=visualize)

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

        # ==========================================
        # Waypoint 5: Lift
        # ==========================================
        lift_translation = translation.copy()
        lift_translation[2] += self.config.lift_height

        for i in range(self.config.lift_steps):
            t = (i + 1) / self.config.lift_steps
            current_pos = translation + t * (lift_translation - translation)
            self._set_hand_pose(current_pos, rotation)
            self.step(render=visualize)

        # ==========================================
        # Check result
        # ==========================================
        self._gym.refresh_actor_root_state_tensor(self._sim)
        final_object_height = float(self._root_states[self._object_indices[0], 2].cpu())

        # Success if object lifted at least 3cm above initial position
        height_gained = final_object_height - initial_object_height
        is_stable = height_gained > self.config.success_threshold

        return is_stable, final_object_height

    def cleanup(self) -> None:
        """Clean up simulation resources."""
        if self._viewer is not None:
            self._gym.destroy_viewer(self._viewer)
            self._viewer = None

        if self._sim is not None:
            self._gym.destroy_sim(self._sim)
            self._sim = None

        self._initialized = False
        logger.info("Simulation cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
