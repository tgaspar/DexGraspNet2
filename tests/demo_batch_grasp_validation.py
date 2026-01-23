#!/usr/bin/env python3
"""
Demo: Batch grasp validation following the original DexGraspNet2 approach.

Key insight from original: Disable object gravity during approach/grasp,
enable only after squeeze to test if grasp holds.
"""

import os
import sys
from pathlib import Path
import argparse

os.environ['VK_ICD_FILENAMES'] = '/etc/vulkan/icd.d/nvidia_icd.json'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from isaacgym import gymapi, gymtorch
import torch
import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, Optional
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    qx, qy, qz, qw = quat
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


@dataclass
class SimConfig:
    """Simulation configuration."""
    dt: float = 1.0 / 60.0
    substeps: int = 2

    # Waypoint steps (from original config)
    pregrasp_steps: int = 30
    approach_steps: int = 60
    grasp_steps: int = 60
    squeeze_steps: int = 30
    lift_steps: int = 90

    pregrasp_distance: float = 0.10
    lift_height: float = 0.10
    success_threshold: float = 0.015  # Lower threshold (1.5cm) for partial success detection


class GraspValidator:
    """Validates grasps following original DexGraspNet2 approach."""

    VIRTUAL_JOINTS = ["x_joint", "y_joint", "z_joint",
                      "x_rotation_joint", "y_rotation_joint", "z_rotation_joint"]

    # Links that should contact the object (fingertips and palm)
    CONTACT_LINKS = [
        "thumb_fingertip", "index_fingertip", "middle_fingertip", "ring_fingertip",
        "thumb_tip", "index_tip", "middle_tip", "ring_tip",
        "palm_lower", "mcp_joint", "pip", "dip", "fingertip"  # Generic names
    ]

    def __init__(self, num_envs: int = 1, device: str = "cuda:0",
                 headless: bool = False, config: SimConfig = None):
        self.num_envs = num_envs
        self.device = device
        self.headless = headless
        self.config = config or SimConfig()
        self._gym = None
        self._sim = None
        self._viewer = None

    def setup(self, robot_urdf: Path, object_urdf: Path,
              object_position: np.ndarray, object_quaternion: np.ndarray):
        """Initialize simulation."""
        self._object_init_pos = object_position
        self._object_init_quat = object_quaternion

        self._gym = gymapi.acquire_gym()

        # Sim params
        sim_params = gymapi.SimParams()
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
        sim_params.dt = self.config.dt
        sim_params.substeps = self.config.substeps
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 2
        # Enable contact collection for contact-based validation
        sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS

        device_id = int(self.device.split(":")[-1])
        self._sim = self._gym.create_sim(device_id, device_id, gymapi.SIM_PHYSX, sim_params)

        # Ground
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self._gym.add_ground(self._sim, plane_params)

        # Viewer
        if not self.headless:
            cam_props = gymapi.CameraProperties()
            cam_props.width = 1280
            cam_props.height = 720
            self._viewer = self._gym.create_viewer(self._sim, cam_props)
            cam_pos = gymapi.Vec3(0.5, -0.5, 0.5)
            cam_target = gymapi.Vec3(0, 0, 0.1)
            self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)

        # Load assets
        self._load_assets(robot_urdf, object_urdf)
        self._create_envs()
        self._gym.prepare_sim(self._sim)
        self._setup_tensors()

        logger.info(f"Simulator ready: {self.num_envs} envs, {self._num_dofs} DOFs per robot")

    def _load_assets(self, robot_urdf: Path, object_urdf: Path):
        """Load robot and object assets."""
        # Robot
        robot_opts = gymapi.AssetOptions()
        robot_opts.fix_base_link = True
        robot_opts.disable_gravity = True
        robot_opts.armature = 0.01
        self._robot_asset = self._gym.load_asset(
            self._sim, str(robot_urdf.parent), robot_urdf.name, robot_opts)
        self._num_dofs = self._gym.get_asset_dof_count(self._robot_asset)

        # Object
        obj_opts = gymapi.AssetOptions()
        obj_opts.density = 500.0
        self._object_asset = self._gym.load_asset(
            self._sim, str(object_urdf.parent), object_urdf.name, obj_opts)

    def _create_envs(self):
        """Create environments."""
        spacing = 1.0
        lower = gymapi.Vec3(-spacing/2, -spacing/2, 0)
        upper = gymapi.Vec3(spacing/2, spacing/2, spacing)

        self._envs = []
        self._robot_handles = []
        self._object_handles = []

        for i in range(self.num_envs):
            env = self._gym.create_env(self._sim, lower, upper, int(np.sqrt(self.num_envs)))
            self._envs.append(env)

            # Robot at origin
            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(0, 0, 0)
            robot_handle = self._gym.create_actor(env, self._robot_asset, robot_pose, "robot", i, 1)
            self._robot_handles.append(robot_handle)

            # Set robot drive properties
            props = self._gym.get_actor_dof_properties(env, robot_handle)
            props["driveMode"].fill(gymapi.DOF_MODE_POS)
            dof_names = self._gym.get_actor_dof_names(env, robot_handle)
            for j, name in enumerate(dof_names):
                if name in self.VIRTUAL_JOINTS:
                    props["stiffness"][j] = 4000.0
                    props["damping"][j] = 400.0
                else:
                    props["stiffness"][j] = 1000.0
                    props["damping"][j] = 100.0
            self._gym.set_actor_dof_properties(env, robot_handle, props)

            # Object
            obj_pose = gymapi.Transform()
            obj_pose.p = gymapi.Vec3(*self._object_init_pos)
            obj_pose.r = gymapi.Quat(*self._object_init_quat)
            obj_handle = self._gym.create_actor(env, self._object_asset, obj_pose, "object", i, 2)
            self._object_handles.append(obj_handle)

            # Object friction
            shape_props = self._gym.get_actor_rigid_shape_properties(env, obj_handle)
            for p in shape_props:
                p.friction = 1.0
            self._gym.set_actor_rigid_shape_properties(env, obj_handle, shape_props)

        self._dof_names = self._gym.get_actor_dof_names(self._envs[0], self._robot_handles[0])

    def _setup_tensors(self):
        """Setup GPU tensors."""
        self._root_tensor = gymtorch.wrap_tensor(
            self._gym.acquire_actor_root_state_tensor(self._sim))
        self._dof_state_tensor = gymtorch.wrap_tensor(
            self._gym.acquire_dof_state_tensor(self._sim))
        self._rigid_body_state_tensor = gymtorch.wrap_tensor(
            self._gym.acquire_rigid_body_state_tensor(self._sim))
        self._contact_force_tensor = gymtorch.wrap_tensor(
            self._gym.acquire_net_contact_force_tensor(self._sim))

        self._dof_targets = torch.zeros(
            self.num_envs * self._num_dofs, dtype=torch.float32, device=self.device)

        # Index mappings
        self._virtual_dof_idx = [self._dof_names.index(n) for n in self.VIRTUAL_JOINTS
                                  if n in self._dof_names]
        self._finger_dof_idx = [i for i, n in enumerate(self._dof_names)
                                 if n not in self.VIRTUAL_JOINTS]

        # Create mapping from grasp joint index to DOF index
        # Grasp data has j0-j15 in order, but URDF DOFs may be in different order
        self._grasp_to_dof_map = {}
        for dof_idx, name in enumerate(self._dof_names):
            if name.startswith('j') and name[1:].isdigit():
                grasp_joint_idx = int(name[1:])  # e.g., 'j5' -> 5
                self._grasp_to_dof_map[grasp_joint_idx] = dof_idx

        logger.info(f"Grasp to DOF mapping: {self._grasp_to_dof_map}")

        self._robot_indices = torch.tensor(
            [self._gym.get_actor_index(e, self._robot_handles[i], gymapi.DOMAIN_SIM)
             for i, e in enumerate(self._envs)], dtype=torch.int32, device=self.device)
        self._object_indices = torch.tensor(
            [self._gym.get_actor_index(e, self._object_handles[i], gymapi.DOMAIN_SIM)
             for i, e in enumerate(self._envs)], dtype=torch.int32, device=self.device)

        # Get rigid body indices for contact checking
        self._setup_contact_indices()

    def _setup_contact_indices(self):
        """Setup indices for contact checking between fingers and object."""
        # Get all rigid body names for robot
        robot_body_names = self._gym.get_actor_rigid_body_names(
            self._envs[0], self._robot_handles[0])
        logger.info(f"Robot bodies: {robot_body_names}")

        # Find finger body indices (any body that's not the base/virtual chain)
        # All bodies after the virtual 6-DOF chain are finger links
        self._finger_body_local_idx = []
        for i, name in enumerate(robot_body_names):
            name_lower = name.lower()
            # Include all non-base links as potential contact links
            if any(kw in name_lower for kw in ['finger', 'tip', 'pip', 'dip', 'mcp', 'thumb', 'index', 'middle', 'ring', 'palm']):
                self._finger_body_local_idx.append(i)

        # If no specific finger names found, use all bodies except first 6 (virtual chain)
        if not self._finger_body_local_idx:
            self._finger_body_local_idx = list(range(6, len(robot_body_names)))

        logger.info(f"Finger body indices: {self._finger_body_local_idx}")
        logger.info(f"Finger body names: {[robot_body_names[i] for i in self._finger_body_local_idx]}")

        # Get global rigid body indices per env
        self._finger_body_indices = []  # List of lists: [env][finger_bodies]
        self._object_body_indices = []  # List: [env]

        num_robot_bodies = self._gym.get_actor_rigid_body_count(
            self._envs[0], self._robot_handles[0])

        for i, env in enumerate(self._envs):
            # Robot body start index
            robot_rb_start = self._gym.get_actor_rigid_body_index(
                env, self._robot_handles[i], 0, gymapi.DOMAIN_SIM)
            finger_indices = [robot_rb_start + idx for idx in self._finger_body_local_idx]
            self._finger_body_indices.append(finger_indices)

            # Object body index
            obj_rb_idx = self._gym.get_actor_rigid_body_index(
                env, self._object_handles[i], 0, gymapi.DOMAIN_SIM)
            self._object_body_indices.append(obj_rb_idx)

    def _set_object_gravity(self, enable: bool):
        """Enable or disable gravity on objects."""
        for i, env in enumerate(self._envs):
            body_props = self._gym.get_actor_rigid_body_properties(env, self._object_handles[i])
            for prop in body_props:
                prop.flags = gymapi.RIGID_BODY_NONE if enable else gymapi.RIGID_BODY_DISABLE_GRAVITY
            self._gym.set_actor_rigid_body_properties(env, self._object_handles[i], body_props)

    def _get_finger_contacts(self) -> np.ndarray:
        """
        Check if fingers have contact forces (indicating contact with object).

        Returns:
            Tuple of (has_contact, contact_counts):
                - has_contact: Boolean array [num_envs] indicating if at least 2 fingers have contact
                - contact_counts: Int array [num_envs] with number of finger contacts per env
        """
        self._gym.refresh_net_contact_force_tensor(self._sim)

        # Single GPU->CPU transfer to avoid sync bottleneck
        contact_forces_cpu = self._contact_force_tensor.cpu().numpy()

        has_contact = np.zeros(self.num_envs, dtype=bool)
        contact_counts = np.zeros(self.num_envs, dtype=int)

        for i in range(self.num_envs):
            for finger_idx in self._finger_body_indices[i]:
                force = contact_forces_cpu[finger_idx]  # CPU indexing (fast)
                force_magnitude = np.linalg.norm(force)
                if force_magnitude > 0.1:  # Threshold for meaningful contact
                    contact_counts[i] += 1
            has_contact[i] = contact_counts[i] >= 2  # At least 2 finger contacts

        return has_contact, contact_counts

    def _get_hand_height(self) -> np.ndarray:
        """Get hand wrist/palm height (z position of first robot body after virtual chain)."""
        self._gym.refresh_rigid_body_state_tensor(self._sim)

        # Single GPU->CPU transfer to avoid sync bottleneck
        rb_state_cpu = self._rigid_body_state_tensor.cpu().numpy()

        heights = np.zeros(self.num_envs)
        for i in range(self.num_envs):
            # Use first finger body as reference for hand height
            if self._finger_body_indices[i]:
                body_idx = self._finger_body_indices[i][0]
                heights[i] = rb_state_cpu[body_idx, 2]  # CPU indexing (fast)

        return heights
    
    def _rotation_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler XYZ (intrinsic) for URDF joint chain."""
        cy = np.sqrt(R[0,0]**2 + R[0,1]**2)
        if cy > 1e-6:
            rx = np.arctan2(-R[1,2], R[2,2])
            ry = np.arctan2(R[0,2], cy)
            rz = np.arctan2(-R[0,1], R[0,0])
        else:
            rx = np.arctan2(-R[2,1], R[1,1])
            ry = np.arctan2(R[0,2], cy)
            rz = 0
        return np.array([rx, ry, rz])

    def _compute_dof_targets(self, translation: np.ndarray, rotation: np.ndarray,
                              joint_angles: np.ndarray) -> np.ndarray:
        """Compute DOF target array from grasp pose.

        Args:
            translation: (3,) wrist position
            rotation: (3,3) wrist rotation matrix
            joint_angles: (16,) joint angles in grasp data order (j0, j1, ..., j15)
        """
        targets = np.zeros(self._num_dofs)

        # Virtual joints: translation
        for j, idx in enumerate(self._virtual_dof_idx[:3]):
            targets[idx] = translation[j]

        # Virtual joints: rotation (Euler)
        euler = self._rotation_to_euler(rotation)
        for j, idx in enumerate(self._virtual_dof_idx[3:6]):
            targets[idx] = euler[j]

        # Finger joints - use the grasp-to-DOF mapping
        for grasp_idx, dof_idx in self._grasp_to_dof_map.items():
            if grasp_idx < len(joint_angles):
                targets[dof_idx] = joint_angles[grasp_idx]

        return targets

    def _set_dof_targets_all(self, targets_per_env: np.ndarray):
        """Set DOF targets for all environments."""
        for i in range(self.num_envs):
            base = i * self._num_dofs
            for j in range(self._num_dofs):
                self._dof_targets[base + j] = float(targets_per_env[i, j])

    def _step(self):
        """Step simulation."""
        self._gym.set_dof_position_target_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_targets))
        self._gym.simulate(self._sim)
        self._gym.fetch_results(self._sim, True)
        if self._viewer:
            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(self._viewer, self._sim, True)
            self._gym.sync_frame_time(self._sim)

    def _get_object_xyz(self) -> np.ndarray:
        """Get current object positions."""
        self._gym.refresh_actor_root_state_tensor(self._sim)

        # Single GPU->CPU transfer to avoid sync bottleneck
        root_cpu = self._root_tensor.cpu().numpy()

        positions = np.zeros((self.num_envs, 3))
        for i in range(self.num_envs):
            idx = self._object_indices[i]
            positions[i] = root_cpu[idx, :3]  # Get x,y,z in one indexing op (fast)
        return positions

    def _get_object_rot_mat(self) -> np.ndarray:
        """Get current object rotation matrices."""
        self._gym.refresh_actor_root_state_tensor(self._sim)

        # Single GPU->CPU transfer to avoid sync bottleneck
        root_cpu = self._root_tensor.cpu().numpy()

        rotations = np.zeros((self.num_envs, 3, 3))
        for i in range(self.num_envs):
            quat = root_cpu[self._object_indices[i], 3:7]  # CPU indexing (fast)
            rotations[i] = quat_to_rot_matrix(quat)
        return rotations

    def _get_object_heights(self) -> np.ndarray:
        """Get current object heights."""
        positions = self._get_object_xyz()
        heights = positions[:, 2]

        return heights

    def _reset_simulation(self, pregrasp_targets: np.ndarray):
        """Reset simulation to initial state with robot at pregrasp."""
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)

        # Reset objects
        for i in range(self.num_envs):
            self._root_tensor[self._object_indices[i], :3] = torch.tensor(
                self._object_init_pos, device=self.device)
            self._root_tensor[self._object_indices[i], 3:7] = torch.tensor(
                self._object_init_quat, device=self.device)
            self._root_tensor[self._object_indices[i], 7:] = 0

        self._gym.set_actor_root_state_tensor_indexed(
            self._sim, gymtorch.unwrap_tensor(self._root_tensor),
            gymtorch.unwrap_tensor(self._object_indices), self.num_envs)

        # Set robot DOF positions AND targets to pregrasp
        for i in range(self.num_envs):
            base = i * self._num_dofs
            for j in range(self._num_dofs):
                self._dof_state_tensor[base + j, 0] = pregrasp_targets[i, j]  # position
                self._dof_state_tensor[base + j, 1] = 0.0  # velocity
                self._dof_targets[base + j] = pregrasp_targets[i, j]

        self._gym.set_dof_state_tensor(self._sim, gymtorch.unwrap_tensor(self._dof_state_tensor))

    def validate_grasps(self, translations: np.ndarray, rotations: np.ndarray,
                        joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate grasps:
        1. Let object settle on ground with gravity
        2. Adjust grasp targets to settled object position
        3. Move through pregrasp -> approach -> grasp -> squeeze -> lift
        """
        batch_size = min(len(translations), self.num_envs)
        cfg = self.config

        # === PHASE 0: Let object settle with gravity ===
        # First, reset with a dummy pregrasp (hand far away)
        dummy_pregrasp = np.zeros((batch_size, self._num_dofs))
        # Place hand high above to not interfere
        for i in range(batch_size):
            for j, idx in enumerate(self._virtual_dof_idx[:3]):
                dummy_pregrasp[i, idx] = [0.0, 0.0, 0.5][j]  # Hand at z=0.5

        self._reset_simulation(dummy_pregrasp)
        self._set_object_gravity(True)  # Gravity ON from start

        # Let object settle
        logger.info("Letting object settle with gravity...")
        for _ in range(60):  # 1 second at 60Hz
            self._step()

        # Measure settled object position
        # settled_heights = self._get_object_heights()
        settled_positions = self._get_object_xyz()
        settled_rotations = self._get_object_rot_mat()
        # logger.info(f"Object settled at heights: {settled_heights}")

        # Compute offsets: how much the object has moved from initial position
        pos_offset = settled_positions - self._object_init_pos
        rot_offset = np.zeros((batch_size, 3, 3))
        for i in range(batch_size):
            rot_offset[i] = np.linalg.inv(quat_to_rot_matrix(self._object_init_quat)) @ settled_rotations[i]

        # Adjust grasp translations to account for settled position
        adjusted_translations = translations.copy()
        for i in range(batch_size):
            adjusted_translations[i] = rot_offset[i] @ translations[i] + pos_offset[i]
            # adjusted_translations[i] = translations[i] + pos_offset[i]

        # Now adjust rotations
        adjusted_rotations = rotations.copy()
        for i in range(batch_size):
            adjusted_rotations[i] = rot_offset[i] @ rotations[i]
            # adjusted_rotations[i] = rotations[i]

        # Compute all waypoint targets with adjusted positions
        pregrasp_targets = np.zeros((batch_size, self._num_dofs))
        grasp_targets = np.zeros((batch_size, self._num_dofs))
        squeeze_targets = np.zeros((batch_size, self._num_dofs))
        lift_targets = np.zeros((batch_size, self._num_dofs))

        for i in range(batch_size):
            # Approach direction is -Z of hand frame
            approach_dir = -adjusted_rotations[i][:, 2]

            # Pregrasp: back from grasp position, fingers open
            pregrasp_pos = adjusted_translations[i] - approach_dir * cfg.pregrasp_distance
            pregrasp_targets[i] = self._compute_dof_targets(
                pregrasp_pos, adjusted_rotations[i], np.zeros_like(joint_angles[i]))

            # Grasp: at grasp position, fingers closed
            grasp_targets[i] = self._compute_dof_targets(
                adjusted_translations[i], adjusted_rotations[i], joint_angles[i])

            # Squeeze: same position, fingers tighter
            squeeze_angles = np.clip(joint_angles[i] * 1.2, 0, 2.5)
            squeeze_targets[i] = self._compute_dof_targets(
                adjusted_translations[i], adjusted_rotations[i], squeeze_angles)

            # Lift: move up
            lift_pos = adjusted_translations[i].copy()
            lift_pos[2] += cfg.lift_height
            lift_targets[i] = self._compute_dof_targets(
                lift_pos, adjusted_rotations[i], squeeze_angles)

        logger.info(f"Validating {batch_size} grasps")

        # === PHASE 1: Move hand to pregrasp ===
        # Set hand to pregrasp position (object already settled)
        self._set_dof_targets_all(pregrasp_targets)
        for _ in range(30):  # Let hand reach pregrasp
            self._step()

        logger.info(f"At pregrasp - object heights: {self._get_object_heights()}")

        # === PHASE 2: Approach (pregrasp -> grasp position, fingers stay open) ===
        approach_targets = pregrasp_targets.copy()
        for t in np.linspace(0, 1, cfg.approach_steps):
            # Interpolate position only, keep fingers open
            for i in range(batch_size):
                for j in self._virtual_dof_idx:
                    approach_targets[i, j] = (1-t) * pregrasp_targets[i, j] + t * grasp_targets[i, j]
            self._set_dof_targets_all(approach_targets)
            self._step()
        logger.info(f"After approach: {self._get_object_heights()}")

        # === PHASE 3: Close fingers ===
        close_targets = approach_targets.copy()
        for t in np.linspace(0, 1, cfg.grasp_steps):
            for i in range(batch_size):
                for j in self._finger_dof_idx:
                    close_targets[i, j] = t * grasp_targets[i, j]
            self._set_dof_targets_all(close_targets)
            self._step()
        logger.info(f"After grasp: {self._get_object_heights()}")

        # === PHASE 4: Squeeze ===
        self._set_dof_targets_all(squeeze_targets)
        for _ in range(cfg.squeeze_steps):
            self._step()

        # Check contacts after squeeze (before lift)
        has_contact_squeeze, contact_counts_squeeze = self._get_finger_contacts()
        logger.info(f"After squeeze - contacts: {contact_counts_squeeze}")
        logger.info(f"After squeeze - object heights: {self._get_object_heights()}")

        # Record positions before lift
        hand_height_before_lift = self._get_hand_height()
        object_height_before_lift = self._get_object_heights()
        logger.info(f"Hand height before lift: {hand_height_before_lift}")
        logger.info(f"Object height before lift: {object_height_before_lift}")

        # === PHASE 5: Lift (gravity already ON) ===
        # Track contacts during lift
        contact_during_lift = np.zeros(batch_size, dtype=bool)

        for t in np.linspace(0, 1, cfg.lift_steps):
            current = squeeze_targets.copy()
            for i in range(batch_size):
                for j in self._virtual_dof_idx[:3]:  # Only interpolate position
                    current[i, j] = (1-t) * squeeze_targets[i, j] + t * lift_targets[i, j]
            self._set_dof_targets_all(current)
            self._step()

            # Check contacts during lift (sample every 10 steps)
            if int(t * cfg.lift_steps) % 10 == 0:
                has_contact, _ = self._get_finger_contacts()
                contact_during_lift |= has_contact

        # Final measurements
        final_obj_heights = self._get_object_heights()
        final_hand_heights = self._get_hand_height()
        has_contact_final, contact_counts_final = self._get_finger_contacts()

        # Success criteria:
        # 1. Object lifted above its pre-lift position (on ground)
        # 2. Fingers have contact with object during/after lift
        # 3. Hand actually moved up
        object_lifted = (final_obj_heights - object_height_before_lift) > cfg.success_threshold
        hand_lifted = (final_hand_heights - hand_height_before_lift) > cfg.lift_height * 0.5
        has_contact = contact_during_lift | has_contact_final

        success = object_lifted & has_contact & hand_lifted

        logger.info(f"\n=== VALIDATION RESULTS ===")
        logger.info(f"Object heights (before lift -> final): {object_height_before_lift} -> {final_obj_heights}")
        logger.info(f"Hand heights (before -> after lift): {hand_height_before_lift} -> {final_hand_heights}")
        logger.info(f"Contact counts (final): {contact_counts_final}")
        logger.info(f"Object lifted: {object_lifted}")
        logger.info(f"Hand lifted: {hand_lifted}")
        logger.info(f"Has contact: {has_contact}")
        logger.info(f"SUCCESS: {success}")

        return success[:batch_size], final_obj_heights[:batch_size]

    def run_viewer(self, timeout: float = None):
        """Run viewer loop."""
        if timeout:
            start = time.time()
            while time.time() - start < timeout:
                self._step()
        else:
            try:
                while True:
                    self._step()
            except KeyboardInterrupt:
                pass

    def cleanup(self):
        if self._viewer:
            self._gym.destroy_viewer(self._viewer)
        if self._sim:
            self._gym.destroy_sim(self._sim)


def load_grasps(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load grasps from npz.

    Returns:
        translations: (N, 3) wrist positions in world frame
        rotations: (N, 3, 3) wrist rotation matrices
        joints: (N, 16) joint angles
        points: (N, 3) contact points on object surface (used to locate object)
    """
    data = np.load(path)
    translations = data['translation']
    rotations = data['rotation']
    joints = np.stack([data[f'j{i}'] for i in range(16)], axis=1)
    points = data['point']  # Contact points on object
    return translations, rotations, joints, points


def calculate_spawn_height(mesh_path: Path, rotation_matrix: np.ndarray, margin: float = 0.005) -> float:
    """Calculate spawn height so object rests on ground (z=0).

    Args:
        mesh_path: Path to object mesh file (.obj or similar)
        rotation_matrix: 3x3 rotation matrix to apply to mesh
        margin: Small margin above ground (default 5mm)

    Returns:
        Z height for object origin so bottom of rotated mesh is at z=margin
    """
    import trimesh

    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    vertices = np.array(mesh.vertices)

    # Apply rotation to vertices
    rotated_vertices = (rotation_matrix @ vertices.T).T

    # Find min z (bottom of rotated mesh)
    min_z = rotated_vertices[:, 2].min()

    # Spawn height: object origin should be at height such that bottom is at margin
    # If object origin is at z=h, bottom is at z = h + min_z
    # We want h + min_z = margin, so h = margin - min_z
    spawn_height = margin - min_z

    return spawn_height


def load_object_pose_from_scene(scene_dir: Path, obj_id: int) -> np.ndarray:
    """Load object pose from scene annotations.

    Args:
        scene_dir: Path to scene directory (e.g., data/scenes/scene_0001)
        obj_id: Object ID to find

    Returns:
        4x4 pose matrix (rotation + translation)
    """
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation

    # Use first annotation file (frame 0)
    # ann_file = scene_dir / "kinect/annotations/0000.xml"
    # cam_wrt_table_file = scene_dir / "kinect/cam0_wrt_table.npy"
    # if not ann_file.exists():
    ann_file = scene_dir / "realsense/annotations/0000.xml"
    cam_wrt_table_file = scene_dir / "realsense/cam0_wrt_table.npy"

    print(f"We are loading annotation from: {ann_file}")

    tree = ET.parse(ann_file)
    root = tree.getroot()

    cam_wrt_table = np.load(cam_wrt_table_file)

    for obj in root.findall('obj'):
        if int(obj.find('obj_id').text) == obj_id:
            pos = [float(x) for x in obj.find('pos_in_world').text.split()]
            ori = [float(x) for x in obj.find('ori_in_world').text.split()]

            # Convert wxyz quaternion to rotation matrix
            quat_xyzw = [ori[1], ori[2], ori[3], ori[0]]
            R = Rotation.from_quat(quat_xyzw).as_matrix()

            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = pos
            return pose, cam_wrt_table

    raise ValueError(f"Object {obj_id} not found in scene")


def transform_grasps(grasp_translations: np.ndarray, grasp_rotations: np.ndarray,
                     scene_obj_pose: np.ndarray, our_obj_pose: np.ndarray,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Transform grasps from scene frame to our object frame.

    Grasps are originally in the camera frame, designed for scene_obj_pose.
    We transform them to work with our_obj_pose.

    grasp_our = our_obj_pose @ scene_obj_pose^-1 @ grasp_scene
    """
    # Compute transform: scene world -> our world
    scene_obj_pose_inv = np.linalg.inv(scene_obj_pose)
    transform = our_obj_pose @ scene_obj_pose_inv

    # Transform translations: new_t = R @ t + p
    new_translations = (transform[:3, :3] @ grasp_translations.T).T + transform[:3, 3]

    # Transform rotations: new_R = transform_R @ old_R
    new_rotations = np.einsum('ij,njk->nik', transform[:3, :3], grasp_rotations)

    return new_translations, new_rotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--grasp-idx", type=int, default=None)
    args = parser.parse_args()

    # Object ID to test
    obj_id = 0  # Object ID from the grasp path (000.npz -> object 0)

    # Scene ID
    scene_id = 1  # scene_0001

    # Paths
    robot_urdf = Path("robot_models/urdf/leap_hand_simplified_free.urdf")

    grasp_path = Path(f"data/dex_grasps_new/scene_{scene_id:04d}/leap_hand/{obj_id:03d}.npz")
    object_urdf = Path(f"data/meshdata/{obj_id:03d}/nontextured_simplified.urdf")
    object_mesh = Path(f"data/meshdata/{obj_id:03d}/simplified.obj")

    for p in [robot_urdf, object_urdf, grasp_path, object_mesh]:
        if not p.exists():
            print(f"Missing: {p}")
            return
    print(f"Using grasps from: {grasp_path}")
    
    # Load grasps
    translations, rotations, joint_angles, grasp_points = load_grasps(grasp_path)
    print(f"Loaded {len(translations)} grasps")

    # Load actual object pose from scene annotations
    scene_dir = Path(f"data/scenes/scene_{scene_id:04d}")
    scene_obj_pose, cam_wrt_table = load_object_pose_from_scene(scene_dir, obj_id)
    print(f"Scene object pose:\n{scene_obj_pose}")
    print(f"Camera wrt table:\n{cam_wrt_table}")

    # Calculate proper spawn height based on mesh bounding box
    from scipy.spatial.transform import Rotation
    obj_rotation = scene_obj_pose[:3, :3]
    spawn_z = calculate_spawn_height(object_mesh, obj_rotation, margin=0.005)
    print(f"Calculated spawn height: {spawn_z:.4f}")

    # Our object pose: SAME rotation as scene, spawn at calculated height
    our_obj_pose = cam_wrt_table @ scene_obj_pose
    # our_obj_pose[:3, :3] = obj_rotation  # Same rotation as original scene
    # our_obj_pose[:3, 3] = [0.0, 0.0, spawn_z]  # Spawn so bottom rests on ground
    print(f"Our object pose:\n{our_obj_pose}")


    # Transform ALL grasps from scene frame to our object frame
    # Since rotation is the same, this mainly shifts the position
    transformed_trans, transformed_rot = transform_grasps(
        translations, rotations, scene_obj_pose, our_obj_pose)

    print(f"Grasp translation {transformed_trans}")
    print(f"Grasp rotation {transformed_rot}")
    
    # Filter grasps: only keep those that won't end up below ground
    # Grasps designed for object on table (z~0.45) may be below ground when object is at z~0.1
    # We keep grasps with transformed_z > threshold (accounting for ~0.03m settling margin)
    min_grasp_z = 0.02  # Minimum grasp z to keep (above ground clearance)
    valid_mask = transformed_trans[:, 2] > min_grasp_z
    valid_indices = np.where(valid_mask)[0]
    print(f"Valid grasps (z > {min_grasp_z}): {len(valid_indices)} / {len(transformed_trans)}")

    if len(valid_indices) == 0:
        print("ERROR: No valid grasps found! All grasps would be below ground.")
        print(f"Grasp z range: [{transformed_trans[:, 2].min():.3f}, {transformed_trans[:, 2].max():.3f}]")
        return

    # Sample from valid grasps only
    num_envs = args.num_envs
    if args.grasp_idx is not None:
        indices = np.array([args.grasp_idx])
        num_envs = 1
    else:
        indices = np.random.choice(valid_indices, min(num_envs, len(valid_indices)), replace=False)

    batch_trans = transformed_trans[indices]
    # Add some arbitrary offsets around the axis
    batch_rot = transformed_rot[indices]
    batch_joints = joint_angles[indices]

    print(f"Testing grasp indices: {indices}")
    print(f"Original grasp translations: {translations[indices]}")
    print(f"Transformed grasp translations: {batch_trans}")

    # Object position and quaternion for Isaac Gym
    obj_pos = our_obj_pose[:3, 3]
    # Convert rotation matrix to quaternion (xyzw format for Isaac Gym)
    rot = Rotation.from_matrix(our_obj_pose[:3, :3])
    quat_xyzw = rot.as_quat()  # scipy returns xyzw
    # Place W component first for Isaac Gym (wxyz)
    obj_quat = quat_xyzw
    print(f"Object position: {obj_pos}")
    print(f"Object quaternion (xyzw): {obj_quat}")

    # exit()
    # Setup and run
    validator = GraspValidator(num_envs=num_envs, headless=False)
    validator.setup(robot_urdf, object_urdf, obj_pos, obj_quat)

    success, heights = validator.validate_grasps(batch_trans, batch_rot, batch_joints)

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    for i in range(len(success)):
        status = "SUCCESS" if success[i] else "FAILED"
        print(f"  Grasp {indices[i]}: {status} (height={heights[i]:.4f})")
    print(f"\nSuccess rate: {success.sum()}/{len(success)}")

    if args.timeout:
        validator.run_viewer(args.timeout)
    else:
        validator.run_viewer()

    validator.cleanup()


if __name__ == "__main__":
    main()
