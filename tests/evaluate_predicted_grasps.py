#!/usr/bin/env python3
"""
Evaluate predicted grasps in Isaac Gym simulation.

This script:
1. Spawns a single object in Isaac Gym
2. Captures depth image from a simulated camera
3. Converts depth to point cloud
4. Runs model inference to predict grasps
5. Validates grasps by simulation (approach, close, lift)
6. Reports success rates

The key advantage of this approach is that everything happens in the same
coordinate frame - no transforms between camera frame and world frame are needed.
"""

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Set environment variables for Vulkan/GLX rendering (must be before isaacgym import)
os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import numpy as np

# Isaac Gym must be imported before torch
from isaacgym import gymapi, gymtorch

import torch

from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SimConfig:
    """Simulation configuration."""

    # Physics
    dt: float = 1.0 / 60.0
    substeps: int = 2
    gravity: float = -9.81

    # Camera (matching realsense-like setup)
    camera_width: int = 640
    camera_height: int = 480
    camera_fov: float = 69.0  # horizontal FOV in degrees

    # Camera position (looking down at table from above)
    camera_pos: Tuple[float, float, float] = (0.0, -0.3, 0.5)
    camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Object spawn
    object_spawn_height: float = 0.1  # Initial height, will settle

    # Grasp validation
    settle_steps: int = 100
    pregrasp_distance: float = 0.08
    pregrasp_steps: int = 60
    approach_steps: int = 60
    grasp_steps: int = 60
    squeeze_steps: int = 30
    lift_height: float = 0.1
    lift_steps: int = 60
    hold_steps: int = 30
    success_threshold: float = 0.03  # Object must lift at least 3cm


@dataclass
class PredictedGrasp:
    """A predicted grasp from the model."""

    translation: np.ndarray  # (3,) wrist position
    rotation: np.ndarray  # (3, 3) wrist rotation matrix
    joint_angles: np.ndarray  # (16,) finger joint angles
    score: float
    seed_point: np.ndarray  # (3,) point on object surface


# =============================================================================
# Point Cloud Utilities
# =============================================================================


def depth_to_point_cloud(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth: float = 2.0,
) -> np.ndarray:
    """
    Convert depth image to point cloud using STANDARD CAMERA CONVENTION.

    Standard camera convention (matching RealSense/training data):
    - +X points right
    - +Y points down (image coordinates)
    - +Z points INTO the scene (depth is positive)

    Isaac Gym returns negative depth values (OpenGL convention), so we negate them.

    Args:
        depth: Depth image (H, W) in meters (negative values from Isaac Gym).
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.
        max_depth: Maximum valid depth.

    Returns:
        Point cloud (N, 3) in camera frame (standard convention, Z positive).
    """
    height, width = depth.shape

    # Isaac Gym depth is negative, convert to positive (standard convention)
    depth_pos = -depth.astype(np.float32)

    # Filter invalid depths
    valid = (depth_pos > 0) & (depth_pos < max_depth) & np.isfinite(depth_pos)

    # Create pixel coordinate grid
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # Back-project to 3D using standard pinhole camera model
    # This matches the original depth_image_to_point_cloud function
    points_z = depth_pos
    points_x = (u - cx) * points_z / fx
    points_y = (v - cy) * points_z / fy

    # Stack and filter
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[valid]

    return points


def camera_to_world_transform(
    camera_pos: Tuple[float, float, float],
    camera_target: Tuple[float, float, float],
) -> np.ndarray:
    """
    Compute 4x4 transform matrix from camera frame to world frame.

    Isaac Gym camera: X-right, Y-down, Z-forward (optical convention)
    World frame: X-forward, Y-left, Z-up
    """
    pos = np.array(camera_pos)
    target = np.array(camera_target)

    # Camera Z axis points from camera to target
    z_axis = target - pos
    z_axis = z_axis / np.linalg.norm(z_axis)

    # World up is Z
    world_up = np.array([0.0, 0.0, 1.0])

    # Camera X axis is perpendicular to Z and world up
    x_axis = np.cross(z_axis, world_up)
    if np.linalg.norm(x_axis) < 1e-6:
        # Camera looking straight up/down
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = x_axis / np.linalg.norm(x_axis)

    # Camera Y axis completes the frame
    y_axis = np.cross(z_axis, x_axis)

    # Build rotation matrix (camera axes as columns)
    R = np.column_stack([x_axis, y_axis, z_axis])

    # Build 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos

    return T


def matrix_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 4x4 transform matrix to position and euler angles (degrees)."""
    pos = T[:3, 3]
    euler_deg = Rotation.from_matrix(T[:3, :3]).as_euler("XYZ", degrees=True)
    return pos, euler_deg


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 4x4 transform matrix."""
    w, x, y, z = quat_wxyz
    # scipy uses (x,y,z,w) order
    rot = Rotation.from_quat([x, y, z, w])
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    return T


def print_pose(name: str, T: np.ndarray):
    """Print pose matrix in human-readable format."""
    pos, euler_deg = matrix_to_pose(T)
    logger.info(f"{name}:")
    logger.info(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    logger.info(
        f"  Euler XYZ (deg): [{euler_deg[0]:.1f}, {euler_deg[1]:.1f}, {euler_deg[2]:.1f}]"
    )
    logger.info(f"  Matrix:\n{T}")


def load_scene_data(scene_id: str = "scene_0220", view_idx: int = 0):
    """
    Load scene data including camera and all objects.

    Returns:
        Dictionary containing camera transforms and a list of object data.
    """
    import xml.etree.ElementTree as ET

    scene_dir = Path(f"data/scenes/{scene_id}/realsense")

    # Load transforms
    cam0_wrt_table = np.load(scene_dir / "cam0_wrt_table.npy")
    camera_poses = np.load(scene_dir / "camera_poses.npy")
    camera_pose_wrt_cam0 = camera_poses[view_idx]

    # Camera pose in table/world frame
    camera_in_world = cam0_wrt_table @ camera_pose_wrt_cam0

    # Load all objects from annotation
    ann_file = scene_dir / "annotations" / f"{view_idx:04d}.xml"
    tree = ET.parse(ann_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall("obj"):
        obj_id = int(obj.find("obj_id").text)
        pos = np.array([float(x) for x in obj.find("pos_in_world").text.split()])
        ori_wxyz = [float(x) for x in obj.find("ori_in_world").text.split()]

        # Build transform matrix (in camera frame)
        obj_pose_in_camera = quat_wxyz_to_matrix(np.array(ori_wxyz))
        obj_pose_in_camera[:3, 3] = pos

        # Transform to world/table frame
        obj_pose_in_world = cam0_wrt_table @ obj_pose_in_camera

        objects.append(
            {
                "id": obj_id,
                "pose_in_camera": obj_pose_in_camera,
                "pose_in_world": obj_pose_in_world,
            }
        )

    return {
        "cam0_wrt_table": cam0_wrt_table,
        "camera_pose_wrt_cam0": camera_pose_wrt_cam0,
        "camera_in_world": camera_in_world,
        "objects": objects,
    }


def transform_point_cloud(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform point cloud by 4x4 matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T).T + t


# =============================================================================
# Grasp Validator
# =============================================================================


class GraspValidator:
    """Single-object grasp validation in Isaac Gym."""

    VIRTUAL_JOINTS = [
        "x_joint",
        "y_joint",
        "z_joint",
        "x_rotation_joint",
        "y_rotation_joint",
        "z_rotation_joint",
    ]

    def __init__(
        self,
        num_envs: int = 1,
        headless: bool = True,
        device: str = "cuda:0",
    ):
        self.num_envs = num_envs
        self.headless = headless
        self.device = device

        self._gym = gymapi.acquire_gym()
        self._sim = None
        self._viewer = None
        self._envs = []

        self._robot_handles = []
        self._object_handles = []
        self._camera_handles = []

        self._dof_names = []
        self._num_dofs = 0
        self._virtual_dof_idx = []
        self._finger_dof_idx = []
        self._grasp_to_dof_map = {}

    def setup(
        self,
        robot_urdf: Path,
        objects_data: List[dict],
        cfg: SimConfig = SimConfig(),
    ):
        """Initialize simulation with robot and objects.

        Args:
            robot_urdf: Path to robot URDF
            objects_data: List of object data dicts (id, pose_in_world, etc.)
            cfg: Simulation config
        """
        self._cfg = cfg
        self._objects_data = objects_data

        # Create sim
        sim_params = gymapi.SimParams()
        sim_params.dt = cfg.dt
        sim_params.substeps = cfg.substeps
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0, 0, cfg.gravity)
        sim_params.use_gpu_pipeline = True

        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0.0

        # Extract device ID from device string (e.g., "cuda:0" -> 0)
        device_id = int(self.device.split(":")[-1])
        self._sim = self._gym.create_sim(
            device_id, device_id, gymapi.SIM_PHYSX, sim_params
        )

        # Ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self._gym.add_ground(self._sim, plane_params)

        # Viewer
        if not self.headless:
            cam_props = gymapi.CameraProperties()
            cam_props.width = 1280
            cam_props.height = 720
            self._viewer = self._gym.create_viewer(self._sim, cam_props)
            if self._viewer:
                # Look at the first object if available
                if len(objects_data) > 0:
                    first_obj_pos = objects_data[0]["pose_in_world"][:3, 3]
                    look_at = gymapi.Vec3(*first_obj_pos)
                    cam_pos = gymapi.Vec3(
                        look_at.x + 0.5, look_at.y - 0.5, look_at.z + 0.5
                    )
                else:
                    look_at = gymapi.Vec3(0, 0, 0.1)
                    cam_pos = gymapi.Vec3(0.5, -0.5, 0.5)

                self._gym.viewer_camera_look_at(
                    self._viewer,
                    None,
                    cam_pos,
                    look_at,
                )

        # Load assets
        robot_asset = self._load_robot_asset(robot_urdf)
        self._object_assets = self._load_object_assets(objects_data)

        # Create environments
        self._create_envs(robot_asset, cfg)

        # Prepare sim
        self._gym.prepare_sim(self._sim)

        # Setup tensors and indices
        self._setup_tensors()

        logger.info(f"Simulation ready: {self.num_envs} envs, {self._num_dofs} DOFs")

    def _load_robot_asset(self, urdf_path: Path):
        """Load robot URDF."""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.armature = 0.01

        asset = self._gym.load_asset(
            self._sim, str(urdf_path.parent), urdf_path.name, asset_options
        )

        self._dof_names = self._gym.get_asset_dof_names(asset)
        self._num_dofs = len(self._dof_names)

        return asset

    def _load_object_assets(self, objects_data: List[dict]):
        """Load object URDFs for all unique objects in the scene."""
        assets = {}
        unique_ids = set(obj["id"] for obj in objects_data)

        asset_options = gymapi.AssetOptions()
        asset_options.density = 500.0
        asset_options.fix_base_link = False

        for obj_id in unique_ids:
            # Construct path based on ID
            urdf_path = Path(f"data/meshdata/{obj_id:03d}/nontextured_simplified.urdf")
            if not urdf_path.exists():
                logger.warning(f"Object URDF not found: {urdf_path}")
                continue

            assets[obj_id] = self._gym.load_asset(
                self._sim, str(urdf_path.parent), urdf_path.name, asset_options
            )

        return assets

    def _create_envs(self, robot_asset, cfg: SimConfig):
        """Create environments with robot, objects, and camera."""
        spacing = 0.5
        env_lower = gymapi.Vec3(-spacing, -spacing, 0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(self.num_envs):
            env = self._gym.create_env(
                self._sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            self._envs.append(env)

            # Spawn all objects from scene data
            first_obj_spawned = False
            for obj_data in self._objects_data:
                obj_id = obj_data["id"]
                if obj_id not in self._object_assets:
                    continue

                asset = self._object_assets[obj_id]

                # Pose from scene data (already in world/table frame)
                pose_mat = obj_data["pose_in_world"]
                pos = pose_mat[:3, 3]
                rot_mat = pose_mat[:3, :3]
                quat = Rotation.from_matrix(rot_mat).as_quat()  # xyzw

                obj_pose = gymapi.Transform()
                obj_pose.p = gymapi.Vec3(*pos)
                obj_pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])

                obj_handle = self._gym.create_actor(
                    env, asset, obj_pose, f"obj_{obj_id}", i, 0
                )

                # Track the first object handle for settling checks
                if not first_obj_spawned:
                    self._object_handles.append(obj_handle)
                    first_obj_spawned = True

            # Spawn robot at origin (0,0,0) so virtual joints match world coordinates
            # We will move it away using DOF targets/states if needed
            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(0, 0, 0)
            robot_handle = self._gym.create_actor(
                env, robot_asset, robot_pose, "robot", i, 1
            )
            self._robot_handles.append(robot_handle)

            # Set robot DOF properties
            dof_props = self._gym.get_actor_dof_properties(env, robot_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"].fill(1000.0)
            dof_props["damping"].fill(100.0)
            self._gym.set_actor_dof_properties(env, robot_handle, dof_props)

            # Create camera sensor (only in first env for point cloud capture)
            if i == 0:
                cam_props = gymapi.CameraProperties()
                cam_props.width = cfg.camera_width
                cam_props.height = cfg.camera_height
                cam_props.horizontal_fov = cfg.camera_fov
                cam_props.near_plane = 0.01
                cam_props.far_plane = 2.0
                cam_props.enable_tensors = False  # Use CPU for simplicity

                cam_handle = self._gym.create_camera_sensor(env, cam_props)
                cam_pos = gymapi.Vec3(*cfg.camera_pos)
                cam_target = gymapi.Vec3(*cfg.camera_target)
                logger.info(
                    f"Setting camera: pos={cfg.camera_pos}, target={cfg.camera_target}"
                )
                self._gym.set_camera_location(cam_handle, env, cam_pos, cam_target)
                self._camera_handles.append(cam_handle)

    def _setup_tensors(self):
        """Setup GPU tensors and index mappings."""
        self._dof_state_tensor = gymtorch.wrap_tensor(
            self._gym.acquire_dof_state_tensor(self._sim)
        )
        self._actor_root_state_tensor = gymtorch.wrap_tensor(
            self._gym.acquire_actor_root_state_tensor(self._sim)
        )
        self._rigid_body_state_tensor = gymtorch.wrap_tensor(
            self._gym.acquire_rigid_body_state_tensor(self._sim)
        )
        self._contact_force_tensor = gymtorch.wrap_tensor(
            self._gym.acquire_net_contact_force_tensor(self._sim)
        )

        self._dof_targets = torch.zeros(
            self.num_envs * self._num_dofs, dtype=torch.float32, device=self.device
        )

        # Virtual joint indices
        self._virtual_dof_idx = [
            self._dof_names.index(n)
            for n in self.VIRTUAL_JOINTS
            if n in self._dof_names
        ]
        self._finger_dof_idx = [
            i for i, n in enumerate(self._dof_names) if n not in self.VIRTUAL_JOINTS
        ]

        # Grasp joint index to DOF index mapping
        self._grasp_to_dof_map = {}
        for dof_idx, name in enumerate(self._dof_names):
            if name.startswith("j") and name[1:].isdigit():
                grasp_joint_idx = int(name[1:])
                self._grasp_to_dof_map[grasp_joint_idx] = dof_idx

        # Actor indices
        self._robot_indices = torch.tensor(
            [
                self._gym.get_actor_index(e, self._robot_handles[i], gymapi.DOMAIN_SIM)
                for i, e in enumerate(self._envs)
            ],
            dtype=torch.int32,
            device=self.device,
        )
        self._object_indices = torch.tensor(
            [
                self._gym.get_actor_index(e, self._object_handles[i], gymapi.DOMAIN_SIM)
                for i, e in enumerate(self._envs)
            ],
            dtype=torch.int32,
            device=self.device,
        )

        # Setup contact indices
        self._setup_contact_indices()

    def _setup_contact_indices(self):
        """Setup finger body indices for contact checking."""
        robot_body_names = self._gym.get_actor_rigid_body_names(
            self._envs[0], self._robot_handles[0]
        )

        self._finger_body_local_idx = []
        for i, name in enumerate(robot_body_names):
            name_lower = name.lower()
            if any(
                kw in name_lower
                for kw in ["finger", "tip", "pip", "dip", "mcp", "thumb"]
            ):
                self._finger_body_local_idx.append(i)

        if not self._finger_body_local_idx:
            self._finger_body_local_idx = list(range(6, len(robot_body_names)))

        # Global rigid body indices
        self._finger_body_indices = []
        for i, env in enumerate(self._envs):
            robot_rb_start = self._gym.get_actor_rigid_body_index(
                env, self._robot_handles[i], 0, gymapi.DOMAIN_SIM
            )
            self._finger_body_indices.append(
                [robot_rb_start + idx for idx in self._finger_body_local_idx]
            )

    def _step(self):
        """Step simulation."""
        self._gym.simulate(self._sim)
        self._gym.fetch_results(self._sim, True)
        # self._gym.step_graphics(self._sim)  # Removed global call

        if self._viewer:
            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(self._viewer, self._sim, True)
            self._gym.sync_frame_time(self._sim)

    def _refresh_tensors(self):
        """Refresh state tensors."""
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)

    def settle_objects(self):
        """Let objects settle under gravity."""
        logger.info("Letting objects settle...")

        # Move robot to safe position above camera (0, 0, 0.6)
        # Camera is at z=0.5, so 0.6 is above it
        safe_targets = np.zeros((self.num_envs, self._num_dofs))
        for i in range(self.num_envs):
            # Virtual joints are X, Y, Z
            safe_targets[i, self._virtual_dof_idx[0]] = 0.0  # X
            safe_targets[i, self._virtual_dof_idx[1]] = 0.0  # Y
            safe_targets[i, self._virtual_dof_idx[2]] = 0.6  # Z

        self._reset_simulation(safe_targets)

        for _ in range(self._cfg.settle_steps):
            self._step()
        self._refresh_tensors()

    def capture_point_cloud(
        self, known_cam_to_world: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Capture depth image and convert to point cloud.

        Args:
            known_cam_to_world: Optional known camera-to-world transform.
                                If provided, this is used instead of view matrix.

        Returns:
            points_world: Point cloud in world frame (N, 3)
            points_cam: Point cloud in camera frame (N, 3)
            cam_to_world: Camera to world transform (4, 4)
        """
        # Must call step_graphics before rendering camera sensors
        self._gym.step_graphics(self._sim)

        # Render cameras
        self._gym.render_all_camera_sensors(self._sim)

        # Get depth image from first environment
        depth = self._gym.get_camera_image(
            self._sim, self._envs[0], self._camera_handles[0], gymapi.IMAGE_DEPTH
        )

        # Debug depth stats
        valid_depth = depth[np.isfinite(depth) & (depth < 0)]
        if len(valid_depth) > 0:
            logger.info(
                f"Depth stats: min={valid_depth.min():.3f}, max={valid_depth.max():.3f}, shape={depth.shape}"
            )

        # Compute camera intrinsics from FOV
        width = self._cfg.camera_width
        height = self._cfg.camera_height
        fov_rad = np.deg2rad(self._cfg.camera_fov)
        fx = width / (2.0 * np.tan(fov_rad / 2.0))
        fy = fx  # Square pixels
        cx = width / 2.0
        cy = height / 2.0

        # Convert depth to point cloud (camera frame)
        points_cam = depth_to_point_cloud(depth, fx, fy, cx, cy)

        logger.info(
            f"Points in camera frame: min={points_cam.min(axis=0)}, max={points_cam.max(axis=0)}"
        )

        if known_cam_to_world is not None:
            cam_to_world = known_cam_to_world
            logger.info("Using provided known camera-to-world transform")
        else:
            # Get camera transform from Isaac Gym
            # Note: Isaac Gym returns column-major (OpenGL style), need to transpose
            view_matrix = (
                np.array(
                    self._gym.get_camera_view_matrix(
                        self._sim, self._envs[0], self._camera_handles[0]
                    )
                )
                .reshape(4, 4)
                .T
            )

            # View matrix is world_to_cam, we need cam_to_world
            cam_to_world = np.linalg.inv(view_matrix)
            logger.info("Computed camera transform from view matrix")

        logger.info(f"Camera position: {cam_to_world[:3, 3]}")

        points_world = transform_point_cloud(points_cam, cam_to_world)

        logger.info(
            f"Points in world frame: min={points_world.min(axis=0)}, max={points_world.max(axis=0)}"
        )
        logger.info(f"Captured point cloud: {len(points_world)} points")

        return points_world, points_cam, cam_to_world

    def get_object_state(self, env_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Get object position and rotation."""
        self._refresh_tensors()
        obj_idx = self._object_indices[env_idx].item()
        state = self._actor_root_state_tensor[obj_idx]
        pos = state[:3].cpu().numpy()
        quat_xyzw = state[3:7].cpu().numpy()
        rot = Rotation.from_quat(quat_xyzw).as_matrix()
        return pos, rot

    def _rotation_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles for URDF."""
        euler_zyx = Rotation.from_matrix(R).as_euler("ZYX", degrees=False)
        return euler_zyx[::-1].copy()  # rx, ry, rz

    def _compute_dof_targets(
        self, translation: np.ndarray, rotation: np.ndarray, joint_angles: np.ndarray
    ) -> np.ndarray:
        """Compute DOF target array from grasp pose."""
        targets = np.zeros(self._num_dofs)

        # Virtual joints: translation
        for j, idx in enumerate(self._virtual_dof_idx[:3]):
            targets[idx] = translation[j]

        # Virtual joints: rotation
        euler = self._rotation_to_euler(rotation)
        for j, idx in enumerate(self._virtual_dof_idx[3:6]):
            targets[idx] = euler[j]

        # Finger joints
        for grasp_idx, dof_idx in self._grasp_to_dof_map.items():
            if grasp_idx < len(joint_angles):
                targets[dof_idx] = joint_angles[grasp_idx]

        return targets

    def _set_dof_targets_all(self, targets_per_env: np.ndarray):
        """Set DOF targets for all environments."""
        targets_flat = targets_per_env.reshape(-1).astype(np.float32)
        self._dof_targets = torch.from_numpy(targets_flat).to(self.device)
        self._gym.set_dof_position_target_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_targets)
        )

    def _get_object_heights(self) -> np.ndarray:
        """Get object Z positions."""
        self._refresh_tensors()
        heights = []
        for i in range(self.num_envs):
            obj_idx = self._object_indices[i].item()
            heights.append(self._actor_root_state_tensor[obj_idx, 2].item())
        return np.array(heights)

    def _get_finger_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        """Check finger contacts with objects."""
        self._refresh_tensors()
        forces = self._contact_force_tensor.cpu().numpy()

        has_contact = np.zeros(self.num_envs, dtype=bool)
        contact_counts = np.zeros(self.num_envs, dtype=int)

        for i in range(self.num_envs):
            for body_idx in self._finger_body_indices[i]:
                force_mag = np.linalg.norm(forces[body_idx])
                if force_mag > 0.1:
                    contact_counts[i] += 1
            has_contact[i] = contact_counts[i] > 0

        return has_contact, contact_counts

    def _reset_simulation(self, targets: np.ndarray):
        """Reset simulation state with robot at targets (teleport)."""
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)

        # Set robot DOF positions AND targets (vectorized)
        targets_tensor = torch.from_numpy(targets.astype(np.float32)).to(self.device)
        targets_flat = targets_tensor.reshape(-1)

        # DOF state tensor is (num_envs * num_dofs, 2) - col 0 is pos, col 1 is vel
        self._dof_state_tensor[:, 0] = targets_flat
        self._dof_state_tensor[:, 1] = 0.0
        self._dof_targets[:] = targets_flat

        self._gym.set_dof_state_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_state_tensor)
        )
        self._gym.set_dof_position_target_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_targets)
        )

    def cleanup(self):
        """Cleanup simulation resources."""
        if self._viewer:
            self._gym.destroy_viewer(self._viewer)
        if self._sim:
            self._gym.destroy_sim(self._sim)


# =============================================================================
# Model Inference
# =============================================================================


def run_inference(
    point_cloud: np.ndarray,
    ckpt_path: str,
    num_grasps: int = 10,
    device: str = "cuda:0",
) -> List[PredictedGrasp]:
    """
    Run grasp prediction model on point cloud.

    Args:
        point_cloud: Point cloud (N, 3) in world frame
        ckpt_path: Path to model checkpoint
        num_grasps: Number of grasps to generate
        device: Torch device

    Returns:
        List of predicted grasps sorted by score
    """
    from src.utils.config import ckpt_to_config
    from src.network.model import get_model
    from src.utils.dataset import get_sparse_tensor

    # Load model
    config = ckpt_to_config(ckpt_path)
    model = get_model(config.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    model.config.voxel_size = config.data.voxel_size

    # Prepare input
    pc_tensor = torch.from_numpy(point_cloud).float()

    with torch.no_grad():
        data = get_sparse_tensor(pc_tensor[None], config.data.voxel_size)
        # Create dummy segmentation (all ones - single object)
        data["seg"] = torch.ones(1, len(point_cloud), dtype=torch.long)
        data = {k: v.to(device) for k, v in data.items()}

        rot, trans, joints, score, obj_indices, seed_points = (
            t.cpu()
            for t in model.sample(
                data,
                num_grasps,
                edge=None,
                graspness_scale=5,
                allow_fail=True,
                with_point=True,
                cate=True,
            )
        )

    # Convert to list of PredictedGrasp objects
    grasps = []
    for i in range(num_grasps):
        grasps.append(
            PredictedGrasp(
                translation=trans[0, i].numpy(),
                rotation=rot[0, i].numpy(),
                joint_angles=joints[0, i].numpy(),
                score=score[0, i].item(),
                seed_point=seed_points[i].numpy(),
            )
        )

    # Sort by score (descending)
    grasps.sort(key=lambda g: g.score, reverse=True)

    logger.info(
        f"Generated {len(grasps)} grasps, scores: [{grasps[-1].score:.1f}, {grasps[0].score:.1f}]"
    )

    return grasps


# =============================================================================
# Visualization
# =============================================================================


def create_visualization(
    point_cloud: np.ndarray,
    grasps: List[PredictedGrasp],
    results: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    point_cloud_cam: Optional[np.ndarray] = None,
    grasps_cam: Optional[List[PredictedGrasp]] = None,
):
    """
    Create HTML visualization of point cloud and grasps.

    Args:
        point_cloud: Point cloud (N, 3) in world frame
        grasps: List of grasps to visualize (world frame)
        results: Optional success array for coloring
        output_path: Path to save HTML file
        point_cloud_cam: Optional point cloud in camera frame for separate visualization
        grasps_cam: Optional list of grasps in camera frame for camera-frame visualization
    """
    from src.utils.vis_plotly import Vis
    import plotly.express as px
    import plotly.graph_objects as go

    vis = Vis(
        robot_name="leap_hand",
        urdf_path="robot_models/urdf/leap_hand_simplified.urdf",
        meta_path="robot_models/meta/leap_hand/meta.yaml",
    )

    # Point cloud
    pc_plotly = vis.pc_plotly(
        torch.from_numpy(point_cloud).float(), size=1, color="blue"
    )

    # Grasps
    pose_plotly = []
    seed_plotly = []

    for i, g in enumerate(grasps):
        # Color based on success if available
        if results is not None:
            color = "green" if results[i] else "red"
        else:
            color = random.choice(px.colors.sequential.Plasma)

        pose_plotly += vis.robot_plotly(
            torch.from_numpy(g.translation)[None].float(),
            torch.from_numpy(g.rotation)[None].float(),
            torch.from_numpy(g.joint_angles)[None].float(),
            opacity=0.8,
            color=color,
        )

        seed_plotly += vis.pc_plotly(
            torch.from_numpy(g.seed_point)[None].float(),
            size=5,
            color="red",
        )

    # Combine
    all_plotly = pc_plotly + pose_plotly + seed_plotly

    # Create figure
    fig = go.Figure(data=all_plotly, layout=go.Layout(scene=dict(aspectmode="data")))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        logger.info(f"Saved visualization (world frame) to {output_path}")

        # Also save camera-frame point cloud visualization with grasps
        # Use camera-frame grasps if provided, otherwise use world-frame grasps
        if point_cloud_cam is not None:
            pc_cam_plotly = vis.pc_plotly(
                torch.from_numpy(point_cloud_cam).float(), size=1, color="blue"
            )

            # Use camera-frame grasps if provided
            cam_grasps_to_vis = grasps_cam if grasps_cam is not None else grasps
            pose_cam_plotly = []
            seed_cam_plotly = []
            for i, g in enumerate(cam_grasps_to_vis):
                if results is not None:
                    color = "green" if results[i] else "red"
                else:
                    color = "orange"

                pose_cam_plotly += vis.robot_plotly(
                    torch.from_numpy(g.translation)[None].float(),
                    torch.from_numpy(g.rotation)[None].float(),
                    torch.from_numpy(g.joint_angles)[None].float(),
                    opacity=0.8,
                    color=color,
                )
                seed_cam_plotly += vis.pc_plotly(
                    torch.from_numpy(g.seed_point)[None].float(),
                    size=8,
                    color="red",
                )

            fig_cam = go.Figure(
                data=pc_cam_plotly + pose_cam_plotly + seed_cam_plotly,
                layout=go.Layout(scene=dict(aspectmode="data")),
            )
            cam_path = output_path.replace(".html", "_camera_frame.html")
            fig_cam.write_html(cam_path)
            logger.info(f"Saved visualization (camera frame) to {cam_path}")
    else:
        fig.show()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted grasps in Isaac Gym"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Model checkpoint path"
    )
    parser.add_argument(
        "--scene_id", type=str, default="scene_0220", help="Scene ID to test"
    )
    parser.add_argument(
        "--num_grasps", type=int, default=10, help="Number of grasps to test"
    )
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument(
        "--timeout", type=float, default=None, help="Viewer timeout in seconds"
    )
    parser.add_argument(
        "--output_vis", type=str, default=None, help="Path to save HTML visualization"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Detect object ID from scene annotations
    import xml.etree.ElementTree as ET

    scene_dir = Path(f"data/scenes/{args.scene_id}/realsense")
    ann_file = scene_dir / "annotations" / "0000.xml"

    if not ann_file.exists():
        logger.error(f"Annotation file not found: {ann_file}")
        return

    # Paths
    robot_urdf = Path("robot_models/urdf/leap_hand_simplified_free.urdf")
    if not robot_urdf.exists():
        logger.error(f"Missing: {robot_urdf}")
        return

    logger.info(f"Testing scene {args.scene_id}")

    # ==========================================================================
    # LOAD SCENE DATA
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info(f"LOADING SCENE: {args.scene_id}")
    logger.info("=" * 60)

    # Load scene data
    scene_data = None
    try:
        scene_data = load_scene_data(args.scene_id, view_idx=0)

        cam_world = scene_data["camera_in_world"]
        objects_data = scene_data["objects"]

        logger.info(f"Loaded {len(objects_data)} objects from scene")
        print_pose("Camera in World", cam_world)

    except Exception as e:
        logger.warning(f"Could not load scene data: {e}")
        import traceback

        traceback.print_exc()
        return

    # ==========================================================================
    # ISAAC GYM SETUP
    # ==========================================================================
    logger.info("\n--- ISAAC GYM SETUP ---")

    cfg = SimConfig()

    # Configure camera from scene data
    cam_pos_world = cam_world[:3, 3]
    # For target, we can look at the first object
    if len(objects_data) > 0:
        cam_target_world = objects_data[0]["pose_in_world"][:3, 3]
    else:
        cam_target_world = np.zeros(3)

    cfg.camera_pos = tuple(cam_pos_world)
    cfg.camera_target = tuple(cam_target_world)

    logger.info(f"Camera position (from scene): {cfg.camera_pos}")

    # Create validator
    validator = GraspValidator(num_envs=args.num_grasps, headless=args.headless)

    # Setup with all objects
    validator.setup(
        robot_urdf,
        objects_data,
        cfg,
    )

    # Let objects settle
    validator.settle_objects()

    # Get actual pose after settling (just for first object as reference)
    if len(objects_data) > 0:
        obj_pos, obj_rot = validator.get_object_state(0)
        isaac_obj_settled = np.eye(4)
        isaac_obj_settled[:3, :3] = obj_rot
        isaac_obj_settled[:3, 3] = obj_pos
        print_pose(f"Isaac Gym Object 0 Pose (after settling)", isaac_obj_settled)

    logger.info("=" * 60 + "\n")

    # Capture point cloud (returns both camera-frame and world-frame)
    # We use the reference camera pose because Isaac Gym's view matrix is sometimes problematic
    # and we want to ensure alignment with the scene data
    point_cloud_world, point_cloud_cam, cam_to_world = validator.capture_point_cloud(
        known_cam_to_world=scene_data["camera_in_world"] if scene_data else None
    )
    # TESTING FORUM FIX: Not using known transform to force computation from view matrix + env origin correction
    # point_cloud_world, point_cloud_cam, cam_to_world = validator.capture_point_cloud(
    #     known_cam_to_world=None
    # )

    # Log point cloud extents BEFORE filtering
    logger.info(f"Point cloud (world) extents BEFORE filter:")
    logger.info(
        f"  X: [{point_cloud_world[:, 0].min():.4f}, {point_cloud_world[:, 0].max():.4f}]"
    )
    logger.info(
        f"  Y: [{point_cloud_world[:, 1].min():.4f}, {point_cloud_world[:, 1].max():.4f}]"
    )
    logger.info(
        f"  Z: [{point_cloud_world[:, 2].min():.4f}, {point_cloud_world[:, 2].max():.4f}]"
    )

    logger.info(f"Point cloud (camera) extents:")
    logger.info(
        f"  X: [{point_cloud_cam[:, 0].min():.4f}, {point_cloud_cam[:, 0].max():.4f}]"
    )
    logger.info(
        f"  Y: [{point_cloud_cam[:, 1].min():.4f}, {point_cloud_cam[:, 1].max():.4f}]"
    )
    logger.info(
        f"  Z: [{point_cloud_cam[:, 2].min():.4f}, {point_cloud_cam[:, 2].max():.4f}]"
    )

    # Keep full point cloud (including ground) for visualization
    point_cloud_full_world = point_cloud_world
    point_cloud_full_cam = point_cloud_cam

    # For inference, we use CAMERA FRAME point cloud (model was trained on camera-frame data)
    # No filtering needed for now - let's first verify the point cloud looks correct
    point_cloud_for_inference = point_cloud_cam
    logger.info(
        f"Point cloud for inference: {len(point_cloud_for_inference)} points (camera frame)"
    )

    # Log camera-frame extents
    logger.info(f"Point cloud (camera) extents for inference:")
    logger.info(
        f"  X: [{point_cloud_for_inference[:, 0].min():.4f}, {point_cloud_for_inference[:, 0].max():.4f}]"
    )
    logger.info(
        f"  Y: [{point_cloud_for_inference[:, 1].min():.4f}, {point_cloud_for_inference[:, 1].max():.4f}]"
    )
    logger.info(
        f"  Z: [{point_cloud_for_inference[:, 2].min():.4f}, {point_cloud_for_inference[:, 2].max():.4f}]"
    )

    # Run inference on camera-frame point cloud
    grasps_cam = run_inference(
        point_cloud_for_inference, args.ckpt_path, args.num_grasps
    )

    # Visualization (camera frame only as requested)
    if args.output_vis:
        # Create visualization using camera-frame point cloud and grasps
        # We pass grasps_cam as the main 'grasps' argument for simplicity in visualization function
        # But wait, create_visualization expects 'point_cloud' and 'grasps' in world frame usually.
        # However, if we only care about camera frame, we can trick it or use the camera frame args.

        from src.utils.vis_plotly import Vis
        import plotly.express as px
        import plotly.graph_objects as go

        vis = Vis(
            robot_name="leap_hand",
            urdf_path="robot_models/urdf/leap_hand_simplified.urdf",
            meta_path="robot_models/meta/leap_hand/meta.yaml",
        )

        # Camera frame visualization
        pc_cam_plotly = vis.pc_plotly(
            torch.from_numpy(point_cloud_cam).float(), size=1, color="blue"
        )

        pose_cam_plotly = []
        seed_cam_plotly = []
        for i, g in enumerate(grasps_cam):
            color = random.choice(px.colors.sequential.Plasma)

            pose_cam_plotly += vis.robot_plotly(
                torch.from_numpy(g.translation)[None].float(),
                torch.from_numpy(g.rotation)[None].float(),
                torch.from_numpy(g.joint_angles)[None].float(),
                opacity=0.8,
                color=color,
            )
            seed_cam_plotly += vis.pc_plotly(
                torch.from_numpy(g.seed_point)[None].float(),
                size=8,
                color="red",
            )

        fig_cam = go.Figure(
            data=pc_cam_plotly + pose_cam_plotly + seed_cam_plotly,
            layout=go.Layout(scene=dict(aspectmode="data")),
        )
        cam_path = args.output_vis.replace(".html", "_camera_frame.html")
        fig_cam.write_html(cam_path)
        logger.info(f"Saved visualization (camera frame) to {cam_path}")

    # Run viewer if not headless
    if not args.headless and validator._viewer:
        import time

        logger.info("Running viewer... Press ESC or close window to exit.")
        timeout = args.timeout if args.timeout else 300  # Default 5 min timeout
        start = time.time()
        while time.time() - start < timeout:
            validator._step()
            validator._gym.draw_viewer(validator._viewer, validator._sim, True)
            validator._gym.sync_frame_time(validator._sim)
            if validator._gym.query_viewer_has_closed(validator._viewer):
                break

    validator.cleanup()


if __name__ == "__main__":
    main()
