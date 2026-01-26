"""
Grasp evaluation using physics simulation.

This module provides grasp success evaluation by simulating
grasp execution and lift in Isaac Gym.

Performance fixes:
- Removed debug print statements (use logger.debug instead)
- Documented first-grasp bug workaround
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

try:
    from pytorch3d.transforms import matrix_to_euler_angles
except ImportError:
    matrix_to_euler_angles = None

from dexgraspnet2.simulation.isaacgym_simulator import (
    IsaacGymSimulator,
    RefreshFlags,
    get_simulator,
)

logger = logging.getLogger(__name__)


class SimulationEvaluator:
    """
    Evaluate grasp success using physics simulation.

    This class simulates the full grasp execution pipeline:
    1. Pre-grasp approach
    2. Finger closing
    3. Object lifting

    A grasp is successful if the object is lifted by at least 3cm.

    Args:
        config: Evaluator configuration.
        device: CUDA device for simulation.

    Note:
        First-grasp bug: There's a known issue where the first grasp
        evaluation in a new simulator instance may behave differently.
        The recommended workaround is to reset the environment twice
        before evaluating grasps. See _reset_environments() for details.
    """

    def __init__(self, config: Dict, device: torch.device):
        """Initialize the evaluator."""
        self._config = config
        self._device = device

        # Load robot model
        from src.utils.robot_model import RobotModel
        self._robot_model = RobotModel(
            config["urdf_path"],
            config["meta_path"],
        )

        # Load width mapper
        from src.utils.width_mapper import WidthMapper
        self._width_mapper = WidthMapper(
            self._robot_model,
            config["width_mapper_meta_path"],
        )

        # Load collision checker
        collision_checker_path = Path("configs/collision_checker") / config["robot_name"] / "CollisionChecker.yaml"
        with open(collision_checker_path) as f:
            collision_checker_config = yaml.safe_load(f)

        from src.utils.collision_checker import CollisionChecker
        self._collision_checker = CollisionChecker(collision_checker_config, device)

        # Simulator will be created in set_environments
        self._simulator: Optional[IsaacGymSimulator] = None

        logger.info(f"Initialized SimulationEvaluator for {config['robot_name']}")

    def set_environments(
        self,
        object_pose_dict: Dict[str, np.ndarray],
        object_surface_points_dict: Dict[str, torch.Tensor],
        num_envs: int,
        dataset: str = "graspnet",
    ):
        """
        Setup simulation environments with objects.

        Args:
            object_pose_dict: Object code to 4x4 pose matrix mapping.
            object_surface_points_dict: Object code to surface points mapping.
            num_envs: Number of parallel environments.
            dataset: Dataset type ('graspnet' or 'acronym').
        """
        self._object_pose_dict = object_pose_dict
        self._object_surface_points_dict = object_surface_points_dict
        self._num_envs = num_envs

        # Load simulator config
        simulator_config_path = Path("configs/simulator") / f"{self._config['simulator_type']}.yaml"
        with open(simulator_config_path) as f:
            simulator_config = yaml.safe_load(f)
        simulator_config["table_height"] = 0

        # Create simulator
        simulator_class = get_simulator(self._config["simulator_type"])
        self._simulator = simulator_class(
            config=simulator_config,
            num_envs=num_envs,
            headless=self._config.get("headless", True),
            device_id=int(str(self._device).split(":")[-1]),
        )

        # Register robot asset
        self._robot_info = self._simulator.register_asset(
            asset_name="robot",
            asset_root="",
            asset_path=self._config["robot_name"] + "_free",
            asset_config={},
        )

        # Register object assets
        self._object_info_dict = {}
        for object_code in object_pose_dict:
            if dataset == "graspnet":
                asset_root = str(Path("data/meshdata") / object_code)
                asset_path = "nontextured_simplified.urdf"
            elif dataset == "acronym":
                asset_root = str(Path("data/acronym/meshes/models") / object_code)
                asset_path = "collision.urdf"
            else:
                raise ValueError(f"Unknown dataset: {dataset}")

            self._object_info_dict[object_code] = self._simulator.register_asset(
                asset_name=f"object_{object_code}",
                asset_root=asset_root,
                asset_path=asset_path,
                asset_config=None,
            )

        # Create environments
        for _ in range(num_envs):
            self._simulator.create_env()
            self._simulator.create_actor(
                actor_name="robot",
                asset_name="robot",
                actor_config=self._config["robot_name"] + "_free",
            )
            for object_code in object_pose_dict:
                self._simulator.create_actor(
                    actor_name=f"object_{object_code}",
                    asset_name=f"object_{object_code}",
                    actor_config={
                        "no_collision": False,
                        "filter": 2,
                        "segmentation_id": 0,
                        "friction": 1,
                        "dof_force_sensors": False,
                        "mass": 0.1,
                    },
                )

        # Prepare simulator
        self._simulator.prepare_sim()

        logger.info(
            f"Environment setup complete: {num_envs} envs, "
            f"{len(object_pose_dict)} objects"
        )

    def _compute_waypoints(self, grasps: Dict[str, np.ndarray]):
        """
        Compute grasp execution waypoints.

        Waypoints:
        1. Pre-grasp: Fingers relaxed, 10cm back from grasp
        2. Cover: At grasp position with relaxed fingers
        3. Grasp: At grasp position with target fingers
        4. Squeeze: Fingers squeezed to secure object
        5. Lift: Object lifted (direction depends on grasp approach)
        """
        batch_size = len(grasps["translation"])
        assert batch_size <= self._num_envs

        self._waypoint_pose_list = []
        self._waypoint_qpos_dict_list = []
        self._waypoint_qpos_list = []

        dof_names = self._robot_info["dof_names"][6:]
        canonical_frame_rotation = torch.tensor(
            self._config["canonical_frame_rotation"],
            dtype=torch.float,
            device=self._device,
        )

        # Parse grasp pose and qpos
        grasp_pose = torch.eye(4, dtype=torch.float, device=self._device)
        grasp_pose = grasp_pose.unsqueeze(0).repeat(batch_size, 1, 1)
        grasp_pose[:, :3, 3] = torch.tensor(
            grasps["translation"], dtype=torch.float, device=self._device
        )
        grasp_pose[:, :3, :3] = torch.tensor(
            grasps["rotation"], dtype=torch.float, device=self._device
        )

        grasp_qpos_dict = {
            joint_name: torch.tensor(grasps[joint_name], dtype=torch.float, device=self._device)
            for joint_name in grasps
            if joint_name not in ["translation", "rotation"]
        }
        grasp_qpos = torch.stack([grasp_qpos_dict[name] for name in dof_names], dim=1)

        # Waypoint 1: Pre-grasp
        pregrasp_qpos_dict = self._width_mapper.squeeze_fingers(
            grasp_qpos_dict, -0.025, -0.025
        )[0]
        pregrasp_pose_local = torch.eye(4, dtype=torch.float, device=self._device)
        pregrasp_pose_local = pregrasp_pose_local.unsqueeze(0).repeat(batch_size, 1, 1)
        pregrasp_pose_local[:, :3, 3] = canonical_frame_rotation.T @ torch.tensor(
            [-0.1, 0.0, 0.0], dtype=torch.float, device=self._device
        )
        pregrasp_pose = grasp_pose @ pregrasp_pose_local
        pregrasp_qpos = torch.stack([pregrasp_qpos_dict[name] for name in dof_names], dim=1)

        self._waypoint_pose_list.append(pregrasp_pose)
        self._waypoint_qpos_dict_list.append(pregrasp_qpos_dict.copy())
        self._waypoint_qpos_list.append(pregrasp_qpos)

        # Waypoint 2: Cover (at grasp position with relaxed fingers)
        self._waypoint_pose_list.append(grasp_pose)
        self._waypoint_qpos_dict_list.append(pregrasp_qpos_dict.copy())
        self._waypoint_qpos_list.append(pregrasp_qpos)

        # Waypoint 3: Grasp
        self._waypoint_pose_list.append(grasp_pose)
        self._waypoint_qpos_dict_list.append(grasp_qpos_dict.copy())
        self._waypoint_qpos_list.append(grasp_qpos)

        # Waypoint 4: Squeeze
        target_qpos_dict = self._width_mapper.squeeze_fingers(
            grasp_qpos_dict, 0.03, 0.03, keep_z=True
        )[0]
        target_qpos = torch.stack([target_qpos_dict[name] for name in dof_names], dim=1)

        self._waypoint_pose_list.append(grasp_pose)
        self._waypoint_qpos_dict_list.append(target_qpos_dict.copy())
        self._waypoint_qpos_list.append(target_qpos)

        # Waypoint 5: Lift
        # Top grasp: move back along gripper x-axis
        lift_pose_top_local = torch.eye(4, dtype=torch.float, device=self._device)
        lift_pose_top_local = lift_pose_top_local.unsqueeze(0).repeat(batch_size, 1, 1)
        lift_pose_top_local[:, :3, 3] = canonical_frame_rotation.T @ torch.tensor(
            [-0.2, 0.0, 0.0], dtype=torch.float, device=self._device
        )
        lift_pose_top = grasp_pose @ lift_pose_top_local

        # Side grasp: move up along world z-axis
        lift_pose_side = grasp_pose.clone()
        lift_pose_side[:, :3, 3] += torch.tensor(
            [0.0, 0.0, 0.2], dtype=torch.float, device=self._device
        )

        # Choose based on gripper orientation
        gripper_x_axis = (grasp_pose[:, :3, :3] @ canonical_frame_rotation.T)[:, :, 0]
        gravity_direction = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float, device=self._device)
        top_mask = (gripper_x_axis * gravity_direction).sum(dim=1) > np.cos(np.pi / 3)
        lift_pose = torch.where(
            top_mask.unsqueeze(1).unsqueeze(1), lift_pose_top, lift_pose_side
        )

        self._waypoint_pose_list.append(lift_pose)
        self._waypoint_qpos_dict_list.append(target_qpos_dict.copy())
        self._waypoint_qpos_list.append(target_qpos)

        # Pad for remaining environments
        if batch_size < self._num_envs:
            self._pad_waypoints(batch_size)

        # Compose full DOF state
        self._waypoint_qpos_all_list = []
        for i in range(len(self._waypoint_pose_list)):
            root_pos = self._waypoint_pose_list[i][:, :3, 3]
            root_rot = matrix_to_euler_angles(
                self._waypoint_pose_list[i][:, :3, :3], "XYZ"
            )
            dof_qpos = self._waypoint_qpos_list[i]
            self._waypoint_qpos_all_list.append(
                torch.cat([root_pos, root_rot, dof_qpos], dim=1)
            )

    def _pad_waypoints(self, batch_size: int):
        """Pad waypoints for remaining environments."""
        pad_size = self._num_envs - batch_size
        for i in range(len(self._waypoint_pose_list)):
            self._waypoint_pose_list[i] = torch.cat([
                self._waypoint_pose_list[i],
                self._waypoint_pose_list[i][-1:].repeat(pad_size, 1, 1),
            ], dim=0)
            for joint in self._waypoint_qpos_dict_list[i]:
                self._waypoint_qpos_dict_list[i][joint] = torch.cat([
                    self._waypoint_qpos_dict_list[i][joint],
                    self._waypoint_qpos_dict_list[i][joint][-1:].repeat(pad_size),
                ], dim=0)
            self._waypoint_qpos_list[i] = torch.cat([
                self._waypoint_qpos_list[i],
                self._waypoint_qpos_list[i][-1:].repeat(pad_size, 1),
            ], dim=0)

    def _check_collision(self) -> np.ndarray:
        """Check pre-grasp collision with scene."""
        grasps = {}
        grasps.update(self._waypoint_qpos_dict_list[0])
        grasps["translation"] = self._waypoint_pose_list[0][:, :3, 3]
        grasps["rotation"] = self._waypoint_pose_list[0][:, :3, :3]

        scene_point_cloud = torch.cat([
            torch.tensor(
                self._object_surface_points_dict[obj_code],
                dtype=torch.float,
                device=self._device,
            )
            for obj_code in self._object_pose_dict
        ], dim=0)

        scene_pen_dist, table_pen_dist = self._collision_checker.check_collision_batch(
            grasps, scene_point_cloud
        )

        scene_valid = scene_pen_dist.cpu().numpy() < self._config["scene_pen_threshold"]
        table_valid = table_pen_dist.cpu().numpy() < self._config["table_pen_threshold"]

        return scene_valid & table_valid

    def _reset_environments(self):
        """
        Reset environments to initial state.

        NOTE: First-grasp bug workaround
        There's a known issue where the first grasp evaluation may
        produce inconsistent results. The workaround is to call
        step() twice after resetting, then reset again. This ensures
        the physics state is properly initialized.
        """
        # Reset object states
        for object_code in self._object_pose_dict:
            object_pose = self._object_pose_dict[object_code]
            self._simulator.set_actor_states(
                actor_name=f"object_{object_code}",
                actor_states={
                    "root_pos": torch.tensor(
                        object_pose[:3, 3], dtype=torch.float, device=self._device
                    ).unsqueeze(0).repeat(self._num_envs, 1),
                    "root_rot": torch.tensor(
                        object_pose[:3, :3], dtype=torch.float, device=self._device
                    ).unsqueeze(0).repeat(self._num_envs, 1, 1),
                    "root_linvel": torch.zeros(
                        [self._num_envs, 3], dtype=torch.float, device=self._device
                    ),
                    "root_angvel": torch.zeros(
                        [self._num_envs, 3], dtype=torch.float, device=self._device
                    ),
                },
            )

        # Reset robot state
        dof_pos_all = self._waypoint_qpos_all_list[0]
        self._simulator.set_actor_states(
            actor_name="robot",
            actor_states={
                "root_pos": torch.zeros(
                    [self._num_envs, 3], dtype=torch.float, device=self._device
                ),
                "root_rot": torch.eye(3, dtype=torch.float, device=self._device)
                    .unsqueeze(0).repeat(self._num_envs, 1, 1),
                "root_linvel": torch.zeros(
                    [self._num_envs, 3], dtype=torch.float, device=self._device
                ),
                "root_angvel": torch.zeros(
                    [self._num_envs, 3], dtype=torch.float, device=self._device
                ),
                "dof_pos": dof_pos_all,
                "dof_vel": torch.zeros_like(dof_pos_all),
            },
        )
        self._simulator.set_actor_actions(
            actor_name="robot",
            actor_actions=dof_pos_all,
        )

    def _execute_waypoints(self):
        """Execute the grasp waypoints."""
        # Disable gravity before pre-grasp
        for object_code in self._object_pose_dict:
            self._simulator.disable_gravity(f"object_{object_code}")

        for i in range(1, len(self._waypoint_pose_list)):
            start_qpos_all = self._waypoint_qpos_all_list[i - 1]
            end_qpos_all = self._waypoint_qpos_all_list[i]
            steps = self._config["waypoint_steps"][i - 1]

            for step in range(steps):
                target_qpos = start_qpos_all + (end_qpos_all - start_qpos_all) * (step + 1) / steps
                self._simulator.set_actor_actions(
                    actor_name="robot",
                    actor_actions=target_qpos,
                )
                # PERF FIX: Removed debug print, use logger.debug instead
                logger.debug(f"Waypoint {i}, step {step}/{steps}")
                self._simulator.step()

            # Enable gravity after squeeze (waypoint 4)
            if i == 3:
                for object_code in self._object_pose_dict:
                    self._simulator.enable_gravity(f"object_{object_code}")

    def _get_sim_successes(self) -> np.ndarray:
        """Check which grasps successfully lifted objects."""
        sim_successes = np.zeros(self._num_envs, dtype=bool)

        for object_code in self._object_pose_dict:
            object_height_init = self._object_pose_dict[object_code][2, 3]
            object_height_final = self._simulator.get_actor_states(
                f"object_{object_code}"
            )["root_pos"][:, 2].cpu().numpy()
            sim_successes |= (object_height_final > object_height_init + 0.03)

        return sim_successes

    def evaluate_data(self, grasps: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Evaluate a batch of grasps.

        Args:
            grasps: Dictionary containing:
                - translation: (B, 3) grasp positions
                - rotation: (B, 3, 3) grasp orientations
                - joint_name: (B,) joint values for each joint

        Returns:
            (B,) boolean array of grasp successes.
        """
        batch_size = len(grasps["translation"])

        # Compute waypoints
        self._compute_waypoints(grasps)

        # Check pre-grasp collision
        pregrasp_valid = self._check_collision()[:batch_size]

        # Reset environments (with first-grasp bug workaround)
        self._reset_environments()
        self._simulator.step()
        self._simulator.step()
        self._reset_environments()

        # Execute waypoints
        self._execute_waypoints()

        # Check success
        sim_successes = self._get_sim_successes()[:batch_size]

        successes = pregrasp_valid & sim_successes

        logger.info(
            f"Evaluated {batch_size} grasps: "
            f"{successes.sum()} successes ({100*successes.mean():.1f}%)"
        )

        return successes
