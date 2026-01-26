"""
Isaac Gym simulator for grasp evaluation.

This module provides a refactored Isaac Gym simulator with
performance optimizations:
- Selective state refresh using RefreshFlags
- Optional state caching with proper invalidation
- Async render option for non-blocking viewer
"""

import logging
from enum import Flag, auto
from typing import Dict, List, Optional, Union

import numpy as np
import torch

try:
    from isaacgym import gymapi, gymtorch
    HAS_ISAACGYM = True
except ImportError:
    HAS_ISAACGYM = False
    gymapi = None
    gymtorch = None

try:
    from pytorch3d.transforms import matrix_to_quaternion
except ImportError:
    matrix_to_quaternion = None

logger = logging.getLogger(__name__)


class RefreshFlags(Flag):
    """
    Flags for selective state tensor refresh.

    Use these flags to only refresh the state tensors you need,
    reducing GPU synchronization overhead.

    Example:
        >>> sim._refresh(RefreshFlags.ROOT_STATE | RefreshFlags.DOF_STATE)
    """

    NONE = 0
    ROOT_STATE = auto()
    DOF_STATE = auto()
    RIGID_BODY_STATE = auto()
    FORCE_SENSOR = auto()
    DOF_FORCE = auto()
    ALL = ROOT_STATE | DOF_STATE | RIGID_BODY_STATE | FORCE_SENSOR | DOF_FORCE


class SimulatorBase:
    """
    Base class for physics simulators.

    Defines the interface for grasp evaluation simulators.
    """

    def __init__(
        self,
        config: Dict,
        num_envs: int,
        device_id: int,
        headless: bool,
    ):
        """
        Initialize the simulator.

        Args:
            config: Simulator configuration.
            num_envs: Number of parallel environments.
            device_id: CUDA device ID.
            headless: Whether to run without visualization.
        """
        self._config = config
        self._num_envs = num_envs
        self._device_id = device_id
        self._headless = headless


class IsaacGymSimulator(SimulatorBase):
    """
    Isaac Gym-based physics simulator for grasp evaluation.

    This class provides a clean interface for:
    - Creating environments with robots and objects
    - Setting and getting actor states
    - Running physics simulation
    - Evaluating grasp success

    Performance optimizations:
    - Selective refresh: Only refresh needed state tensors
    - State caching: Cache actor states with proper invalidation
    - Async rendering: Non-blocking viewer updates

    Args:
        config: Simulator configuration dictionary.
        num_envs: Number of parallel environments.
        device_id: CUDA device ID.
        headless: Whether to run without visualization.
    """

    # Drive mode mapping
    DRIVE_MODES = {
        "pos": gymapi.DOF_MODE_POS if HAS_ISAACGYM else 0,
        "none": gymapi.DOF_MODE_NONE if HAS_ISAACGYM else 0,
    }

    def __init__(
        self,
        config: Dict,
        num_envs: int,
        device_id: int = 0,
        headless: bool = True,
    ):
        """Initialize the Isaac Gym simulator."""
        if not HAS_ISAACGYM:
            raise ImportError(
                "Isaac Gym is required for simulation. "
                "Please install from: https://developer.nvidia.com/isaac-gym"
            )

        super().__init__(config, num_envs, device_id, headless)

        self._device = torch.device(f"cuda:{device_id}")

        # Initialize simulation
        self._init_sim()

        # Asset and actor tracking
        self._asset_handle: Dict[str, any] = {}
        self._env_list: List[any] = []
        self._actor_indices: Dict[str, List[int]] = {}

        # Index tracking
        self._total_dofs = 0
        self._total_rigid_bodies = 0
        self._total_force_sensors = 0
        self._total_dof_force_sensors = 0

        self._actor_dof_indices: Dict[str, List[List[int]]] = {}
        self._actor_rigid_body_indices: Dict[str, List[List[int]]] = {}
        self._actor_force_sensor_indices: Dict[str, List[List[int]]] = {}
        self._actor_dof_force_sensor_indices: Dict[str, List[List[int]]] = {}

        # Rigid body mass tracking
        self._rigid_body_mass: List[float] = []

        # State caching
        self._state_cache_enabled = True
        self._actor_state_cache: Dict[str, Dict] = {}
        self._cache_valid = False

        # Gravity control
        self._enable_gravity: Dict[str, bool] = {}

        # Async render
        self._async_render = False

        logger.info(
            f"Initialized IsaacGymSimulator: num_envs={num_envs}, "
            f"device={self._device}, headless={headless}"
        )

    def _init_sim(self):
        """Initialize gym, configure sim, and add ground plane."""
        # Initialize gym
        self._gym = gymapi.acquire_gym()

        # Configure simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / self._config.get("env_hz", 120)
        sim_params.substeps = 2
        sim_params.num_client_threads = 0
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        # PhysX settings
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 1000.0
        sim_params.physx.default_buffer_size_multiplier = 5.0
        sim_params.use_gpu_pipeline = True
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = True
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

        # Create simulation
        if self._headless:
            self._sim = self._gym.create_sim(
                self._device_id, self._device_id, gymapi.SIM_PHYSX, sim_params
            )
        else:
            self._sim = self._gym.create_sim(
                self._device_id, 0, gymapi.SIM_PHYSX, sim_params
            )

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self._gym.add_ground(self._sim, plane_params)

    def register_asset(
        self,
        asset_name: str,
        asset_root: str,
        asset_path: str,
        asset_config: Optional[Dict],
    ) -> Dict:
        """
        Register an asset to the simulator.

        Args:
            asset_name: Unique name for the asset.
            asset_root: Root directory for asset files.
            asset_path: Path to asset file relative to root.
            asset_config: Asset configuration options.

        Returns:
            Asset info dictionary with num_dofs, dof_names, limits, etc.
        """
        if asset_name in self._asset_handle:
            raise ValueError(f"Asset '{asset_name}' already registered")

        # Load config from defaults if needed
        if asset_root == "":
            asset_cfg = self._config["asset"][asset_path]
            asset_root = asset_cfg["asset_root"]
            asset_config = asset_cfg.get("asset_config", {})
            asset_path = asset_cfg["asset_path"]

        if asset_config is None:
            if asset_name.startswith("object"):
                asset_config = self._config["asset"]["object"]["asset_config"]
            else:
                asset_config = self._config["asset"][asset_name]["asset_config"]

        # Parse asset options
        asset_options = gymapi.AssetOptions()
        for key, value in asset_config.get("asset_options", {}).items():
            setattr(asset_options, key, eval(str(value)))

        # Enable VHACD for objects
        if asset_name.startswith("object"):
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 100000

        # Load asset
        asset = self._gym.load_asset(self._sim, asset_root, asset_path, asset_options)

        # Create force sensors
        body_handles = []
        for body_name in asset_config.get("force_sensors", []):
            body_handle = self._gym.find_asset_rigid_body_index(asset, body_name)
            body_handles.append(body_handle)
        body_handles.sort()

        for body_handle in body_handles:
            sensor_pose = gymapi.Transform()
            self._gym.create_asset_force_sensor(asset, body_handle, sensor_pose)

        # Find body indices
        body_indices = []
        for body_name in asset_config.get("body_names", []):
            body_indices.append(self._gym.find_asset_rigid_body_index(asset, body_name))
        body_indices.sort()
        body_indices = torch.tensor(body_indices, dtype=torch.long, device=self._device)

        # Store handle
        self._asset_handle[asset_name] = asset

        # Return info
        dof_props = self._gym.get_asset_dof_properties(asset)
        return {
            "num_dofs": len(dof_props),
            "dof_names": self._gym.get_asset_dof_names(asset),
            "dof_lower": torch.tensor(dof_props["lower"], dtype=torch.float, device=self._device),
            "dof_upper": torch.tensor(dof_props["upper"], dtype=torch.float, device=self._device),
            "body_indices": body_indices,
        }

    def create_env(self):
        """Create a new environment."""
        spacing = self._config.get("env_spacing", 1.0)
        num_per_row = int(self._num_envs ** 0.5)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        env = self._gym.create_env(self._sim, lower, upper, num_per_row)
        self._env_list.append(env)

    def create_actor(
        self,
        actor_name: str,
        asset_name: str,
        actor_config: Union[Dict, str],
    ):
        """
        Create an actor in the most recently created environment.

        Args:
            actor_name: Name for the actor (same across all envs).
            asset_name: Name of registered asset to use.
            actor_config: Actor configuration or config key.
        """
        # Load config from defaults if string
        if isinstance(actor_config, str):
            actor_config = self._config["actor"][actor_config]["actor_config"]

        env = self._env_list[-1]
        asset = self._asset_handle[asset_name]

        # Create actor
        collision_group = len(self._env_list) - 1
        if actor_config.get("no_collision", False):
            collision_group += self._num_envs

        actor = self._gym.create_actor(
            env, asset, gymapi.Transform(),
            actor_name,
            collision_group,
            actor_config.get("filter", 0),
            actor_config.get("segmentation_id", 0),
        )

        # Set friction
        shape_props = self._gym.get_actor_rigid_shape_properties(env, actor)
        for prop in shape_props:
            prop.friction = actor_config.get("friction", 1.0)
        self._gym.set_actor_rigid_shape_properties(env, actor, shape_props)

        # Set DOF properties
        if "dof_props" in actor_config:
            self._set_actor_dof_props(env, actor, actor_config["dof_props"])

        # Set mass
        if "mass" in actor_config:
            self._set_actor_mass(env, actor, actor_config["mass"])

        # Enable DOF force sensors
        if actor_config.get("dof_force_sensors", False):
            self._gym.enable_actor_dof_force_sensors(env, actor)

        # Track indices
        self._track_actor_indices(env, actor, actor_name, actor_config)

    def _set_actor_dof_props(self, env, actor, dof_props_config: Dict):
        """Set DOF properties for an actor."""
        dof_props = self._gym.get_actor_dof_properties(env, actor)
        dof_names = self._gym.get_actor_dof_names(env, actor)

        # Expand config if not per-DOF
        if dof_names[0] not in dof_props_config:
            dof_props_config = {name: dof_props_config for name in dof_names}

        for i, dof_name in enumerate(dof_names):
            cfg = dof_props_config.get(dof_name, {})
            if "driveMode" in cfg:
                dof_props["driveMode"][i] = self.DRIVE_MODES[cfg["driveMode"]]
            if "stiffness" in cfg:
                dof_props["stiffness"][i] = cfg["stiffness"]
            if "damping" in cfg:
                dof_props["damping"][i] = cfg["damping"]
            if "effort" in cfg:
                dof_props["effort"][i] = cfg["effort"]
            if "velocity" in cfg:
                dof_props["velocity"][i] = cfg["velocity"]

        self._gym.set_actor_dof_properties(env, actor, dof_props)

    def _set_actor_mass(self, env, actor, target_mass: float):
        """Set total mass for an actor."""
        rigid_props = self._gym.get_actor_rigid_body_properties(env, actor)
        current_mass = sum(p.mass for p in rigid_props)
        scale = target_mass / current_mass

        for prop in rigid_props:
            prop.mass *= scale
            prop.invMass = 1.0 / prop.mass
            prop.inertia.x *= scale
            prop.inertia.y *= scale
            prop.inertia.z *= scale

        self._gym.set_actor_rigid_body_properties(env, actor, rigid_props)

    def _track_actor_indices(self, env, actor, actor_name: str, actor_config: Dict):
        """Track indices for an actor."""
        if actor_name not in self._actor_indices:
            self._actor_indices[actor_name] = []
            self._actor_dof_indices[actor_name] = []
            self._actor_rigid_body_indices[actor_name] = []
            self._actor_force_sensor_indices[actor_name] = []
            self._actor_dof_force_sensor_indices[actor_name] = []

        # Actor index
        actor_index = self._gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)
        self._actor_indices[actor_name].append(actor_index)

        # DOF indices
        dof_count = self._gym.get_actor_dof_count(env, actor)
        self._actor_dof_indices[actor_name].append(
            list(range(self._total_dofs, self._total_dofs + dof_count))
        )
        self._total_dofs += dof_count

        # Rigid body indices
        rb_count = self._gym.get_actor_rigid_body_count(env, actor)
        self._actor_rigid_body_indices[actor_name].append(
            list(range(self._total_rigid_bodies, self._total_rigid_bodies + rb_count))
        )
        self._total_rigid_bodies += rb_count

        # Force sensor indices
        fs_count = self._gym.get_actor_force_sensor_count(env, actor)
        self._actor_force_sensor_indices[actor_name].append(
            list(range(self._total_force_sensors, self._total_force_sensors + fs_count))
        )
        self._total_force_sensors += fs_count

        # DOF force sensor indices
        dof_fs = dof_count if actor_config.get("dof_force_sensors", False) else 0
        self._actor_dof_force_sensor_indices[actor_name].append(
            list(range(self._total_dof_force_sensors, self._total_dof_force_sensors + dof_fs))
        )
        self._total_dof_force_sensors += dof_fs

        # Track rigid body masses
        rb_props = self._gym.get_actor_rigid_body_properties(env, actor)
        self._rigid_body_mass.extend(p.mass for p in rb_props)

    def prepare_sim(self):
        """Prepare simulator after creating all environments."""
        # Tensorfy indices
        self._tensorfy_indices()

        # Prepare sim
        self._gym.prepare_sim(self._sim)

        # Acquire state tensors
        self._acquire_state_tensors()

        # Initialize gravity control
        for actor_name in self._actor_indices:
            self._enable_gravity[actor_name] = True
        self._forces = torch.zeros(
            [self._total_rigid_bodies, 3],
            dtype=torch.float,
            device=self._device,
        )

        # Create viewer
        if not self._headless:
            self._create_viewer()
        else:
            self._viewer = None

    def _tensorfy_indices(self):
        """Convert index lists to tensors."""
        for actor_name in self._actor_indices:
            self._actor_indices[actor_name] = torch.tensor(
                self._actor_indices[actor_name], dtype=torch.long, device=self._device
            )
            self._actor_dof_indices[actor_name] = torch.tensor(
                self._actor_dof_indices[actor_name], dtype=torch.long, device=self._device
            )
            self._actor_rigid_body_indices[actor_name] = torch.tensor(
                self._actor_rigid_body_indices[actor_name], dtype=torch.long, device=self._device
            )
            self._actor_force_sensor_indices[actor_name] = torch.tensor(
                self._actor_force_sensor_indices[actor_name], dtype=torch.long, device=self._device
            )
            self._actor_dof_force_sensor_indices[actor_name] = torch.tensor(
                self._actor_dof_force_sensor_indices[actor_name], dtype=torch.long, device=self._device
            )

        self._rigid_body_mass = torch.tensor(
            self._rigid_body_mass, dtype=torch.float, device=self._device
        )

    def _acquire_state_tensors(self):
        """Acquire GPU state tensors."""
        self._root_state_tensor = gymtorch.wrap_tensor(
            self._gym.acquire_actor_root_state_tensor(self._sim)
        ).view(-1, 13)

        self._dof_state_tensor = gymtorch.wrap_tensor(
            self._gym.acquire_dof_state_tensor(self._sim)
        ).view(-1, 2)

        self._rigid_body_state_tensor = gymtorch.wrap_tensor(
            self._gym.acquire_rigid_body_state_tensor(self._sim)
        ).view(-1, 13)

        if self._total_force_sensors > 0:
            self._force_sensor_tensor = gymtorch.wrap_tensor(
                self._gym.acquire_force_sensor_tensor(self._sim)
            ).view(-1, 6)
        else:
            self._force_sensor_tensor = torch.zeros(
                [0, 6], dtype=torch.float32, device=self._device
            )

        if self._total_dof_force_sensors > 0:
            self._dof_force_tensor = gymtorch.wrap_tensor(
                self._gym.acquire_dof_force_tensor(self._sim)
            ).view(-1)
        else:
            self._dof_force_tensor = torch.zeros(
                [0], dtype=torch.float32, device=self._device
            )

        # DOF position targets
        self._dof_position_target_tensor = torch.zeros(
            self._total_dofs, dtype=torch.float32, device=self._device
        )

    def _create_viewer(self):
        """Create visualization viewer."""
        self._enable_viewer_sync = True

        self._viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
        if self._viewer is None:
            logger.error("Failed to create viewer")
            return

        # Keyboard shortcuts
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_ESCAPE, "QUIT"
        )
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_V, "toggle_viewer_sync"
        )

        # Camera position
        cam_pos = gymapi.Vec3(*self._config.get("camera_pos", [1, 1, 1]))
        cam_target = gymapi.Vec3(*self._config.get("camera_target", [0, 0, 0]))
        self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)

        # Initial render
        self._gym.fetch_results(self._sim, True)
        self._gym.step_graphics(self._sim)
        self._gym.draw_viewer(self._viewer, self._sim, False)

    def _refresh(self, flags: RefreshFlags = RefreshFlags.ALL):
        """
        Refresh state tensors selectively.

        Args:
            flags: Which state tensors to refresh.
        """
        if flags & RefreshFlags.ROOT_STATE:
            self._gym.refresh_actor_root_state_tensor(self._sim)
        if flags & RefreshFlags.DOF_STATE:
            self._gym.refresh_dof_state_tensor(self._sim)
        if flags & RefreshFlags.RIGID_BODY_STATE:
            self._gym.refresh_rigid_body_state_tensor(self._sim)
        if flags & RefreshFlags.FORCE_SENSOR:
            self._gym.refresh_force_sensor_tensor(self._sim)
        if flags & RefreshFlags.DOF_FORCE:
            self._gym.refresh_dof_force_tensor(self._sim)

        # Invalidate cache
        self._cache_valid = False

    def get_actor_states(self, actor_name: str) -> Dict[str, torch.Tensor]:
        """
        Get batched states for an actor.

        Args:
            actor_name: Name of the actor.

        Returns:
            Dictionary of state tensors.
        """
        self._refresh()

        # Check cache
        if self._state_cache_enabled and self._cache_valid:
            if actor_name in self._actor_state_cache:
                return self._actor_state_cache[actor_name]

        actor_states = {}

        # Root states
        actor_indices = self._actor_indices[actor_name]
        root_states = self._root_state_tensor[actor_indices].clone()
        root_states[:, 2] -= self._config.get("table_height", 0)

        actor_states.update({
            "root_state": root_states,
            "root_pos": root_states[:, :3],
            "root_rot": root_states[:, 3:7],
            "root_linvel": root_states[:, 7:10],
            "root_angvel": root_states[:, 10:],
        })

        # DOF states
        dof_indices = self._actor_dof_indices[actor_name]
        dof_states = self._dof_state_tensor[dof_indices]
        actor_states.update({
            "dof_state": dof_states,
            "dof_pos": dof_states[:, :, 0],
            "dof_vel": dof_states[:, :, 1],
        })

        # Body states
        rb_indices = self._actor_rigid_body_indices[actor_name]
        body_states = self._rigid_body_state_tensor[rb_indices]
        actor_states.update({
            "body_state": body_states,
            "body_pos": body_states[:, :, :3],
            "body_rot": body_states[:, :, 3:7],
            "body_linvel": body_states[:, :, 7:10],
            "body_angvel": body_states[:, :, 10:],
        })

        # Force sensors
        fs_indices = self._actor_force_sensor_indices[actor_name]
        fs_states = self._force_sensor_tensor[fs_indices]
        actor_states.update({
            "sensor_state": fs_states,
            "sensor_force": fs_states[:, :, :3],
            "sensor_torque": fs_states[:, :, 3:],
        })

        # DOF forces
        dof_fs_indices = self._actor_dof_force_sensor_indices[actor_name]
        actor_states["dof_force"] = self._dof_force_tensor[dof_fs_indices]

        # Cache
        if self._state_cache_enabled:
            self._actor_state_cache[actor_name] = actor_states
            self._cache_valid = True

        return actor_states

    def set_actor_states(
        self,
        actor_name: str,
        actor_states: Dict[str, torch.Tensor],
        env_ids: Optional[torch.Tensor] = None,
    ):
        """
        Set batched states for an actor.

        Args:
            actor_name: Name of the actor.
            actor_states: Dictionary of state tensors.
            env_ids: Optional tensor of env indices to update.
        """
        self._refresh(RefreshFlags.ROOT_STATE | RefreshFlags.DOF_STATE)

        actor_indices = self._actor_indices[actor_name]
        if env_ids is not None:
            actor_indices = actor_indices[env_ids]

        # Root states
        root_states = self._root_state_tensor[actor_indices]
        if "root_pos" in actor_states:
            root_states[:, :3] = self._to_tensor(actor_states["root_pos"])
            root_states[:, 2] += self._config.get("table_height", 0)
        if "root_rot" in actor_states:
            root_rot = self._to_tensor(actor_states["root_rot"])
            if root_rot.shape[1] == 3:  # Matrix
                root_rot = matrix_to_quaternion(root_rot).roll(-1, dims=1)
            root_states[:, 3:7] = root_rot
        if "root_linvel" in actor_states:
            root_states[:, 7:10] = self._to_tensor(actor_states["root_linvel"])
        if "root_angvel" in actor_states:
            root_states[:, 10:] = self._to_tensor(actor_states["root_angvel"])
        self._root_state_tensor[actor_indices] = root_states

        # DOF states
        dof_indices = self._actor_dof_indices[actor_name]
        if env_ids is not None:
            dof_indices = dof_indices[env_ids]
        dof_states = self._dof_state_tensor[dof_indices]
        if "dof_pos" in actor_states:
            dof_states[:, :, 0] = self._to_tensor(actor_states["dof_pos"])
        if "dof_vel" in actor_states:
            dof_states[:, :, 1] = self._to_tensor(actor_states["dof_vel"])
        self._dof_state_tensor[dof_indices] = dof_states

        self._flush_states()
        self._cache_valid = False

    def _to_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert to tensor on device."""
        if isinstance(data, torch.Tensor):
            return data.to(self._device)
        return torch.tensor(data, dtype=torch.float, device=self._device)

    def _flush_states(self):
        """Flush state changes to simulator."""
        self._gym.set_actor_root_state_tensor(
            self._sim, gymtorch.unwrap_tensor(self._root_state_tensor)
        )
        self._gym.set_dof_state_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_state_tensor)
        )

    def set_actor_actions(self, actor_name: str, actor_actions: torch.Tensor):
        """Set DOF position targets for an actor."""
        dof_indices = self._actor_dof_indices[actor_name]
        self._dof_position_target_tensor[dof_indices] = self._to_tensor(actor_actions)
        self._gym.set_dof_position_target_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_position_target_tensor)
        )

    def disable_gravity(self, actor_name: Optional[str] = None):
        """Disable gravity for actor(s)."""
        if actor_name is None:
            for name in self._actor_indices:
                self._enable_gravity[name] = False
            self._forces[:, 2] = 9.81 * self._rigid_body_mass
        else:
            self._enable_gravity[actor_name] = False
            rb_indices = self._actor_rigid_body_indices[actor_name]
            self._forces[rb_indices, 2] = 9.81 * self._rigid_body_mass[rb_indices]

    def enable_gravity(self, actor_name: Optional[str] = None):
        """Enable gravity for actor(s)."""
        if actor_name is None:
            for name in self._actor_indices:
                self._enable_gravity[name] = True
            self._forces[:, 2] = 0.0
        else:
            self._enable_gravity[actor_name] = True
            rb_indices = self._actor_rigid_body_indices[actor_name]
            self._forces[rb_indices, 2] = 0.0

    def step(self):
        """Step physics simulation."""
        control_freq_inv = self._config.get("control_freq_inv", 1)

        for _ in range(control_freq_inv):
            # Apply gravity compensation
            self._gym.apply_rigid_body_force_tensors(
                self._sim, gymtorch.unwrap_tensor(self._forces)
            )

            # Simulate
            self._gym.simulate(self._sim)
            self._gym.fetch_results(self._sim, True)

            # Clear velocities for gravity-disabled actors
            self._refresh(RefreshFlags.ROOT_STATE)
            for actor_name, enabled in self._enable_gravity.items():
                if not enabled:
                    actor_indices = self._actor_indices[actor_name]
                    self._root_state_tensor[actor_indices, 7:] = 0.0
            self._flush_states()

            # Update viewer
            self._update_viewer()

        self._refresh()
        self._cache_valid = False

    def _update_viewer(self):
        """Update visualization viewer."""
        if self._viewer is None:
            return

        # Check window close
        if self._gym.query_viewer_has_closed(self._viewer):
            logger.info("Viewer closed")
            self.close()
            raise SystemExit(0)

        # Handle keyboard events
        for evt in self._gym.query_viewer_action_events(self._viewer):
            if evt.action == "QUIT" and evt.value > 0:
                logger.info("Escape pressed")
                raise SystemExit(0)
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self._enable_viewer_sync = not self._enable_viewer_sync

        # Render
        if self._enable_viewer_sync and not self._async_render:
            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(self._viewer, self._sim, True)
            self._gym.sync_frame_time(self._sim)
        else:
            self._gym.poll_viewer_events(self._viewer)

    def set_async_render(self, enabled: bool):
        """Enable/disable async rendering."""
        self._async_render = enabled

    def close(self):
        """Close simulator and viewer."""
        if not self._headless and self._viewer is not None:
            self._gym.destroy_viewer(self._viewer)
        self._gym.destroy_sim(self._sim)
        logger.info("Simulator closed")


def get_simulator(simulator_type: str):
    """
    Get simulator class by type.

    Args:
        simulator_type: Type of simulator ('isaacgym').

    Returns:
        Simulator class.
    """
    if simulator_type == "isaacgym":
        return IsaacGymSimulator
    else:
        raise ValueError(f"Unknown simulator type: {simulator_type}")
