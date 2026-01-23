"""
Hand configuration module for DexGraspNet2.

Provides a unified interface for configuring different dexterous hands.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class HandConfig:
    """
    Configuration for a dexterous hand.

    This class encapsulates all hand-specific parameters needed for:
    - Loading the robot model (URDF)
    - Neural network configuration (DoF count)
    - Joint limits and ordering

    Attributes:
        name: Unique identifier for the hand (e.g., 'leap_hand', 'inspire_hand').
        urdf_path: Path to the URDF file describing the hand kinematics.
        meta_path: Path to the meta YAML file with collision pairs.
        num_dofs: Number of degrees of freedom (actuated joints).
        joint_names: Ordered list of joint names matching the neural network output.
        joint_lower_limits: Lower limits for each joint in radians.
        joint_upper_limits: Upper limits for each joint in radians.
        fingertip_links: Names of links representing fingertips (for contact).
        wrist_link: Name of the wrist/base link.
    """

    name: str
    urdf_path: Path
    meta_path: Path
    num_dofs: int
    joint_names: List[str]
    joint_lower_limits: List[float]
    joint_upper_limits: List[float]
    fingertip_links: List[str] = field(default_factory=list)
    wrist_link: str = "hand_base_link"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if len(self.joint_names) != self.num_dofs:
            raise ValueError(
                f"Number of joint names ({len(self.joint_names)}) must match "
                f"num_dofs ({self.num_dofs})"
            )
        if len(self.joint_lower_limits) != self.num_dofs:
            raise ValueError(
                f"Number of lower limits ({len(self.joint_lower_limits)}) must match "
                f"num_dofs ({self.num_dofs})"
            )
        if len(self.joint_upper_limits) != self.num_dofs:
            raise ValueError(
                f"Number of upper limits ({len(self.joint_upper_limits)}) must match "
                f"num_dofs ({self.num_dofs})"
            )

        self.urdf_path = Path(self.urdf_path)
        self.meta_path = Path(self.meta_path)

        if not self.urdf_path.exists():
            logger.warning(f"URDF path does not exist: {self.urdf_path}")
        if not self.meta_path.exists():
            logger.warning(f"Meta path does not exist: {self.meta_path}")

    @classmethod
    def from_yaml(cls, config_path: Path) -> "HandConfig":
        """
        Load hand configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            HandConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If required fields are missing.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        required_fields = [
            "name", "urdf_path", "meta_path", "num_dofs",
            "joint_names", "joint_lower_limits", "joint_upper_limits"
        ]

        for field_name in required_fields:
            if field_name not in config_dict:
                raise ValueError(f"Missing required field: {field_name}")

        # Make paths relative to config file location
        base_dir = config_path.parent
        if not Path(config_dict["urdf_path"]).is_absolute():
            config_dict["urdf_path"] = base_dir / config_dict["urdf_path"]
        if not Path(config_dict["meta_path"]).is_absolute():
            config_dict["meta_path"] = base_dir / config_dict["meta_path"]

        return cls(**config_dict)

    def to_yaml(self, config_path: Path) -> None:
        """
        Save hand configuration to a YAML file.

        Args:
            config_path: Path where to save the configuration.
        """
        config_dict = {
            "name": self.name,
            "urdf_path": str(self.urdf_path),
            "meta_path": str(self.meta_path),
            "num_dofs": self.num_dofs,
            "joint_names": self.joint_names,
            "joint_lower_limits": self.joint_lower_limits,
            "joint_upper_limits": self.joint_upper_limits,
            "fingertip_links": self.fingertip_links,
            "wrist_link": self.wrist_link,
        }

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

    @classmethod
    def leap_hand(cls, base_path: Optional[Path] = None) -> "HandConfig":
        """
        Create configuration for the LEAP Hand (16 DoF).

        The LEAP Hand has 4 fingers with 4 joints each:
        - Index finger: j0-j3
        - Middle finger: j4-j7
        - Ring finger: j8-j11
        - Thumb: j12-j15

        Args:
            base_path: Base path for robot_models directory.
                       If None, uses default relative path.

        Returns:
            HandConfig for LEAP Hand.
        """
        if base_path is None:
            base_path = Path("robot_models")
        else:
            base_path = Path(base_path)

        # Joint names in the order expected by the neural network
        joint_names = [
            "j0", "j1", "j2", "j3",     # Index finger
            "j4", "j5", "j6", "j7",     # Middle finger
            "j8", "j9", "j10", "j11",   # Ring finger
            "j12", "j13", "j14", "j15", # Thumb
        ]

        # Joint limits from URDF (in radians)
        joint_lower_limits = [
            -1.047, -0.314, -0.506, -0.366,  # Index
            -1.047, -0.314, -0.506, -0.366,  # Middle
            -1.047, -0.314, -0.506, -0.366,  # Ring
            -0.349, -0.47, -1.20, -1.34,     # Thumb
        ]

        joint_upper_limits = [
            1.047, 2.23, 1.885, 2.042,  # Index
            1.047, 2.23, 1.885, 2.042,  # Middle
            1.047, 2.23, 1.885, 2.042,  # Ring
            2.094, 2.443, 1.90, 1.88,   # Thumb
        ]

        fingertip_links = [
            "fingertip",    # Index
            "fingertip_2",  # Middle
            "fingertip_3",  # Ring
            "thumb_fingertip",  # Thumb
        ]

        return cls(
            name="leap_hand",
            urdf_path=base_path / "urdf" / "leap_hand_simplified.urdf",
            meta_path=base_path / "meta" / "leap_hand" / "meta.yaml",
            num_dofs=16,
            joint_names=joint_names,
            joint_lower_limits=joint_lower_limits,
            joint_upper_limits=joint_upper_limits,
            fingertip_links=fingertip_links,
            wrist_link="hand_base_link",
        )

    def get_joint_limits(self) -> Tuple[List[float], List[float]]:
        """
        Get joint limits as a tuple.

        Returns:
            Tuple of (lower_limits, upper_limits).
        """
        return self.joint_lower_limits.copy(), self.joint_upper_limits.copy()

    def clamp_joints(self, joints: List[float]) -> List[float]:
        """
        Clamp joint values to be within limits.

        Args:
            joints: List of joint values.

        Returns:
            Clamped joint values.
        """
        return [
            max(lo, min(hi, val))
            for val, lo, hi in zip(
                joints, self.joint_lower_limits, self.joint_upper_limits
            )
        ]

    @classmethod
    def inspire_hand(cls, base_path: Optional[Path] = None) -> "HandConfig":
        """
        Create configuration for the Inspire Hand (6 DoF).

        The Inspire Hand has 5 fingers but only 6 actuated DoF:
        - Thumb: 2 DoF (yaw + pitch, with coupled intermediate/distal)
        - Index/Middle/Ring/Pinky: 1 DoF each (proximal, with coupled intermediate)

        Args:
            base_path: Base path for third_party/dex-urdf directory.
                       If None, uses default relative path.

        Returns:
            HandConfig for Inspire Hand.
        """
        if base_path is None:
            base_path = Path("third_party/dex-urdf/robots/hands/inspire_hand")
        else:
            base_path = Path(base_path)

        # Actuated joint names (mimic joints excluded)
        joint_names = [
            "thumb_proximal_yaw_joint",
            "thumb_proximal_pitch_joint",
            "index_proximal_joint",
            "middle_proximal_joint",
            "ring_proximal_joint",
            "pinky_proximal_joint",
        ]

        # Joint limits from URDF (in radians)
        joint_lower_limits = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joint_upper_limits = [1.308, 0.6, 1.47, 1.47, 1.47, 1.47]

        fingertip_links = [
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ]

        return cls(
            name="inspire_hand",
            urdf_path=base_path / "inspire_hand_right.urdf",
            meta_path=Path("dexgraspnet2/configs/hands/inspire_hand_meta.yaml"),
            num_dofs=6,
            joint_names=joint_names,
            joint_lower_limits=joint_lower_limits,
            joint_upper_limits=joint_upper_limits,
            fingertip_links=fingertip_links,
            wrist_link="hand_base_link",
        )
