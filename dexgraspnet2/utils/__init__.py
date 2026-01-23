"""Utility functions for DexGraspNet2."""

from dexgraspnet2.utils.point_cloud import (
    voxelize_point_cloud,
    depth_to_point_cloud,
)
from dexgraspnet2.utils.geometric_conversions import (
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler_xyz,
    euler_xyz_to_rotation_matrix,
    rotation_matrix_to_euler_zyx,
    euler_zyx_to_rotation_matrix,
    rotation_matrix_to_euler_xyz_intrinsic,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_inverse,
    transform_point,
    transform_points,
)

__all__ = [
    # Point cloud utilities
    "voxelize_point_cloud",
    "depth_to_point_cloud",
    # Geometric conversions
    "rotation_matrix_to_quaternion",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_euler_xyz",
    "euler_xyz_to_rotation_matrix",
    "rotation_matrix_to_euler_zyx",
    "euler_zyx_to_rotation_matrix",
    "rotation_matrix_to_euler_xyz_intrinsic",
    "normalize_quaternion",
    "quaternion_multiply",
    "quaternion_inverse",
    "transform_point",
    "transform_points",
]
