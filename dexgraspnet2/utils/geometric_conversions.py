"""
Geometric conversion utilities for rotations and transformations.

Provides conversions between different rotation representations:
- Rotation matrices (3x3)
- Quaternions (x, y, z, w)
- Euler angles (various orders)
"""

import numpy as np
from typing import Tuple


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Quaternion as (x, y, z, w).
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([x, y, z, w])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        q: Quaternion as (x, y, z, w).

    Returns:
        3x3 rotation matrix.
    """
    x, y, z, w = q

    # Normalize quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

    return R


def rotation_matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to Euler angles (XYZ order).

    The rotation is decomposed as R = Rz * Ry * Rx.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Euler angles [rx, ry, rz] in radians.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def euler_xyz_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (XYZ order) to 3x3 rotation matrix.

    The rotation is composed as R = Rz * Ry * Rx.

    Args:
        euler: Euler angles [rx, ry, rz] in radians.

    Returns:
        3x3 rotation matrix.
    """
    rx, ry, rz = euler

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    # R = Rz * Ry * Rx
    R = np.array([
        [cy*cz, cz*sx*sy - cx*sz, cx*cz*sy + sx*sz],
        [cy*sz, cx*cz + sx*sy*sz, cx*sy*sz - cz*sx],
        [-sy, cy*sx, cx*cy]
    ])

    return R


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to Euler angles (ZYX order).

    The rotation is decomposed as R = Rx * Ry * Rz.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Euler angles [rz, ry, rx] in radians.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[0, 1] ** 2)

    singular = sy < 1e-6

    if not singular:
        z = np.arctan2(R[0, 1], R[0, 0])
        y = np.arctan2(-R[0, 2], sy)
        x = np.arctan2(R[1, 2], R[2, 2])
    else:
        z = 0
        y = np.arctan2(-R[0, 2], sy)
        x = np.arctan2(-R[2, 1], R[1, 1])

    return np.array([z, y, x])


def euler_zyx_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (ZYX order) to 3x3 rotation matrix.

    The rotation is composed as R = Rx * Ry * Rz.

    Args:
        euler: Euler angles [rz, ry, rx] in radians.

    Returns:
        3x3 rotation matrix.
    """
    rz, ry, rx = euler

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    # R = Rx * Ry * Rz
    R = np.array([
        [cy*cz, -cy*sz, sy],
        [cx*sz + cz*sx*sy, cx*cz - sx*sy*sz, -cy*sx],
        [sx*sz - cx*cz*sy, cz*sx + cx*sy*sz, cx*cy]
    ])

    return R


def rotation_matrix_to_euler_xyz_intrinsic(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to Euler angles for URDF virtual joint chain.

    The URDF joint chain applies rotations as: x_rotation -> y_rotation -> z_rotation
    This results in R = Rx(rx) @ Ry(ry) @ Rz(rz) (intrinsic XYZ order).

    This is the correct decomposition for setting DOF targets on the virtual 6-DOF
    joint chain used in *_free URDF variants.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Euler angles [rx, ry, rz] in radians, in DOF order for URDF.
    """
    # For R = Rx @ Ry @ Rz:
    # R[0,2] = sin(ry)
    # R[1,2] = -sin(rx)*cos(ry)
    # R[2,2] = cos(rx)*cos(ry)
    # R[0,1] = -cos(ry)*sin(rz)
    # R[0,0] = cos(ry)*cos(rz)
    cy = np.sqrt(R[0, 0]**2 + R[0, 1]**2)

    if cy > 1e-6:
        rx = np.arctan2(-R[1, 2], R[2, 2])
        ry = np.arctan2(R[0, 2], cy)
        rz = np.arctan2(-R[0, 1], R[0, 0])
    else:
        # Gimbal lock case
        rx = np.arctan2(-R[2, 1], R[1, 1])
        ry = np.arctan2(R[0, 2], cy)
        rz = 0

    return np.array([rx, ry, rz])


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion to unit length.

    Args:
        q: Quaternion as (x, y, z, w).

    Returns:
        Normalized quaternion.
    """
    norm = np.sqrt(np.sum(q ** 2))
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.

    Args:
        q1: First quaternion (x, y, z, w).
        q2: Second quaternion (x, y, z, w).

    Returns:
        Product quaternion q1 * q2.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a quaternion.

    Args:
        q: Quaternion as (x, y, z, w).

    Returns:
        Inverse quaternion.
    """
    x, y, z, w = q
    norm_sq = x*x + y*y + z*z + w*w
    return np.array([-x, -y, -z, w]) / norm_sq


def transform_point(point: np.ndarray, translation: np.ndarray,
                    rotation: np.ndarray) -> np.ndarray:
    """
    Transform a 3D point by rotation and translation.

    Args:
        point: 3D point (3,).
        translation: Translation vector (3,).
        rotation: 3x3 rotation matrix.

    Returns:
        Transformed point (3,).
    """
    return rotation @ point + translation


def transform_points(points: np.ndarray, translation: np.ndarray,
                     rotation: np.ndarray) -> np.ndarray:
    """
    Transform multiple 3D points by rotation and translation.

    Args:
        points: 3D points (N, 3).
        translation: Translation vector (3,).
        rotation: 3x3 rotation matrix.

    Returns:
        Transformed points (N, 3).
    """
    return (rotation @ points.T).T + translation
