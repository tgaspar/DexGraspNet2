"""
Point cloud processing utilities for DexGraspNet2.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import MinkowskiEngine as ME

logger = logging.getLogger(__name__)


def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """
    Convert depth image to point cloud.

    Args:
        depth: Depth image (H, W) in millimeters or meters.
        intrinsics: Camera intrinsic matrix (3, 3).
        depth_scale: Scale factor (1000 for mm->m, 1 for m).

    Returns:
        Point cloud (H*W, 3) in meters.
    """
    height, width = depth.shape

    # Create pixel coordinate grid
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # Get intrinsic parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Convert depth to meters
    z = depth.astype(np.float32) / depth_scale

    # Back-project to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack into point cloud
    points = np.stack([x, y, z], axis=-1)
    points = points.reshape(-1, 3)

    return points


def voxelize_point_cloud(
    point_cloud: np.ndarray,
    voxel_size: float = 0.005,
) -> Dict[str, torch.Tensor]:
    """
    Voxelize point cloud for MinkowskiEngine sparse convolution.

    This function converts a dense point cloud into a sparse tensor format
    compatible with MinkowskiEngine's sparse convolution operations.

    Args:
        point_cloud: Point cloud (N, 3) or (B, N, 3).
        voxel_size: Size of each voxel in meters.

    Returns:
        Dictionary containing:
            - point_clouds: Original point clouds (B, N, 3)
            - coors: Sparse coordinates with batch indices (M, 4)
            - feats: Features (ones) for each coordinate (M, 3)
            - quantize2original: Mapping from quantized to original indices (B*N,)
    """
    # Handle single point cloud (add batch dimension)
    if point_cloud.ndim == 2:
        point_cloud = point_cloud[np.newaxis, ...]

    point_cloud = torch.as_tensor(point_cloud, dtype=torch.float32)
    batch_size, num_points, _ = point_cloud.shape

    # Quantize coordinates
    coors = point_cloud / voxel_size
    feats = torch.ones_like(point_cloud)

    # Collate batch with sparse operations
    coors_list = [coor for coor in coors]
    feats_list = [feat for feat in feats]

    coordinates_batch, features_batch = ME.utils.sparse_collate(
        coors_list, feats_list
    )

    # Quantize and get mapping
    coordinates_batch, features_batch, original2quantize, quantize2original = \
        ME.utils.sparse_quantize(
            coordinates_batch.float(),
            features_batch,
            return_index=True,
            return_inverse=True
        )

    return {
        "point_clouds": point_cloud,
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original,
    }


def get_workspace_mask(
    points: np.ndarray,
    segmentation: np.ndarray,
    camera_pose: np.ndarray,
    workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Get mask for points within the workspace.

    Args:
        points: Point cloud in camera frame (N, 3).
        segmentation: Object segmentation mask (N,).
        camera_pose: Camera-to-world transformation (4, 4).
        workspace_bounds: Workspace bounds in world frame.
            Default: x=[-0.5, 0.5], y=[-0.5, 0.5], z=[-0.05, 0.5]

    Returns:
        Boolean mask (N,) indicating points within workspace.
    """
    if workspace_bounds is None:
        workspace_bounds = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.05, 0.5),
        }

    # Transform points to world frame
    rotation = camera_pose[:3, :3]
    translation = camera_pose[:3, 3]
    points_world = (rotation @ points.T).T + translation

    # Create bounds mask
    mask = (
        (points_world[:, 0] >= workspace_bounds["x"][0]) &
        (points_world[:, 0] <= workspace_bounds["x"][1]) &
        (points_world[:, 1] >= workspace_bounds["y"][0]) &
        (points_world[:, 1] <= workspace_bounds["y"][1]) &
        (points_world[:, 2] >= workspace_bounds["z"][0]) &
        (points_world[:, 2] <= workspace_bounds["z"][1])
    )

    # Only include object points (segmentation > 0)
    mask = mask & (segmentation > 0)

    return mask


def to_voxel_center(
    points: torch.Tensor,
    voxel_size: float,
) -> torch.Tensor:
    """
    Calculate the center of the voxel containing each point.

    Args:
        points: Point coordinates (..., 3).
        voxel_size: Size of each voxel.

    Returns:
        Voxel centers (..., 3).
    """
    return torch.div(points, voxel_size, rounding_mode='floor') * voxel_size + voxel_size / 2


def sample_farthest_points(
    points: torch.Tensor,
    num_samples: int,
    random_start: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points using farthest point sampling (FPS).

    Args:
        points: Input points (N, 3) or (B, N, 3).
        num_samples: Number of points to sample.
        random_start: Whether to start from a random point.

    Returns:
        Tuple of (sampled_points, indices).
    """
    from pytorch3d.ops import sample_farthest_points as fps

    # Handle single point cloud
    single_batch = points.ndim == 2
    if single_batch:
        points = points.unsqueeze(0)

    sampled, indices = fps(
        points.contiguous(),
        K=num_samples,
        random_start_point=random_start
    )

    if single_batch:
        sampled = sampled.squeeze(0)
        indices = indices.squeeze(0)

    return sampled, indices
