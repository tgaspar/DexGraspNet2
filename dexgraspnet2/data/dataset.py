"""
Dataset module for DexGraspNet2 training.

Provides dataset classes for loading grasp data from the GraspNet dataset.
"""

import collections.abc as container_abcs
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.io as scio
import torch
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None

from dexgraspnet2.configs.training_config import DataConfig, TrainingConfig

logger = logging.getLogger(__name__)


# Dataset split definitions
SPLITS = {
    "single": range(1),
    "single1": range(1, 2),  # Scene 1 only (for testing with limited data)
    "half": range(45),
    "train": range(100),
    "train_1": range(90),
    "val": range(90, 100),
    "test": range(100, 190),
    "test_seen": range(100, 130),
    "test_similar": range(130, 160),
    "test_novel": range(160, 190),
}


class InfiniteLoader:
    """
    Wrapper for DataLoader that provides infinite iteration.

    This class wraps a PyTorch DataLoader to provide continuous data
    iteration without raising StopIteration, automatically resetting
    the iterator when the dataset is exhausted.

    Args:
        loader: PyTorch DataLoader instance to wrap.

    Example:
        >>> loader = InfiniteLoader(DataLoader(dataset, batch_size=8))
        >>> for _ in range(10000):
        ...     batch = loader.get()
    """

    def __init__(self, loader: DataLoader):
        """Initialize the infinite loader."""
        self._loader = loader
        self._iter = iter(self._loader)

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get the next batch of data.

        Returns:
            Dictionary containing batch data tensors.
        """
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            data = next(self._iter)
        return data

    @property
    def dataset(self) -> Dataset:
        """Get the underlying dataset."""
        return self._loader.dataset


# Alias for backward compatibility
Loader = InfiniteLoader


class GraspNetDataset(Dataset):
    """
    Dataset for loading GraspNet scenes with grasp annotations.

    This dataset loads depth images, point clouds, and grasp annotations
    from the GraspNet dataset format. It supports both training and
    evaluation modes.

    Args:
        config: Full training configuration or data configuration.
        split: Dataset split name (e.g., 'train', 'val', 'test_seen').
        is_train: Whether this is for training (enables augmentation).
        is_eval: Whether this is for evaluation (deterministic iteration).
        data_root: Root directory for data (default: 'data').

    Attributes:
        scene_id: List of scene identifiers.
        views: List of (scene_id, view_index) tuples.
    """

    def __init__(
        self,
        config: Union[TrainingConfig, DataConfig, Dict],
        split: str,
        is_train: bool = False,
        is_eval: bool = False,
        data_root: str = "data",
    ):
        """Initialize the GraspNet dataset."""
        self._data_root = Path(data_root)
        self._is_train = is_train
        self._is_eval = is_eval

        # Handle different config types
        if isinstance(config, TrainingConfig):
            self._full_config = config
            self._config = config.data
        elif isinstance(config, DataConfig):
            self._full_config = None
            self._config = config
        elif isinstance(config, dict):
            # Legacy dict config support
            self._full_config = config
            self._config = config.get("data", config)
            if isinstance(self._config, dict):
                self._config = _dict_to_namespace(self._config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        # Parse split and build scene/view lists
        split_base = split.split("-")[0]
        if split_base not in SPLITS:
            raise ValueError(
                f"Unknown split: {split_base}. Valid: {list(SPLITS.keys())}"
            )

        scene_fraction = getattr(self._config, "scene_fraction", 1)
        scene_range = SPLITS[split_base]
        if is_train:
            scene_range = scene_range[::scene_fraction]

        self.scene_id = [f"scene_{str(x).zfill(4)}" for x in scene_range]

        ann_id = range(1 if split == "single" else 256)
        self.views: List[Tuple[str, int]] = []
        for scene in self.scene_id:
            for i in ann_id:
                self.views.append((scene, i))

        # Initialize robot model for dexterous hands
        robot = getattr(self._config, "robot", "leap_hand")
        if robot == "gripper":
            self._robot_model = None
            self._joint_names = None
            # Import pose refine only when needed
            try:
                from src.utils.pose_refine import PoseRefine

                self._refiner = PoseRefine()
            except ImportError:
                self._refiner = None
                logger.warning("PoseRefine not available")
        else:
            self._refiner = None
            try:
                from src.utils.robot_model import RobotModel

                urdf_path = os.path.join("robot_models", "urdf", f"{robot}.urdf")
                meta_path = os.path.join("robot_models", "meta", robot, "meta.yaml")
                self._robot_model = RobotModel(urdf_path, meta_path)
                self._joint_names = self._robot_model.joint_names
            except Exception as e:
                logger.warning(f"Could not load robot model: {e}")
                self._robot_model = None
                self._joint_names = None

        # Determine which categories to use
        self._cates = ["orig"]
        if is_train and "-part" in split:
            self._cates.append("part")

        logger.info(
            f"Initialized GraspNetDataset: split={split}, "
            f"scenes={len(self.scene_id)}, views={len(self.views)}, "
            f"train={is_train}, eval={is_eval}"
        )

    def __len__(self) -> int:
        """
        Get dataset length.

        Returns:
            100000 for training (virtual epoch), actual length for eval.
        """
        return 100000 if self._is_train else len(self.views)

    def __getitem__(self, dataset_idx: int) -> Dict[str, np.ndarray]:
        """
        Get a single data sample.

        Args:
            dataset_idx: Index into the dataset.

        Returns:
            Dictionary containing:
                - point_clouds: (N, 3) point cloud coordinates
                - coors: (N, 3) voxel coordinates
                - feats: (N, 3) point features (ones)
                - seg: (N,) segmentation labels
                - objectness: (N,) binary objectness labels
                - graspness: (N,) graspness scores
                - rot: (K, 3, 3) grasp rotations
                - trans: (K, 3) grasp translations
                - centers: (K,) indices of grasp center points
                - qpos: (K, J) joint positions
                - has_graspness: (1,) whether graspness data exists
        """
        cate = random.choice(self._cates)

        try:
            return self._load_sample(dataset_idx, cate)
        except Exception as e:
            logger.debug(f"Error loading sample {dataset_idx} (cate={cate}): {e}")
            # Retry with random sample (with retry limit to prevent infinite recursion)
            if not hasattr(self, "_retry_count"):
                self._retry_count = 0
            self._retry_count += 1
            if self._retry_count > 10:
                self._retry_count = 0
                raise RuntimeError(
                    f"Failed to load sample after 10 retries. Last error: {e}"
                )
            result = self.__getitem__(random.randint(0, len(self) - 1))
            self._retry_count = 0
            return result

    def _load_sample(self, dataset_idx: int, cate: str) -> Dict[str, np.ndarray]:
        """
        Load a single sample from disk.

        Args:
            dataset_idx: Index into the dataset.
            cate: Category type ('orig' or 'part').

        Returns:
            Dictionary containing sample data.

        Raises:
            Exception: If loading fails.
        """
        # Determine scene and view
        if cate == "orig":
            if self._is_eval:
                scene, view = self.views[dataset_idx]
            else:
                scene, view = random.choice(self.views)
        elif cate == "part":
            scene_fraction = getattr(self._config, "scene_fraction", 1)
            orig_scenes = list(range(100))[::scene_fraction]
            scene = f"scene_{1000 + random.choice(orig_scenes) * 75 + random.randint(0, 74)}"
            view = random.randint(0, 255)

        str_view = str(view).zfill(4)
        render = getattr(self._config, "render", False)
        suffix = "_gt" if render else ""
        camera = getattr(self._config, "camera", "realsense")

        # Load depth, segmentation, and metadata
        path = self._data_root / "scenes" / scene / camera
        depth = np.array(Image.open(path / f"depth{suffix}" / f"{str_view}.png"))
        seg = np.array(Image.open(path / f"label{suffix}" / f"{str_view}.png"))

        # Try to load edge data
        try:
            edge = np.array(Image.open(path / f"edge{suffix}" / f"{str_view}.png"))
        except Exception:
            edge = None

        meta = scio.loadmat(str(path / "meta" / f"{str_view}.mat"))
        intrinsics = meta["intrinsic_matrix"]
        factor_depth = meta["factor_depth"]
        camera_poses = np.load(str(path / "camera_poses.npy"))
        align_mat = np.load(str(path / "cam0_wrt_table.npy"))

        # Convert depth to point cloud (or load precomputed)
        cloud = self._load_or_compute_cloud(
            path, str_view, depth, intrinsics, factor_depth
        )
        depth_mask = depth > 0
        trans = np.dot(align_mat, camera_poses[view])

        # Check if segmentation has valid objects
        if not seg.any():
            raise ValueError("No valid segmentation")

        workspace_mask = self._get_workspace_mask(cloud, seg, trans)
        mask = depth_mask & workspace_mask
        cloud = cloud[mask]
        seg = seg[mask]

        # Random sample points
        num_points = getattr(self._config, "num_points", 40000)
        idxs = np.random.choice(len(cloud), num_points, replace=True)
        cloud = cloud[idxs]
        seg = seg[idxs]

        # For evaluation, return early
        if self._is_eval:
            voxel_size = getattr(self._config, "voxel_size", 0.005)
            ret_dict = {
                "scene": np.array([int(scene.split("_")[-1])]),
                "view": np.array([view]),
                "point_clouds": cloud.astype(np.float32),
                "coors": (cloud / voxel_size).astype(np.float32),
                "feats": np.ones_like(cloud).astype(np.float32),
                "seg": seg.astype(np.int64),
            }
            if edge is not None:
                ret_dict["edge"] = edge[mask][idxs]
            return ret_dict

        # Load graspness data
        fraction = getattr(self._config, "fraction", 1)
        frac_suffix = "" if fraction == 1 else f"_{fraction}"
        graspness_data = getattr(self._config, "graspness_data", "dex_graspness")
        graspness_path = (
            self._data_root
            / f"{graspness_data}{frac_suffix}"
            / scene
            / camera
            / f"{str_view}.npy"
        )

        if graspness_path.exists():
            graspness = np.load(str(graspness_path)).reshape(-1)[idxs]
            graspness = np.log(graspness + 1e-3)
            has_graspness = 1
        else:
            graspness = np.zeros(len(cloud), dtype=np.float32)
            has_graspness = 0

        # Load grasp poses
        robot = getattr(self._config, "robot", "leap_hand")
        sample_total = getattr(self._config, "sample_total", 256)
        k = getattr(self._config, "k", 64)

        if robot == "gripper":
            new_rot, new_trans, qpos, grasp_points = self._load_gripper_grasps(
                scene, camera, view, camera_poses, fraction, sample_total
            )
        else:
            new_rot, new_trans, qpos, grasp_points = self._load_dex_grasps(
                scene, camera, view, camera_poses, fraction, sample_total, frac_suffix
            )

        # Find nearest points in cloud for each grasp
        max_point_dis = getattr(self._config, "max_point_dis", 0.02)
        centers, indices = self._match_grasps_to_cloud(
            cloud, grasp_points, k, max_point_dis
        )

        if len(indices) == 0:
            raise ValueError("No valid grasps found")

        # Select grasps
        new_rot = new_rot[indices]
        new_trans = new_trans[indices]
        qpos = qpos[indices]
        centers = centers[indices]

        voxel_size = getattr(self._config, "voxel_size", 0.005)
        ret_dict = {
            "point_clouds": cloud.astype(np.float32),
            "coors": (cloud / voxel_size).astype(np.float32),
            "feats": np.ones_like(cloud).astype(np.float32),
            "seg": seg.astype(np.int64),
            "objectness": (seg > 0).astype(np.int64),
            "graspness": graspness.astype(np.float32),
            "rot": new_rot.astype(np.float32),
            "trans": new_trans.astype(np.float32),
            "centers": centers.astype(np.float32),
            "has_graspness": np.array([has_graspness]),
            "qpos": qpos.astype(np.float32),
        }

        if self._is_train:
            ret_dict = self._augment_data(ret_dict)

        return ret_dict

    def _load_or_compute_cloud(
        self,
        scene_path: Path,
        str_view: str,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        factor_depth: float,
    ) -> np.ndarray:
        """
        Load precomputed point cloud or compute from depth.

        First checks for precomputed cloud in data/precomputed_clouds/,
        falls back to computing from depth image if not available.

        Args:
            scene_path: Path to scene camera directory.
            str_view: View index as zero-padded string (e.g., "0000").
            depth: (H, W) depth image in raw units.
            intrinsics: (3, 3) camera intrinsic matrix.
            factor_depth: Scale factor for depth values.

        Returns:
            (H*W, 3) point cloud coordinates.
        """
        # Check for precomputed cloud
        # scene_path is like data/scenes/scene_0000/realsense
        # precomputed would be data/precomputed_clouds/scene_0000/realsense/0000.npy
        precomputed_path = (
            self._data_root
            / "precomputed_clouds"
            / scene_path.parent.name
            / scene_path.name
            / f"{str_view}.npy"
        )

        if precomputed_path.exists():
            cloud = np.load(str(precomputed_path))
            # Convert float16 back to float32 if needed
            return cloud.astype(np.float32)

        # Fall back to computing from depth
        return self._depth_to_point_cloud(depth, intrinsics, factor_depth)

    def _depth_to_point_cloud(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        factor_depth: float,
    ) -> np.ndarray:
        """
        Convert depth image to point cloud.

        Args:
            depth: (H, W) depth image in raw units.
            intrinsics: (3, 3) camera intrinsic matrix.
            factor_depth: Scale factor for depth values.

        Returns:
            (H*W, 3) point cloud coordinates.
        """
        # Import from original codebase for consistency
        try:
            from src.utils.pc import depth_image_to_point_cloud

            return depth_image_to_point_cloud(depth, intrinsics, factor_depth)
        except ImportError:
            # Fallback implementation
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            h, w = depth.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            z = depth.astype(np.float32) / factor_depth
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            return np.stack([x, y, z], axis=-1).reshape(-1, 3)

    def _get_workspace_mask(
        self,
        cloud: np.ndarray,
        seg: np.ndarray,
        trans: np.ndarray,
    ) -> np.ndarray:
        """
        Get workspace mask for filtering point cloud.

        Args:
            cloud: (N, 3) point cloud.
            seg: (N,) segmentation labels.
            trans: (4, 4) transformation matrix.

        Returns:
            (N,) boolean mask for valid workspace points.
        """
        try:
            from src.utils.pc import get_workspace_mask

            return get_workspace_mask(cloud, seg, trans)
        except ImportError:
            # Simple fallback: keep all segmented points
            return seg > 0

    def _load_gripper_grasps(
        self,
        scene: str,
        camera: str,
        view: int,
        camera_poses: np.ndarray,
        fraction: int,
        sample_total: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load gripper grasp data.

        Args:
            scene: Scene identifier.
            camera: Camera type.
            view: View index.
            camera_poses: Camera pose matrices.
            fraction: Data fraction.
            sample_total: Total samples to load.

        Returns:
            Tuple of (rotations, translations, joint positions, grasp points).
        """
        poses_6d = np.load(
            str(self._data_root / "gripper_grasps" / scene / camera / "poses.npy")
        )[::fraction]
        grasp_points = np.load(
            str(self._data_root / "gripper_grasps" / scene / camera / "points.npy")
        )[::fraction]

        resample = getattr(self._config, "resample", True)
        if resample:
            can_grasp_ids = np.unique(poses_6d[:, -1])
            rand_idxs = np.random.randint(0, len(can_grasp_ids), sample_total)
            samples = []
            point_samples = []
            for i, idx in enumerate(can_grasp_ids):
                num = (rand_idxs == i).sum()
                obj_poses_6d = poses_6d[poses_6d[:, -1] == idx]
                obj_rand_idxs = np.random.choice(len(obj_poses_6d), num, replace=True)
                samples.append(obj_poses_6d[obj_rand_idxs])
                point_samples.append(
                    grasp_points[poses_6d[:, -1] == idx][obj_rand_idxs]
                )
            poses_6d = np.concatenate(samples)
            point_samples = np.concatenate(point_samples)
            permute = np.random.permutation(len(poses_6d))
            poses_6d = poses_6d[permute]
            grasp_points = point_samples[permute]
        else:
            idxs = np.random.choice(len(poses_6d), sample_total, replace=True)
            poses_6d = poses_6d[idxs]
            grasp_points = grasp_points[idxs]

        rot = poses_6d[:, -13:-4].reshape(-1, 3, 3)
        trans = poses_6d[:, -4:-1]

        # Transform to camera frame
        new_rot = np.einsum("ji,njk->nik", camera_poses[view, :3, :3], rot)
        new_trans = np.einsum(
            "ji,nj->ni", camera_poses[view, :3, :3], trans - camera_poses[view, :3, 3]
        )
        grasp_points = np.einsum(
            "ba,nb->na",
            camera_poses[view, :3, :3],
            grasp_points - camera_poses[view, :3, 3],
        )

        qpos = poses_6d[:, [1]]  # Width for gripper

        return new_rot, new_trans, qpos, grasp_points

    def _load_dex_grasps(
        self,
        scene: str,
        camera: str,
        view: int,
        camera_poses: np.ndarray,
        fraction: int,
        sample_total: int,
        frac_suffix: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load dexterous hand grasp data.

        Args:
            scene: Scene identifier.
            camera: Camera type.
            view: View index.
            camera_poses: Camera pose matrices.
            fraction: Data fraction.
            sample_total: Total samples to load.
            frac_suffix: Fraction suffix for data path.

        Returns:
            Tuple of (rotations, translations, joint positions, grasp points).
        """
        robot = getattr(self._config, "robot", "leap_hand")
        grasp_dir = self._data_root / f"dex_grasps_new{frac_suffix}" / scene / robot

        grasp_files = list(grasp_dir.glob("*.npz")) if grasp_dir.exists() else []
        if len(grasp_files) == 0:
            raise ValueError(f"No grasp files found in {grasp_dir}")

        rand_idxs = np.random.randint(0, len(grasp_files), sample_total)
        samples = []
        for i, f in enumerate(grasp_files):
            num = (rand_idxs == i).sum()
            if num == 0:
                continue
            grasps = np.load(str(f))
            sel_idxs = np.random.choice(len(grasps["point"]), num, replace=True)
            samples.append({k: grasps[k][sel_idxs] for k in grasps.keys()})

        permute = np.random.permutation(sample_total)
        samples = {
            k: np.concatenate([sample[k] for sample in samples])[permute]
            for k in samples[0].keys()
        }

        rot = samples["rotation"]
        trans = samples["translation"]
        grasp_points = samples["point"]

        # Transform to camera frame
        new_rot = np.einsum("ji,njk->nik", camera_poses[view, :3, :3], rot)
        new_trans = np.einsum(
            "ji,nj->ni", camera_poses[view, :3, :3], trans - camera_poses[view, :3, 3]
        )
        grasp_points = np.einsum(
            "ba,nb->na",
            camera_poses[view, :3, :3],
            grasp_points - camera_poses[view, :3, 3],
        )

        # Stack joint positions
        qpos = np.stack([samples[j] for j in self._joint_names], axis=-1)

        return new_rot, new_trans, qpos, grasp_points

    def _match_grasps_to_cloud(
        self,
        cloud: np.ndarray,
        grasp_points: np.ndarray,
        k: int,
        max_point_dis: float,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Match grasp points to nearest cloud points using KD-tree.

        Uses scipy.spatial.cKDTree for O(M log N) complexity instead of O(M * N)
        where M = number of grasp points, N = number of cloud points.

        Args:
            cloud: (N, 3) point cloud.
            grasp_points: (M, 3) grasp center points.
            k: Number of grasps to select.
            max_point_dis: Maximum distance threshold.

        Returns:
            Tuple of (center indices array, valid grasp indices list).
        """
        # Build KD-tree for efficient nearest neighbor queries
        tree = cKDTree(cloud)

        # Query all grasp points at once - returns (distances, indices)
        distances, nearest_indices = tree.query(grasp_points, k=1)

        # Initialize centers array
        centers = np.zeros(len(grasp_points), dtype=np.float64)

        # Find valid grasps (within max_point_dis)
        valid_mask = distances <= max_point_dis
        valid_grasp_indices = np.where(valid_mask)[0]

        if len(valid_grasp_indices) == 0:
            return centers, []

        # Store the nearest cloud point index for each valid grasp
        centers[valid_grasp_indices] = nearest_indices[valid_grasp_indices]

        # Sample k indices from valid grasps (with replacement if fewer than k)
        replace = len(valid_grasp_indices) < k
        selected_indices = np.random.choice(valid_grasp_indices, k, replace=replace)

        return centers, selected_indices.tolist()

    def _augment_data(self, ret_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply random rotation augmentation.

        Args:
            ret_dict: Dictionary containing sample data.

        Returns:
            Augmented data dictionary.
        """
        cloud = ret_dict["point_clouds"]
        rot = ret_dict["rot"]
        trans = ret_dict["trans"]

        # Random rotation around z-axis
        theta = np.random.rand() * 2 * np.pi
        rotmat = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        ).astype(np.float32)

        cloud = np.einsum("ij,nj->ni", rotmat, cloud)
        trans = np.einsum("ij,nj->ni", rotmat, trans)
        rot = np.einsum("ij,njk->nik", rotmat, rot)

        ret_dict["point_clouds"] = cloud
        ret_dict["rot"] = rot
        ret_dict["trans"] = trans
        voxel_size = getattr(self._config, "voxel_size", 0.005)
        ret_dict["coors"] = cloud / voxel_size

        return ret_dict


def get_sparse_tensor(
    pc: torch.Tensor,
    voxel_size: float,
) -> Dict[str, torch.Tensor]:
    """
    Convert point cloud batch to MinkowskiEngine sparse tensor format.

    Args:
        pc: (B, N, 3) batch of point clouds.
        voxel_size: Voxel size for quantization.

    Returns:
        Dictionary containing:
            - point_clouds: Original point clouds
            - coors: Sparse coordinates
            - feats: Sparse features
            - quantize2original: Mapping from quantized to original
    """
    if ME is None:
        raise ImportError("MinkowskiEngine is required for sparse tensors")

    coors = pc / voxel_size
    feats = torch.ones_like(pc)
    coordinates_batch, features_batch = ME.utils.sparse_collate(
        [coor for coor in coors], [feat for feat in feats]
    )
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch.float(),
        features_batch,
        return_index=True,
        return_inverse=True,
    )
    return {
        "point_clouds": pc,
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original,
    }


def minkowski_collate_fn(list_data: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for MinkowskiEngine sparse tensors.

    This function properly batches point cloud data for use with
    MinkowskiEngine's sparse convolution operations.

    Args:
        list_data: List of sample dictionaries from dataset.

    Returns:
        Batched dictionary with sparse tensor format.
    """
    if ME is None:
        raise ImportError("MinkowskiEngine is required for sparse collation")

    # Collate sparse coordinates and features
    coordinates_batch, features_batch = ME.utils.sparse_collate(
        [d["coors"] for d in list_data],
        [d["feats"] for d in list_data],
    )
    coordinates_batch, features_batch, original2quantize, quantize2original = (
        ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True
        )
    )

    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "original2quantize": original2quantize,
        "quantize2original": quantize2original,
    }

    def collate_fn_(batch: List) -> Any:
        """Recursively collate batch data."""
        if type(batch[0]).__module__ == "numpy":
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key in ("coors", "feats"):
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
        return batch

    return collate_fn_(list_data)


def _dict_to_namespace(d: Dict) -> object:
    """
    Convert dictionary to namespace object for attribute access.

    Args:
        d: Dictionary to convert.

    Returns:
        Namespace-like object with attribute access.
    """

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def get(self, key, default=None):
            return getattr(self, key, default)

    return Namespace(**d)


def create_data_loaders(
    config: TrainingConfig,
    data_root: str = "data",
) -> Tuple[InfiniteLoader, List[InfiniteLoader]]:
    """
    Create training and validation data loaders.

    Args:
        config: Training configuration.
        data_root: Root directory for data.

    Returns:
        Tuple of (train_loader, list of val_loaders).
    """
    train_dataset = GraspNetDataset(
        config=config,
        split=config.train_split,
        is_train=True,
        data_root=data_root,
    )
    val_datasets = [
        GraspNetDataset(
            config=config,
            split=split,
            is_train=False,
            data_root=data_root,
        )
        for split in config.val_split
    ]

    train_loader = InfiniteLoader(
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            drop_last=True,
            num_workers=config.num_workers,
            shuffle=True,
            collate_fn=minkowski_collate_fn,
        )
    )
    val_loaders = [
        InfiniteLoader(
            DataLoader(
                dataset,
                batch_size=config.batch_size,
                drop_last=True,
                num_workers=config.num_workers,
                shuffle=True,
                collate_fn=minkowski_collate_fn,
            )
        )
        for dataset in val_datasets
    ]

    return train_loader, val_loaders
