#!/usr/bin/env python3
"""
Test script for validating the DexGraspNet2 inference pipeline.

This script tests the production inference implementation by:
1. Loading the pre-trained LEAP hand checkpoint
2. Running inference on a test scene
3. Comparing results to the original implementation
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_scene(scene_id: str, view_id: str, camera: str = "realsense") -> dict:
    """Load a test scene from the dataset."""
    from PIL import Image
    import scipy.io as scio
    from src.utils.pc import depth_image_to_point_cloud, get_workspace_mask

    path = Path("data") / "scenes" / scene_id / camera

    # Load depth and segmentation
    depth = np.array(Image.open(path / "depth_gt" / f"{view_id}.png"))
    seg = np.array(Image.open(path / "label_gt" / f"{view_id}.png"))

    # Load camera parameters
    meta = scio.loadmat(path / "meta" / f"{view_id}.mat")
    intrinsics = meta["intrinsic_matrix"]
    factor_depth = meta["factor_depth"]
    camera_poses = np.load(path / "camera_poses.npy")
    align_mat = np.load(path / "cam0_wrt_table.npy")

    # Convert to point cloud
    cloud = depth_image_to_point_cloud(depth, intrinsics, factor_depth)
    depth_mask = depth > 0
    trans = np.dot(align_mat, camera_poses[int(view_id)])
    workspace_mask = get_workspace_mask(cloud, seg, trans)

    mask = depth_mask & workspace_mask
    cloud = cloud[mask]
    seg = seg[mask]

    # Sample points
    num_points = 40000
    if len(cloud) > num_points:
        idxs = np.random.choice(len(cloud), num_points, replace=False)
    else:
        idxs = np.random.choice(len(cloud), num_points, replace=True)

    cloud = cloud[idxs]
    seg = seg[idxs]

    return {
        "point_cloud": cloud.astype(np.float32),
        "segmentation": seg.astype(np.int64),
        "extrinsics": trans,
    }


def test_original_implementation(
    checkpoint_path: str,
    scene_data: dict,
    num_grasps: int = 100,
) -> dict:
    """Run inference using the original implementation."""
    from src.utils.config import ckpt_to_config
    from src.network.model import get_model
    from src.utils.dataset import get_sparse_tensor

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    config = ckpt_to_config(checkpoint_path)
    model = get_model(config.model)
    model.config.voxel_size = config.data.voxel_size
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()

    # Prepare data
    pc = scene_data["point_cloud"]
    seg = scene_data["segmentation"]

    with torch.no_grad():
        data = get_sparse_tensor(torch.tensor(pc[None]).float(), config.data.voxel_size)
        data["seg"] = torch.tensor(seg[None]).long()
        data = {k: v.to(device) for k, v in data.items()}

        outputs = model.sample(
            data, num_grasps,
            graspness_scale=5,
            allow_fail=True,
            cate=False,
            with_point=True,
            with_score_parts=True,
        )

        rotation, translation, joints, score, obj_indices, graspness, log_prob, seed_points = outputs

    return {
        "rotation": rotation.cpu().numpy(),
        "translation": translation.cpu().numpy(),
        "joints": joints.cpu().numpy(),
        "score": score.cpu().numpy(),
        "graspness": graspness.cpu().numpy(),
        "log_prob": log_prob.cpu().numpy(),
    }


def test_new_implementation(
    checkpoint_path: str,
    scene_data: dict,
    num_grasps: int = 100,
) -> dict:
    """Run inference using the new production implementation."""
    from dexgraspnet2 import GraspPredictor, HandConfig

    hand_config = HandConfig.leap_hand(base_path=project_root / "robot_models")

    predictor = GraspPredictor(
        checkpoint_path=checkpoint_path,
        hand_config=hand_config,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    result = predictor.predict(
        point_cloud=scene_data["point_cloud"],
        segmentation=scene_data["segmentation"],
        num_grasps=num_grasps,
        graspness_scale=5.0,
        use_category_sampling=False,
    )

    return {
        "rotation": np.stack([g.rotation for g in result.grasps]),
        "translation": np.stack([g.translation for g in result.grasps]),
        "joints": np.stack([g.joint_angles for g in result.grasps]),
        "score": np.array([g.score for g in result.grasps]),
        "graspness": np.array([g.graspness for g in result.grasps]),
        "log_prob": np.array([g.log_prob for g in result.grasps]),
        "result": result,
    }


def compare_results(orig: dict, new: dict) -> bool:
    """Compare results from original and new implementations."""
    logger.info("Comparing results...")

    all_passed = True

    # Compare shapes (accounting for batch dimension differences)
    for key in ["rotation", "translation", "joints", "score", "graspness", "log_prob"]:
        if key not in orig or key not in new:
            logger.warning(f"Key '{key}' missing in one of the results")
            continue

        orig_data = orig[key].squeeze()  # Remove batch dim
        new_data = new[key].squeeze()

        if orig_data.shape != new_data.shape:
            logger.warning(f"{key}: shape mismatch - orig={orig_data.shape}, new={new_data.shape}")
            all_passed = False
        else:
            logger.info(f"{key}: shape OK ({orig_data.shape})")

    # Check score distribution (should be similar statistically)
    orig_scores = orig["score"].flatten()
    new_scores = new["score"].flatten()

    logger.info(f"Original scores: mean={orig_scores.mean():.4f}, std={orig_scores.std():.4f}")
    logger.info(f"New scores: mean={new_scores.mean():.4f}, std={new_scores.std():.4f}")

    # Check top-1 score
    orig_best = orig_scores.max()
    new_best = new_scores.max()
    logger.info(f"Best score: orig={orig_best:.4f}, new={new_best:.4f}")

    # The implementations should produce similar score ranges
    score_mean_diff = abs(orig_scores.mean() - new_scores.mean())
    if score_mean_diff > 5.0:
        logger.warning(f"Score mean difference too large: {score_mean_diff:.4f}")
        all_passed = False
    else:
        logger.info(f"Score means are within tolerance (diff={score_mean_diff:.4f})")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test DexGraspNet2 inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="scene_0100",
        help="Scene ID to test",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="0000",
        help="View ID to test",
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=100,
        help="Number of grasps to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info(f"Testing with checkpoint: {args.checkpoint}")
    logger.info(f"Scene: {args.scene}, View: {args.view}")

    # Check if data exists
    data_path = Path("data") / "scenes" / args.scene
    if not data_path.exists():
        # Try symlink from mounted data
        mounted_path = Path("/mnt/datasets/DexGraspNet2.0-data/scenes") / args.scene
        if mounted_path.exists():
            logger.info(f"Creating symlink from {mounted_path}")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            if not data_path.exists():
                os.symlink(mounted_path, data_path)
        else:
            logger.error(f"Scene data not found at {data_path} or {mounted_path}")
            return 1

    # Load test scene
    logger.info("Loading test scene...")
    try:
        scene_data = load_test_scene(args.scene, args.view)
    except Exception as e:
        logger.error(f"Failed to load scene: {e}")
        return 1

    logger.info(f"Point cloud shape: {scene_data['point_cloud'].shape}")
    logger.info(f"Unique objects: {np.unique(scene_data['segmentation'])}")

    # Test original implementation
    logger.info("Testing original implementation...")
    try:
        orig_results = test_original_implementation(
            args.checkpoint,
            scene_data,
            args.num_grasps,
        )
        logger.info("Original implementation: OK")
    except Exception as e:
        logger.error(f"Original implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test new implementation
    logger.info("Testing new implementation...")
    try:
        new_results = test_new_implementation(
            args.checkpoint,
            scene_data,
            args.num_grasps,
        )
        logger.info("New implementation: OK")
    except Exception as e:
        logger.error(f"New implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Compare results
    if compare_results(orig_results, new_results):
        logger.info("All tests passed!")
        return 0
    else:
        logger.warning("Some tests failed - check output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
