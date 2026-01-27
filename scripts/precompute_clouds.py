#!/usr/bin/env python3
"""
Pre-compute point clouds from depth images for faster training.

This script converts all depth images in the GraspNet dataset to point clouds,
saving them as compressed numpy files. This eliminates the CPU-intensive
depth-to-point-cloud conversion during training.

Usage:
    # Pre-compute all scenes (0-99 for training)
    python scripts/precompute_clouds.py --scenes 0-99 --camera realsense

    # Pre-compute specific scenes
    python scripts/precompute_clouds.py --scenes 0,1,2,3 --camera realsense

    # Dry run to see what would be processed
    python scripts/precompute_clouds.py --scenes 0-9 --dry-run
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io as scio
from PIL import Image
from tqdm import tqdm


def parse_scenes(scenes_str: str) -> List[int]:
    """Parse scene specification string.

    Examples:
        "0-99" -> [0, 1, 2, ..., 99]
        "0,1,5,10" -> [0, 1, 5, 10]
        "0-9,20-29" -> [0, ..., 9, 20, ..., 29]
    """
    scenes = []
    for part in scenes_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            scenes.extend(range(int(start), int(end) + 1))
        else:
            scenes.append(int(part))
    return sorted(set(scenes))


def depth_to_point_cloud(
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
        (H*W, 3) point cloud coordinates as float16 to save space.
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32) / factor_depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def process_scene(
    scene_id: int,
    data_root: Path,
    output_root: Path,
    camera: str,
    use_float16: bool = True,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """
    Process all views in a scene.

    Args:
        scene_id: Scene number (0-99 for training).
        data_root: Root directory containing the data.
        output_root: Output directory for pre-computed clouds.
        camera: Camera type (e.g., 'realsense').
        use_float16: If True, save as float16 to reduce storage.
        overwrite: If True, overwrite existing files.

    Returns:
        Tuple of (processed_count, skipped_count).
    """
    scene = f"scene_{str(scene_id).zfill(4)}"
    scene_path = data_root / "scenes" / scene / camera
    output_path = output_root / scene / camera
    output_path.mkdir(parents=True, exist_ok=True)

    # Load intrinsics from first view (same for all views in a scene)
    meta = scio.loadmat(str(scene_path / "meta" / "0000.mat"))
    intrinsics = meta["intrinsic_matrix"]
    factor_depth = meta["factor_depth"].item()

    processed = 0
    skipped = 0

    for view in range(256):
        str_view = str(view).zfill(4)
        output_file = output_path / f"{str_view}.npy"

        if output_file.exists() and not overwrite:
            skipped += 1
            continue

        # Load depth image
        depth_path = scene_path / "depth_gt" / f"{str_view}.png"
        if not depth_path.exists():
            depth_path = scene_path / "depth" / f"{str_view}.png"

        if not depth_path.exists():
            print(f"Warning: Depth not found for {scene}/{str_view}")
            continue

        depth = np.array(Image.open(depth_path))
        cloud = depth_to_point_cloud(depth, intrinsics, factor_depth)

        # Save as float16 to reduce storage (~50% reduction)
        if use_float16:
            cloud = cloud.astype(np.float16)

        np.save(output_file, cloud)
        processed += 1

    return processed, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute point clouds from depth images."
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="0-99",
        help="Scene specification (e.g., '0-99', '0,1,2,3', '0-9,50-59')",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="realsense",
        help="Camera type (default: realsense)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory containing the data",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/precomputed_clouds",
        help="Output directory for pre-computed clouds",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Save as float32 instead of float16 (2x storage)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without doing it",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    scenes = parse_scenes(args.scenes)

    print(f"Pre-computing point clouds")
    print(f"  Data root: {data_root}")
    print(f"  Output root: {output_root}")
    print(f"  Camera: {args.camera}")
    print(f"  Scenes: {len(scenes)} ({min(scenes)}-{max(scenes)})")
    print(f"  Format: {'float32' if args.float32 else 'float16'}")
    print(f"  Overwrite: {args.overwrite}")
    print()

    if args.dry_run:
        total_views = len(scenes) * 256
        est_size_per_view = 720 * 1280 * 3 * (4 if args.float32 else 2)  # bytes
        est_total_size = total_views * est_size_per_view / (1024**3)  # GB
        print(f"Dry run - would process {total_views} views")
        print(f"Estimated storage: {est_total_size:.1f} GB")
        return

    total_processed = 0
    total_skipped = 0

    if args.workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_scene,
                    scene_id=scene_id,
                    data_root=data_root,
                    output_root=output_root,
                    camera=args.camera,
                    use_float16=not args.float32,
                    overwrite=args.overwrite,
                ): scene_id
                for scene_id in scenes
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing scenes"
            ):
                processed, skipped = future.result()
                total_processed += processed
                total_skipped += skipped
    else:
        # Sequential processing
        for scene_id in tqdm(scenes, desc="Processing scenes"):
            processed, skipped = process_scene(
                scene_id=scene_id,
                data_root=data_root,
                output_root=output_root,
                camera=args.camera,
                use_float16=not args.float32,
                overwrite=args.overwrite,
            )
            total_processed += processed
            total_skipped += skipped

    print()
    print(f"Done!")
    print(f"  Processed: {total_processed} views")
    print(f"  Skipped: {total_skipped} views")

    # Calculate actual storage used
    if output_root.exists():
        total_size = sum(f.stat().st_size for f in output_root.rglob("*.npy"))
        print(f"  Total storage: {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
