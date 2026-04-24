"""
Smoke test: load a bundled scene, run the refactored GraspPredictor, write HTML.

Exercises the code path we actually ship (`dexgraspnet2.inference.GraspPredictor`
+ checkpoint loading + diffusion sampling + HTML viz) with data bundled in the
repo under `examples/quick_test_data/`, so a fresh clone can verify the Docker
image works without the full dataset download.

Usage:
    python scripts/quick_test.py \\
        --ckpt_path data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth \\
        --data_root examples/quick_test_data \\
        --output_path outputs/quicktest_vis.html
"""

from __future__ import annotations

try:  # noqa: SIM105
    import isaacgym  # noqa: F401  — must precede torch
except ImportError:
    pass

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as scio
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dexgraspnet2.inference.grasp_predictor import GraspPredictor  # noqa: E402
from src.utils.pc import depth_image_to_point_cloud, get_workspace_mask  # noqa: E402
from src.utils.vis_plotly import Vis  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("quick_test")

_BANNER = "=" * 72


def _print_banner(lines):
    """Print a stdout-flushed banner so the message survives log noise."""
    print(_BANNER, flush=True)
    for line in lines:
        print(line, flush=True)
    print(_BANNER, flush=True)


def load_scene(data_root: Path, view: str, num_points: int = 40000, seed: int = 0):
    """Load one view from the bundled quick-test scene.

    Args:
        data_root: Directory containing `realsense/` with depth_gt, label_gt,
            edge_gt, meta, camera_poses.npy, cam0_wrt_table.npy.
        view: View ID, e.g. ``"0000"``.
        num_points: Number of points to sample from the workspace-masked cloud.
        seed: RNG seed for point sampling.

    Returns:
        Tuple ``(cloud, seg, edge)`` of float32 / int64 / int64 arrays.
    """
    rs = data_root / "realsense"

    depth = np.array(Image.open(rs / "depth_gt" / f"{view}.png"))
    seg = np.array(Image.open(rs / "label_gt" / f"{view}.png"))
    edge = np.array(Image.open(rs / "edge_gt" / f"{view}.png"))

    meta = scio.loadmat(rs / "meta" / f"{view}.mat")
    intrinsics = meta["intrinsic_matrix"]
    factor_depth = meta["factor_depth"]
    camera_poses = np.load(rs / "camera_poses.npy")
    align_mat = np.load(rs / "cam0_wrt_table.npy")

    cloud = depth_image_to_point_cloud(depth, intrinsics, factor_depth)
    depth_mask = depth > 0
    trans = np.dot(align_mat, camera_poses[int(view)])
    workspace_mask = get_workspace_mask(cloud, seg, trans)
    mask = depth_mask & workspace_mask
    cloud = cloud[mask]
    seg = seg[mask]
    edge = edge[mask]

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(cloud), num_points, replace=len(cloud) < num_points)
    return (
        cloud[idxs].astype(np.float32),
        seg[idxs].astype(np.int64),
        edge[idxs].astype(np.int64),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_path", type=str, required=True, help="LEAP checkpoint .pth (expects sibling config.yaml at <ckpt>/../..)")
    parser.add_argument("--data_root", type=str, default="examples/quick_test_data", help="Directory with the bundled scene (realsense/ + meshes/)")
    parser.add_argument("--view", type=str, default="0000")
    parser.add_argument("--output_path", type=str, default="outputs/quicktest_vis.html")
    parser.add_argument("--num_grasps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _print_banner([
        "  DexGraspNet 2.0 — Quick Test",
        f"  checkpoint:  {args.ckpt_path}",
        f"  data root:   {args.data_root}",
        f"  device:      {args.device}",
    ])

    try:
        data_root = Path(args.data_root).resolve()
        if not data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")

        logger.info("[1/4] Loading scene from %s (view=%s)", data_root, args.view)
        cloud, seg, edge = load_scene(data_root, args.view, seed=args.seed)
        logger.info("      point cloud: %s, objects: %s", cloud.shape, np.unique(seg).tolist())

        logger.info("[2/4] Initialising GraspPredictor on %s", args.device)
        predictor = GraspPredictor(checkpoint_path=args.ckpt_path, device=args.device)

        logger.info("[3/4] Running inference (num_grasps=%d)", args.num_grasps)
        result = predictor.predict(
            point_cloud=cloud,
            segmentation=seg,
            edge_mask=edge,
            num_grasps=args.num_grasps,
            graspness_scale=5.0,
            use_category_sampling=False,
        )
        best = result.best()
        scores = np.array([g.score for g in result.grasps])

        logger.info("[4/4] Rendering HTML")
        vis = Vis(
            robot_name="leap_hand",
            urdf_path="robot_models/urdf/leap_hand_simplified.urdf",
            meta_path="robot_models/meta/leap_hand/meta.yaml",
        )
        trans_t = torch.from_numpy(best.translation).float().unsqueeze(0)
        rot_t = torch.from_numpy(best.rotation).float().unsqueeze(0)
        qpos_t = torch.from_numpy(best.joint_angles).float().unsqueeze(0)
        plotly = (
            vis.pc_plotly(torch.from_numpy(cloud), color="blue")
            + vis.robot_plotly(trans=trans_t, rot=rot_t, qpos=qpos_t, opacity=1.0, color="violet")
        )

        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        vis.show(plotly, str(out))
        out_abs = out.resolve()
        out_size = out_abs.stat().st_size
    except Exception as exc:  # noqa: BLE001
        logger.exception("Quick test failed during execution")
        _print_banner([
            "  QUICK TEST FAILED",
            f"  error: {type(exc).__name__}: {exc}",
            "  see the traceback above for details",
        ])
        return 1

    _print_banner([
        "  QUICK TEST PASSED",
        f"  grasps generated:  {len(result.grasps)}",
        f"  score range:       [{scores.min():.3f}, {scores.max():.3f}]  (best = {best.score:.3f})",
        f"  best translation:  [{best.translation[0]:+.3f}, {best.translation[1]:+.3f}, {best.translation[2]:+.3f}] m",
        f"  output HTML:       {out_abs}  ({out_size/1024:.0f} KB)",
        "  → open the HTML file in any browser to view the predicted grasp.",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
