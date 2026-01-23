#!/usr/bin/env python3
"""
Visualize grasp predictions and graspness heatmap.

Outputs:
- grasp_predictions.html: Top grasp predictions with hand model
- graspness_heatmap.html: Point cloud colored by graspness score
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
import plotly.graph_objects as go
import plotly.express as px

from src.utils.vis_plotly import Vis
from src.utils.config import ckpt_to_config
from src.network.model import get_model
from src.utils.dataset import get_sparse_tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_scene(scene_id: str, view_id: str, camera: str = "realsense") -> dict:
    """Load a scene from the dataset."""
    from PIL import Image
    import scipy.io as scio
    from src.utils.pc import depth_image_to_point_cloud, get_workspace_mask

    path = Path("data") / "scenes" / scene_id / camera

    # Load depth and segmentation
    depth = np.array(Image.open(path / "depth_gt" / f"{view_id}.png"))
    seg = np.array(Image.open(path / "label_gt" / f"{view_id}.png"))

    # Load edge if available
    edge_path = path / "edge_gt" / f"{view_id}.png"
    if edge_path.exists():
        edge = np.array(Image.open(edge_path))
    else:
        edge = np.zeros_like(depth)

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
    edge = edge[mask]

    # Sample points
    num_points = 40000
    if len(cloud) > num_points:
        idxs = np.random.choice(len(cloud), num_points, replace=False)
    else:
        idxs = np.random.choice(len(cloud), num_points, replace=True)

    cloud = cloud[idxs]
    seg = seg[idxs]
    edge = edge[idxs]

    return {
        "point_cloud": cloud.astype(np.float32),
        "segmentation": seg.astype(np.int64),
        "edge": edge.astype(np.int64),
        "extrinsics": trans,
    }


def run_inference(checkpoint_path: str, scene_data: dict, num_grasps: int = 10, device: str = "cuda:0", single_object: bool = True):
    """Run inference and return predictions with graspness.

    Args:
        checkpoint_path: Path to model checkpoint.
        scene_data: Dict with point_cloud, segmentation, edge, extrinsics.
        num_grasps: Number of grasps to sample.
        device: Torch device.
        single_object: If True, focus on most graspable object. If False, distribute across all objects.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

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
    edge = scene_data["edge"]

    with torch.no_grad():
        data = get_sparse_tensor(torch.tensor(pc[None]).float(), config.data.voxel_size)
        data["seg"] = torch.tensor(seg[None]).long()
        edge_tensor = torch.tensor(edge[None]).to(device)
        data = {k: v.to(device) for k, v in data.items()}

        # Get features and graspness
        feature = model.get_feature(data)
        objectness, graspness = model.pred_score(feature)

        # Get graspness map for visualization
        graspness_map = graspness[0].cpu().numpy()
        objectness_map = objectness[0].argmax(dim=-1).cpu().numpy()

        # Sample grasps
        # single_object=True -> cate=False (focus on most graspable region)
        # single_object=False -> cate=True (distribute across all objects)
        outputs = model.sample(
            data, num_grasps,
            edge=edge_tensor,
            graspness_scale=5,
            allow_fail=True,
            cate=not single_object,
            with_point=True,
        )

        rotation, translation, joints, score, obj_indices, seed_points = outputs

    return {
        "rotation": rotation[0].cpu().numpy(),
        "translation": translation[0].cpu().numpy(),
        "joints": joints[0].cpu().numpy(),
        "score": score[0].cpu().numpy(),
        "seed_points": seed_points.cpu().numpy(),
        "graspness_map": graspness_map,
        "objectness_map": objectness_map,
    }


def create_graspness_heatmap(
    point_cloud: np.ndarray,
    graspness: np.ndarray,
    objectness: np.ndarray,
    output_path: str,
):
    """Create graspness heatmap visualization."""
    logger.info("Creating graspness heatmap...")

    # Mask out background points (objectness == 0)
    object_mask = objectness == 1

    # Normalize graspness for visualization
    graspness_viz = graspness.copy()
    graspness_viz[~object_mask] = 0  # Set background to 0

    # Create plotly figure
    fig = go.Figure()

    # Add point cloud with graspness coloring
    fig.add_trace(go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=graspness_viz,
            colorscale='Viridis',
            colorbar=dict(
                title="Graspness",
                thickness=20,
            ),
            showscale=True,
        ),
        name='Point Cloud',
        hovertemplate=(
            'x: %{x:.3f}<br>'
            'y: %{y:.3f}<br>'
            'z: %{z:.3f}<br>'
            'graspness: %{marker.color:.3f}<extra></extra>'
        ),
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text="Graspness Heatmap",
            x=0.5,
            font=dict(size=20),
        ),
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        width=1200,
        height=800,
    )

    # Save to HTML
    fig.write_html(output_path)
    logger.info(f"Saved graspness heatmap to {output_path}")


def create_graspness_heatmap_all(
    point_cloud: np.ndarray,
    graspness: np.ndarray,
    output_path: str,
):
    """Create graspness heatmap for all points (no objectness filtering)."""
    logger.info("Creating graspness heatmap (all points)...")

    # Create plotly figure
    fig = go.Figure()

    # Add point cloud with graspness coloring (no masking)
    fig.add_trace(go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=graspness,
            colorscale='Viridis',
            colorbar=dict(
                title="Graspness",
                thickness=20,
            ),
            showscale=True,
        ),
        name='Point Cloud',
        hovertemplate=(
            'x: %{x:.3f}<br>'
            'y: %{y:.3f}<br>'
            'z: %{z:.3f}<br>'
            'graspness: %{marker.color:.3f}<extra></extra>'
        ),
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text="Graspness Heatmap (All Points)",
            x=0.5,
            font=dict(size=20),
        ),
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        width=1200,
        height=800,
    )

    # Save to HTML
    fig.write_html(output_path)
    logger.info(f"Saved graspness heatmap (all points) to {output_path}")


def create_grasp_visualization(
    vis: Vis,
    point_cloud: np.ndarray,
    segmentation: np.ndarray,
    predictions: dict,
    num_grasps: int,
    output_path: str,
):
    """Create grasp predictions visualization with hand models."""
    logger.info(f"Creating grasp visualization with top {num_grasps} grasps...")

    # Sort by score and get top grasps
    scores = predictions["score"]
    sorted_indices = np.argsort(scores)[::-1][:num_grasps]

    plotly_data = []

    # Add point cloud colored by segmentation
    unique_segs = np.unique(segmentation)
    colors = px.colors.qualitative.Dark24

    for i, seg_id in enumerate(unique_segs):
        mask = segmentation == seg_id
        color = colors[i % len(colors)]

        plotly_data.append(go.Scatter3d(
            x=point_cloud[mask, 0],
            y=point_cloud[mask, 1],
            z=point_cloud[mask, 2],
            mode='markers',
            marker=dict(size=1.5, color=color),
            name=f'Object {seg_id}' if seg_id > 0 else 'Background',
            showlegend=True,
        ))

    # Add hand visualizations for top grasps
    grasp_colors = px.colors.sequential.Plasma

    for rank, idx in enumerate(sorted_indices):
        trans = predictions["translation"][idx]
        rot = predictions["rotation"][idx]
        joints = predictions["joints"][idx]
        score = predictions["score"][idx]

        # Get color based on rank
        color = grasp_colors[rank % len(grasp_colors)]

        # Add hand mesh
        hand_plotly = vis.robot_plotly(
            trans=torch.tensor(trans[None]),
            rot=torch.tensor(rot[None]),
            qpos=torch.tensor(joints[None]),
            opacity=0.8,
            color=color,
            mesh_type='visual',
        )

        # Add grasp info to hover
        for trace in hand_plotly:
            trace.name = f'Grasp {rank+1} (score={score:.2f})'
            trace.showlegend = True

        plotly_data.extend(hand_plotly)

        # Add seed point marker
        if idx < len(predictions["seed_points"]):
            seed = predictions["seed_points"][idx]
            plotly_data.append(go.Scatter3d(
                x=[seed[0]],
                y=[seed[1]],
                z=[seed[2]],
                mode='markers',
                marker=dict(size=8, color='red', symbol='diamond'),
                name=f'Seed {rank+1}',
                showlegend=False,
            ))

    # Create figure
    fig = go.Figure(data=plotly_data)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Top {num_grasps} Grasp Predictions",
            x=0.5,
            font=dict(size=20),
        ),
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        width=1400,
        height=900,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    # Save to HTML
    fig.write_html(output_path)
    logger.info(f"Saved grasp visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize grasp predictions")
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
        help="Scene ID",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="0000",
        help="View ID",
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=5,
        help="Number of top grasps to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for HTML files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--single-object",
        action="store_true",
        help="Focus grasps on single most graspable object. If False, distribute across all objects.",
    )
    parser.add_argument(
        "--show-all-graspness",
        action="store_true",
        help="Output additional heatmap showing graspness for all points (without objectness filtering).",
    )
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading scene {args.scene}, view {args.view}...")
    scene_data = load_scene(args.scene, args.view)
    logger.info(f"Point cloud shape: {scene_data['point_cloud'].shape}")

    logger.info("Running inference...")
    predictions = run_inference(
        args.checkpoint,
        scene_data,
        num_grasps=args.num_grasps * 10,  # Sample more, show top N
        single_object=args.single_object,
    )
    logger.info(f"Generated {len(predictions['score'])} grasp predictions")

    # Create visualizations
    graspness_path = output_dir / "graspness_heatmap.html"
    create_graspness_heatmap(
        scene_data["point_cloud"],
        predictions["graspness_map"],
        predictions["objectness_map"],
        str(graspness_path),
    )

    # Create all-points heatmap if requested
    if args.show_all_graspness:
        graspness_all_path = output_dir / "graspness_heatmap_all.html"
        create_graspness_heatmap_all(
            scene_data["point_cloud"],
            predictions["graspness_map"],
            str(graspness_all_path),
        )

    # Create grasp visualization
    vis = Vis(
        robot_name="leap_hand",
        urdf_path="robot_models/urdf/leap_hand_simplified.urdf",
        meta_path="robot_models/meta/leap_hand/meta.yaml",
    )

    grasp_path = output_dir / "grasp_predictions.html"
    create_grasp_visualization(
        vis,
        scene_data["point_cloud"],
        scene_data["segmentation"],
        predictions,
        args.num_grasps,
        str(grasp_path),
    )

    logger.info("Done!")
    logger.info("Output files:")
    logger.info(f"  - {graspness_path}")
    if args.show_all_graspness:
        logger.info(f"  - {graspness_all_path}")
    logger.info(f"  - {grasp_path}")


if __name__ == "__main__":
    main()
