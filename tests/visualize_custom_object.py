#!/usr/bin/env python3
"""
Visualize grasp predictions for a custom object mesh.

Loads a mesh file, samples a point cloud, and runs graspness prediction.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np
import torch
import trimesh
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


def load_mesh_as_pointcloud(mesh_path: str, num_points: int = 40000, scale: float = 1.0) -> np.ndarray:
    """Load mesh and sample point cloud from surface.

    Args:
        mesh_path: Path to mesh file (.obj, .stl, .ply).
        num_points: Number of points to sample.
        scale: Scale factor for the mesh (e.g., 0.001 to convert mm to m).

    Returns:
        Point cloud as (N, 3) numpy array.
    """
    logger.info(f"Loading mesh from {mesh_path}")
    mesh = trimesh.load(mesh_path)

    if isinstance(mesh, trimesh.Scene):
        # Combine all geometries in scene
        mesh = trimesh.util.concatenate(mesh.dump())

    # Apply scale
    mesh.apply_scale(scale)

    # Sample points from surface
    points, _ = trimesh.sample.sample_surface(mesh, num_points)

    # Center the point cloud
    centroid = points.mean(axis=0)
    points = points - centroid

    # Place on table (z=0 is table surface)
    points[:, 2] -= points[:, 2].min()
    points[:, 2] += 0.01  # Small offset above table

    logger.info(f"Sampled {len(points)} points, bounds: {points.min(axis=0)} to {points.max(axis=0)}")

    return points.astype(np.float32)


def run_graspness_prediction(checkpoint_path: str, point_cloud: np.ndarray, device: str = "cuda:0", num_grasps: int = 10):
    """Run graspness and grasp pose prediction on point cloud.

    Args:
        checkpoint_path: Path to model checkpoint.
        point_cloud: Point cloud as (N, 3) array.
        device: Torch device.
        num_grasps: Number of grasps to sample.

    Returns:
        Dict with graspness_map, objectness_map, and grasp predictions.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    config = ckpt_to_config(checkpoint_path)
    model = get_model(config.model)
    model.config.voxel_size = config.data.voxel_size
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()

    # Prepare data - create dummy segmentation (all points = object 1)
    seg = np.ones(len(point_cloud), dtype=np.int64)

    with torch.no_grad():
        data = get_sparse_tensor(torch.tensor(point_cloud[None]).float(), config.data.voxel_size)
        data["seg"] = torch.tensor(seg[None]).long()
        data = {k: v.to(device) for k, v in data.items()}

        # Get features and graspness
        feature = model.get_feature(data)
        objectness, graspness = model.pred_score(feature)

        graspness_map = graspness[0].cpu().numpy()
        objectness_map = objectness[0].argmax(dim=-1).cpu().numpy()

        # Sample grasp poses
        outputs = model.sample(
            data, num_grasps,
            graspness_scale=5,
            allow_fail=True,
            cate=False,  # Single object
            with_point=True,
        )

        rotation, translation, joints, score, obj_indices, seed_points = outputs

    return {
        "graspness_map": graspness_map,
        "objectness_map": objectness_map,
        "rotation": rotation[0].cpu().numpy(),
        "translation": translation[0].cpu().numpy(),
        "joints": joints[0].cpu().numpy(),
        "score": score[0].cpu().numpy(),
        "seed_points": seed_points.cpu().numpy(),
    }


def create_visualization(
    point_cloud: np.ndarray,
    graspness: np.ndarray,
    objectness: np.ndarray,
    output_path: str,
    title: str = "Graspness Heatmap",
):
    """Create interactive 3D visualization."""
    logger.info("Creating visualization...")

    # Stats
    obj_mask = objectness == 1
    logger.info(f"Objectness=1: {obj_mask.sum()}/{len(objectness)} ({100*obj_mask.mean():.1f}%)")
    logger.info(f"Graspness range: [{graspness.min():.3f}, {graspness.max():.3f}]")
    logger.info(f"Graspness on objects: [{graspness[obj_mask].min():.3f}, {graspness[obj_mask].max():.3f}]" if obj_mask.any() else "No object points")

    fig = go.Figure()

    # Add point cloud with graspness coloring
    fig.add_trace(go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=graspness,
            colorscale='Viridis',
            colorbar=dict(title="Graspness", thickness=20),
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

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        width=1200,
        height=800,
    )

    fig.write_html(output_path)
    logger.info(f"Saved visualization to {output_path}")


def create_grasp_visualization(
    vis: Vis,
    point_cloud: np.ndarray,
    predictions: dict,
    num_grasps: int,
    output_path: str,
    title: str = "Grasp Predictions",
):
    """Create grasp predictions visualization with hand models."""
    logger.info(f"Creating grasp visualization with top {num_grasps} grasps...")

    # Sort by score and get top grasps
    scores = predictions["score"]
    sorted_indices = np.argsort(scores)[::-1][:num_grasps]

    plotly_data = []

    # Add point cloud
    plotly_data.append(go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(size=1.5, color='lightblue'),
        name='Object',
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

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        width=1400,
        height=900,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.write_html(output_path)
    logger.info(f"Saved grasp visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize graspness for custom object")
    parser.add_argument(
        "--mesh",
        type=str,
        default="workplan/bowl/bowl.obj",
        help="Path to mesh file (.obj, .stl, .ply)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=40000,
        help="Number of points to sample from mesh",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for mesh (e.g., 0.001 to convert mm to m)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for HTML files",
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=5,
        help="Number of top grasps to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = Path(args.mesh).stem

    # Load mesh as point cloud
    point_cloud = load_mesh_as_pointcloud(
        args.mesh,
        num_points=args.num_points,
        scale=args.scale,
    )

    # Run prediction (sample more grasps, show top N)
    logger.info("Running graspness and grasp prediction...")
    predictions = run_graspness_prediction(
        args.checkpoint,
        point_cloud,
        num_grasps=args.num_grasps * 10,
    )
    logger.info(f"Generated {len(predictions['score'])} grasp candidates")

    # Create graspness heatmap
    graspness_path = output_dir / f"{mesh_name}_graspness.html"
    create_visualization(
        point_cloud,
        predictions["graspness_map"],
        predictions["objectness_map"],
        str(graspness_path),
        title=f"Graspness Heatmap - {mesh_name}",
    )

    # Create grasp visualization with hand models
    vis = Vis(
        robot_name="leap_hand",
        urdf_path="robot_models/urdf/leap_hand_simplified.urdf",
        meta_path="robot_models/meta/leap_hand/meta.yaml",
    )

    grasp_path = output_dir / f"{mesh_name}_grasps.html"
    create_grasp_visualization(
        vis,
        point_cloud,
        predictions,
        args.num_grasps,
        str(grasp_path),
        title=f"Top {args.num_grasps} Grasp Predictions - {mesh_name}",
    )

    logger.info("Done!")
    logger.info("Output files:")
    logger.info(f"  - {graspness_path}")
    logger.info(f"  - {grasp_path}")


if __name__ == "__main__":
    main()
