"""
Build a USD file visualizing a single grasp candidate, for inspection in Isaac Sim.

Use case: debug the grasp-candidate generation transform pipeline. The script
samples one candidate using the same strategy (`surface_normal` or `dome`) and
hand config the real pipeline uses, computes the pregrasp pose with the same
TCP-derived approach axis the simulator uses, and bakes a self-contained USD
stage containing:

  - The object mesh at its spawn pose in world frame.
  - A thin (2 mm diameter) cylinder placed at the sampled surface point,
    oriented along the surface normal. The cylinder is the "approach line"
    — if the hand isn't sitting on it, the transform chain is wrong.
  - The hand (URDF meshes baked via forward kinematics) at its computed
    pregrasp pose with the pregrasp finger preshape applied.
  - A small axis gizmo at the world origin for visual reference.

Open the .usd in Isaac Sim. If the hand looks off, manually move it to what
you think is right, then tell me what you changed — we compare poses and
fix the transform math.

Does not require GPU / Isaac Gym. Runs purely on CPU.
"""

from __future__ import annotations

import re

# Isaac Gym must be imported before torch — and torch is pulled in transitively
# by the `dexgraspnet2` package `__init__` (via GraspGenerator → grasp_simulator).
# The visualizer itself never *uses* Isaac Gym at runtime, but the import-order
# constraint means we have to touch it here before anything else imports torch.
# If isaacgym isn't available (e.g. running outside the sim Docker image), we
# skip it — the visualizer will then fail later on the HandConfig import, but
# the error message will be clearer about the real environment problem.
try:  # noqa: SIM105
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import trimesh
import yourdfpy
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
from scipy.spatial.transform import Rotation

from dexgraspnet2.configs.hand_config import HandConfig
from dexgraspnet2.generation.sampling_strategies import (
    DomeSamplingStrategy,
    SamplingResult,
    SurfaceNormalStrategy,
)
from dexgraspnet2.utils.scene_loader import load_scene_data

logger = logging.getLogger(__name__)

# Keep in sync with dexgraspnet2/generation/grasp_simulator.py SimulationConfig.
# Duplicated here because importing SimulationConfig pulls in isaacgym.
PREGRASP_DISTANCE = 0.10

# Default spawn pose for isolation grasp-generation. Matches
# GraspGenerator.setup_scene default (identity rotation, 5 cm above ground).
_DEFAULT_SPAWN_Z = 0.05

# Strategy name aliases.
_STRATEGY_ALIASES = {
    "normals": "surface_normal",
    "surface_normal": "surface_normal",
    "dome": "dome",
}


def _approach_axis_wrist(hand_config: HandConfig) -> np.ndarray:
    """TCP +Z expressed in the wrist frame. Mirrors GraspSimulator.__init__."""
    if hand_config.tcp_rotation_rpy is not None:
        R_wrist_tcp = Rotation.from_euler(
            "xyz", hand_config.tcp_rotation_rpy, degrees=True
        ).as_matrix()
        return R_wrist_tcp @ np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 0.0, 1.0])


def _build_strategy(hand_config: HandConfig, name: str):
    normalized = _STRATEGY_ALIASES.get(name)
    if normalized is None:
        raise ValueError(
            f"Unknown strategy '{name}'. Choose from: {sorted(_STRATEGY_ALIASES)}"
        )
    if normalized == "dome":
        return DomeSamplingStrategy(hand_config)
    return SurfaceNormalStrategy(hand_config)


def _resolve_object(
    scene_id: str, obj_id: int, data_root: str
) -> tuple[Path, np.ndarray]:
    """
    Match GraspGenerator.setup_scene: load the object's mesh path, but spawn
    it at (0, 0, 0.05) identity — the grasp-candidate generator spawns the
    object in isolation, not at its in-scene pose.
    """
    scene_data = load_scene_data(scene_id=scene_id, data_root=data_root)
    mesh_path: Optional[Path] = None
    for obj in scene_data["objects"]:
        if int(obj["id"]) == int(obj_id):
            mesh_path = Path(obj["mesh_path"])
            break
    if mesh_path is None:
        ids = [int(o["id"]) for o in scene_data["objects"]]
        raise ValueError(
            f"Object id {obj_id} not found in scene {scene_id}. "
            f"Available: {ids}"
        )

    spawn = np.eye(4)
    spawn[2, 3] = _DEFAULT_SPAWN_Z
    return mesh_path, spawn


# ---------------------------------------------------------------------------
# USD helpers
# ---------------------------------------------------------------------------


def _mat4_to_gf(mat4: np.ndarray) -> Gf.Matrix4d:
    """
    Convert numpy 4x4 (column-vector convention: p_world = M @ p_local) to
    Gf.Matrix4d (USD's row-vector convention: p_world = p_local * M). Requires
    a transpose — USD stores translation in the last row, numpy in the last
    column.
    """
    mt = np.ascontiguousarray(mat4.T, dtype=np.float64)
    return Gf.Matrix4d(
        mt[0, 0], mt[0, 1], mt[0, 2], mt[0, 3],
        mt[1, 0], mt[1, 1], mt[1, 2], mt[1, 3],
        mt[2, 0], mt[2, 1], mt[2, 2], mt[2, 3],
        mt[3, 0], mt[3, 1], mt[3, 2], mt[3, 3],
    )


def _set_xform(prim, transform_4x4: np.ndarray) -> None:
    """Apply a world-space 4x4 transform to a USD prim via a single matrix op."""
    xformable = UsdGeom.Xformable(prim)
    # Clear any pre-existing xformOps to avoid compounding.
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()
    op.Set(_mat4_to_gf(transform_4x4))


def _add_mesh_prim(
    stage: Usd.Stage,
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    transform: Optional[np.ndarray] = None,
    color: Optional[tuple] = None,
) -> UsdGeom.Mesh:
    """
    Create a UsdGeomMesh with optional world transform and solid display color.
    Assumes triangle faces.
    """
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.CreatePointsAttr(
        [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    )
    prim.CreateFaceVertexCountsAttr([3] * len(faces))
    prim.CreateFaceVertexIndicesAttr(
        [int(i) for i in np.asarray(faces).flatten()]
    )
    if transform is not None:
        _set_xform(prim.GetPrim(), transform)
    if color is not None:
        prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return prim


def _add_approach_cylinder(
    stage: Usd.Stage,
    origin: np.ndarray,
    axis: np.ndarray,
    length: float,
    radius: float = 0.001,
    path: str = "/World/ApproachAxis",
    color: tuple = (0.15, 0.85, 0.15),
) -> None:
    """
    Thin cylinder centered at `origin`, oriented so its long axis aligns with
    `axis`. Default USD cylinder axis is +Z; we compute the rotation from
    +Z → axis and apply as the prim's transform.
    """
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(length))
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    z = np.array([0.0, 0.0, 1.0])
    a = axis / (np.linalg.norm(axis) + 1e-12)

    if np.allclose(a, z):
        R = np.eye(3)
    elif np.allclose(a, -z):
        # 180-degree flip; any axis perpendicular to Z works.
        R = Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()
    else:
        rot_axis = np.cross(z, a)
        rot_axis /= np.linalg.norm(rot_axis)
        angle = float(np.arccos(np.clip(np.dot(z, a), -1.0, 1.0)))
        R = Rotation.from_rotvec(rot_axis * angle).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin
    _set_xform(cyl.GetPrim(), T)


def _add_axis_gizmo(
    stage: Usd.Stage,
    path: str,
    transform: np.ndarray,
    arm_length: float = 0.03,
    arm_radius: float = 0.0015,
) -> None:
    """
    Three small colored cylinders (X=red, Y=green, Z=blue) at `transform`, each
    oriented along its axis. Useful as a world-origin reference or as a
    stand-in for a frame we want to debug.
    """
    parent = UsdGeom.Xform.Define(stage, path)
    _set_xform(parent.GetPrim(), transform)

    axes = [
        ("X", (1.0, 0.0, 0.0), (0.95, 0.15, 0.15)),
        ("Y", (0.0, 1.0, 0.0), (0.15, 0.95, 0.15)),
        ("Z", (0.0, 0.0, 1.0), (0.15, 0.35, 0.95)),
    ]
    z_axis = np.array([0.0, 0.0, 1.0])
    for label, direction, color in axes:
        d = np.array(direction, dtype=float)
        cyl = UsdGeom.Cylinder.Define(stage, f"{path}/Axis_{label}")
        cyl.CreateRadiusAttr(arm_radius)
        cyl.CreateHeightAttr(arm_length)
        cyl.CreateAxisAttr(UsdGeom.Tokens.z)
        cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])

        # Rotate +Z -> direction; translate so the cylinder base sits at origin
        # and tip points along +direction.
        if np.allclose(d, z_axis):
            R = np.eye(3)
        else:
            axis = np.cross(z_axis, d)
            n = np.linalg.norm(axis)
            if n < 1e-9:
                R = Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()
            else:
                axis /= n
                angle = float(np.arccos(np.clip(np.dot(z_axis, d), -1.0, 1.0)))
                R = Rotation.from_rotvec(axis * angle).as_matrix()

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = d * (arm_length / 2.0)
        _set_xform(cyl.GetPrim(), T)


def _bake_hand(
    stage: Usd.Stage,
    hand_config: HandConfig,
    wrist_translation: np.ndarray,
    wrist_rotation: np.ndarray,
    pregrasp_joint_angles: np.ndarray,
    root_path: str = "/World/Hand",
) -> np.ndarray:
    """
    Load the hand's URDF, apply the pregrasp joint angles via yourdfpy FK,
    and bake each link's visual mesh into the stage as a UsdGeomMesh so the
    resulting USD is fully self-contained.

    The simulator's TCP/approach math targets `hand_config.wrist_link` — but
    the URDF root (e.g. `base`) may be offset from it by a fixed transform.
    We compose with the inverse of `root -> wrist_link` so that the wrist
    link ends up at the caller's pregrasp pose regardless of URDF root layout.
    """
    # HandConfig stores urdf_path with `../..` segments relative to the YAML
    # file location; resolve() normalizes so yourdfpy's mesh-path resolution
    # (relative to URDF directory) works reliably.
    urdf_path = Path(hand_config.urdf_path).resolve()
    urdf = yourdfpy.URDF.load(str(urdf_path))

    # Apply the pregrasp preshape to the actuated joints.
    cfg = {}
    for name, angle in zip(hand_config.joint_names, pregrasp_joint_angles):
        if name in urdf.actuated_joint_names:
            cfg[name] = float(angle)
    if cfg:
        urdf.update_cfg(cfg)

    # Target pose of the wrist link in world.
    wrist_world = np.eye(4)
    wrist_world[:3, :3] = wrist_rotation
    wrist_world[:3, 3] = wrist_translation

    # `root -> wrist` in the current URDF configuration. Inverse of this is
    # the offset we need to apply to URDF root so that wrist lands at
    # `wrist_world`. If the URDF root already IS the wrist link, this is
    # identity and the composition is a no-op.
    try:
        root_to_wrist = np.asarray(
            urdf.get_transform(frame_to=hand_config.wrist_link), dtype=float
        )
    except Exception as exc:
        logger.warning(
            f"Could not resolve transform to wrist_link "
            f"'{hand_config.wrist_link}': {exc}. Falling back to identity."
        )
        root_to_wrist = np.eye(4)

    root_pose = wrist_world @ np.linalg.inv(root_to_wrist)

    hand_xform = UsdGeom.Xform.Define(stage, root_path)
    _set_xform(hand_xform.GetPrim(), root_pose)

    # yourdfpy.URDF.scene is a trimesh.Scene where each node holds a visual
    # mesh and the graph carries the current-config FK transform from the
    # URDF root. Iterate every geometry node and bake it.
    scene = urdf.scene
    baked = 0
    for node_name in scene.graph.nodes_geometry:
        transform_from_root, geom_name = scene.graph.get(frame_to=node_name)
        if geom_name not in scene.geometry:
            continue
        geometry = scene.geometry[geom_name]

        # GLBs may load as a trimesh.Scene with multiple sub-meshes. Flatten
        # with .dump() and attach each sub-mesh at the same node transform.
        if isinstance(geometry, trimesh.Scene):
            sub_meshes = geometry.dump(concatenate=False) or []
        else:
            sub_meshes = [geometry]

        # USD prim-name grammar: [A-Za-z_][A-Za-z0-9_]*. Dots, slashes,
        # colons, hyphens, etc. all need to go. Leading digit also banned.
        safe = re.sub(r"[^A-Za-z0-9_]", "_", node_name)
        if safe and safe[0].isdigit():
            safe = "_" + safe
        for sub_idx, mesh in enumerate(sub_meshes):
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                continue
            prim_path = f"{root_path}/Link_{safe}_{sub_idx}"
            _add_mesh_prim(
                stage,
                prim_path,
                np.asarray(mesh.vertices, dtype=float),
                np.asarray(mesh.faces, dtype=int),
                transform=transform_from_root,
                color=(0.85, 0.55, 0.30),
            )
            baked += 1

    if baked == 0:
        logger.warning(
            "No hand visual meshes were baked — URDF may reference geometry "
            "types we don't handle (primitives only, or missing meshes)."
        )
    else:
        logger.info(f"Baked {baked} hand visual mesh prims.")

    # Return `root_to_wrist` so callers can place other prims as children of
    # the hand relative to its wrist link (e.g. the TCP gizmo). A child's
    # local transform = root_to_wrist @ T_wrist_frame composes with the
    # parent's `root_pose` to land it at `wrist_world @ T_wrist_frame`.
    return root_to_wrist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _default_output(scene_id: str, obj_id: int, strategy: str,
                    preshape: Optional[str], seed: int) -> Path:
    preshape_tag = preshape or "default"
    name = (
        f"{scene_id}_obj{obj_id:03d}_{strategy}_{preshape_tag}_seed{seed}.usd"
    )
    return Path(".logs") / "visualize" / name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize one grasp candidate as a USD file."
    )
    parser.add_argument(
        "--hand",
        type=str,
        default="dexgraspnet2/configs/hands/inspire_hand.yaml",
        help="Path to the hand YAML config.",
    )
    parser.add_argument("--scene_id", type=str, default="scene_0000")
    parser.add_argument("--obj_id", type=int, default=14)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Sampling strategy: normals|surface_normal|dome. "
             "Default: value from hand YAML.",
    )
    parser.add_argument(
        "--preshape",
        type=str,
        default=None,
        help="Preshape name (e.g. 'power'). Default: first preshape in config.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output USD path. Default: .logs/visualize/<auto>.usd",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # --- Load config and object ---
    hand_config = HandConfig.from_yaml(Path(args.hand))
    if not Path(hand_config.urdf_path).exists():
        logger.error(f"Hand URDF not found: {hand_config.urdf_path}")
        return 1

    mesh_path, spawn_pose = _resolve_object(
        args.scene_id, args.obj_id, args.data_root
    )
    if not mesh_path.exists():
        logger.error(f"Object mesh not found: {mesh_path}")
        return 1

    mesh = trimesh.load(str(mesh_path), force="mesh")
    sampling_mesh = mesh.copy()
    sampling_mesh.apply_transform(spawn_pose)

    # --- Choose strategy and preshape ---
    strategy_name_raw = args.strategy or hand_config.sampling_strategy
    strategy = _build_strategy(hand_config, strategy_name_raw)
    strategy_name = _STRATEGY_ALIASES[strategy_name_raw]
    logger.info(
        f"Sampling one candidate with strategy '{strategy_name}' seed={args.seed}"
    )

    results: List[SamplingResult] = strategy.sample(
        sampling_mesh, num_candidates=1, seed=args.seed
    )
    if not results:
        logger.error(
            "Sampling returned no candidates (strategy filter may have rejected "
            "the only sample). Try a different seed."
        )
        return 1
    res = results[0]

    preshapes = hand_config.preshapes or {}
    preshape_name = args.preshape
    if preshape_name is None and preshapes:
        preshape_name = next(iter(preshapes.keys()))
    if preshape_name and preshape_name in preshapes:
        pregrasp_joints = np.asarray(preshapes[preshape_name]["pregrasp"], dtype=float)
    else:
        pregrasp_joints = np.asarray(hand_config.get_open_pose(), dtype=float)
        preshape_name = preshape_name or "open"

    logger.info(f"Using preshape '{preshape_name}': {pregrasp_joints.tolist()}")

    # --- Compute pregrasp pose (same math as simulator) ---
    approach_axis_w = _approach_axis_wrist(hand_config)
    approach_world = res.rotation @ approach_axis_w
    pregrasp_translation = res.translation - approach_world * PREGRASP_DISTANCE
    pregrasp_rotation = res.rotation

    # Surface normal at the sampled point, in world frame. For
    # SurfaceNormalStrategy this is the true outward normal (TCP +Z was
    # aligned with -normal, so normal = -approach_world). For DomeSampling
    # there is no actual surface normal; we visualize the approach axis
    # (again the opposite of approach_world — the vector from sampled point
    # back toward the hand).
    normal_world = -approach_world
    cylinder_origin = np.asarray(res.point, dtype=float)
    cylinder_length = PREGRASP_DISTANCE + 0.15  # ~25 cm, spans surface → hand

    logger.info(f"Sampled point (world): {cylinder_origin.tolist()}")
    logger.info(f"Wrist translation (world): {res.translation.tolist()}")
    logger.info(
        f"Pregrasp translation (world): {pregrasp_translation.tolist()}"
    )
    logger.info(
        f"Approach axis (world, hand -> object): {approach_world.tolist()}"
    )

    # --- Resolve output path ---
    output_path = (
        Path(args.output)
        if args.output
        else _default_output(
            args.scene_id, args.obj_id, strategy_name, preshape_name, args.seed
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Build USD stage ---
    # USD requires the stage to be created; use CreateNew which truncates
    # an existing file with the same name — that's the desired iterate-fast
    # behavior (rerun with same params to regenerate after a code edit).
    if output_path.exists():
        output_path.unlink()
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # World-origin axis gizmo (small, offset upward slightly to be visible).
    _add_axis_gizmo(stage, "/World/OriginGizmo", np.eye(4))

    # Object mesh — already-transformed vertices (spawn_pose baked in) so no
    # prim-level transform is needed.
    object_mesh_world = mesh.copy()
    object_mesh_world.apply_transform(spawn_pose)
    _add_mesh_prim(
        stage,
        "/World/Object",
        np.asarray(object_mesh_world.vertices, dtype=float),
        np.asarray(object_mesh_world.faces, dtype=int),
        color=(0.55, 0.58, 0.70),
    )

    # Approach-axis cylinder at sampled point.
    _add_approach_cylinder(
        stage,
        origin=cylinder_origin,
        axis=normal_world,
        length=cylinder_length,
        radius=0.001,  # 2 mm diameter
    )

    # Marker at hand pregrasp pose (coordinate axes).
    pregrasp_T = np.eye(4)
    pregrasp_T[:3, :3] = pregrasp_rotation
    pregrasp_T[:3, 3] = pregrasp_translation
    _add_axis_gizmo(
        stage,
        "/World/PregraspGizmo",
        pregrasp_T,
        arm_length=0.04,
        arm_radius=0.0018,
    )

    # Build the TCP transform in wrist frame from the YAML. Used below as a
    # CHILD of the hand so the gizmo moves with it when you manipulate the
    # hand in Isaac Sim.
    T_wrist_tcp = np.eye(4)
    if hand_config.tcp_rotation_rpy is not None:
        T_wrist_tcp[:3, :3] = Rotation.from_euler(
            "xyz", hand_config.tcp_rotation_rpy, degrees=True
        ).as_matrix()
    if hand_config.tcp_position is not None:
        T_wrist_tcp[:3, 3] = np.asarray(hand_config.tcp_position, dtype=float)
    tcp_world_for_log = pregrasp_T @ T_wrist_tcp
    logger.info(
        f"TCP translation (world): {tcp_world_for_log[:3, 3].tolist()}"
    )

    # Hand baked via URDF FK. Returns the URDF root→wrist transform so we can
    # parent the TCP gizmo under the hand and still have it land at the right
    # world pose.
    root_to_wrist = _bake_hand(
        stage,
        hand_config,
        wrist_translation=pregrasp_translation,
        wrist_rotation=pregrasp_rotation,
        pregrasp_joint_angles=pregrasp_joints,
    )

    # TCP gizmo — placed as a CHILD of /World/Hand so moving the hand in
    # Isaac Sim carries the TCP with it (iteration-friendly when tuning
    # tcp_position / tcp_rotation_rpy against the visible URDF geometry).
    # World pose decomposition:
    #   /World/Hand xform           = wrist_world @ inv(root_to_wrist)
    #   /World/Hand/TcpGizmo local  = root_to_wrist @ T_wrist_tcp
    #   composed world pose         = wrist_world @ T_wrist_tcp   ✓
    tcp_local = root_to_wrist @ T_wrist_tcp
    _add_axis_gizmo(
        stage,
        "/World/Hand/TcpGizmo",
        tcp_local,
        arm_length=0.03,
        arm_radius=0.0015,
    )

    # Lighting.
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(3000.0)

    stage.GetRootLayer().Save()
    logger.info(f"Wrote USD: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
