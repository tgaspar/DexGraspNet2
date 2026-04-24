"""
FastAPI grasp-prediction server.

Single responsibility: expose a pretrained `GraspPredictor` checkpoint over HTTP
per the contract in `docs/api/predict.md`. The server is configured at startup
for ONE hand family (gripper, LEAP, Inspire, …) and does not transform frames
— all grasps are returned in the same coordinate frame as the input point cloud.

Typical usage inside the container:
    python scripts/serve_grasp_predictor.py \\
        --checkpoint data/DexGraspNet2.0-ckpts/OURS_gripper/ckpt/ckpt_50000.pth \\
        --hand-config dexgraspnet2/configs/hands/gripper.yaml \\
        --host 0.0.0.0 --port 8000

See `docs/api/predict.md` for the full request/response schema.
"""

from __future__ import annotations

# Isaac Gym must be imported before torch if any part of the dexgraspnet2
# package pulls it transitively (grasp_generator → grasp_simulator → isaacgym).
# The server itself never uses it.
try:  # noqa: SIM105
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse
import base64
import datetime as _dt
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation

from dexgraspnet2.configs.hand_config import HandConfig
from dexgraspnet2.data.grasp_result import GraspPose
from dexgraspnet2.inference.grasp_predictor import GraspPredictor

# ---------------------------------------------------------------------------
# Constants / version
# ---------------------------------------------------------------------------

SERVER_VERSION = "0.1.0"
API_VERSION = "0.1.0"
BUILT_AT = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

_ALLOWED_PC_DTYPES = ("float32", "float64")
_ALLOWED_MASK_DTYPES = ("int32", "int64", "uint8", "uint32")  # unused at the
#                                                              request level
#                                                              but kept here
#                                                              in case the
#                                                              contract adds
#                                                              masks back.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("grasp_predictor_server")

# ---------------------------------------------------------------------------
# Pydantic schemas (mirror docs/api/predict.md)
# ---------------------------------------------------------------------------


class EncodedArray(BaseModel):
    dtype: str
    shape: List[int]
    data_b64: str


class PredictRequest(BaseModel):
    point_cloud: EncodedArray
    scene_points: Optional[EncodedArray] = None
    num_grasps: int = Field(default=20, ge=1)
    min_score: float = 0.0
    # Per-request override for category-balanced seed sampling. None means
    # "use the server's startup default (--category-sampling / not)". True
    # forces uniform sampling across segmentation instances (so you get target
    # AND scene grasps in roughly balanced counts). False forces global
    # sampling weighted by graspness (scene typically dominates). See
    # docs/api/predict.md for the full discussion.
    category_sampling: Optional[bool] = None


# ---------------------------------------------------------------------------
# Server state (filled in by `_build_app`)
# ---------------------------------------------------------------------------


class ServerState:
    """Holds everything loaded at startup. One instance per process."""

    predictor: Optional[GraspPredictor] = None
    hand_config: Optional[HandConfig] = None
    hand_family: str = ""             # "gripper" or "dex"
    grasp_reference: str = ""         # human-readable
    checkpoint_name: str = ""
    model_name: str = ""
    min_points: int = 128
    max_points: int = 65536
    approach_axis_wrist: np.ndarray = np.array([0.0, 0.0, 1.0])
    loaded: bool = False
    model_commit: str = ""
    default_category_sampling: bool = False

    # Debug / observability.
    log_dir: Optional[Path] = None
    requests_jsonl: Optional[Path] = None
    dump_html: bool = False
    dump_usd: bool = False
    dump_npz: bool = False
    top_k_dump: int = 10
    vis: Any = None  # src.utils.vis_plotly.Vis, lazily constructed


STATE = ServerState()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_array(
    enc: EncodedArray,
    allowed_dtypes: tuple,
) -> np.ndarray:
    """
    Decode an EncodedArray payload into a numpy array. Raises HTTPException
    with 400 on any validation failure.
    """
    if enc.dtype not in allowed_dtypes:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "bad_dtype",
                "message": f"dtype must be one of {allowed_dtypes}, got {enc.dtype}",
            },
        )
    try:
        raw = base64.b64decode(enc.data_b64, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_base64", "message": str(exc)},
        )
    try:
        arr = np.frombuffer(raw, dtype=np.dtype(enc.dtype)).reshape(enc.shape)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "shape_dtype_mismatch",
                "message": f"decoded bytes do not match shape={enc.shape} dtype={enc.dtype}: {exc}",
            },
        )
    # Return a writable copy — `np.frombuffer` yields a read-only view which
    # downstream code (e.g. voxelizer) may dislike.
    return np.array(arr, copy=True)


def _get_vis():
    """
    Lazily build a `Vis` (Plotly visualizer) instance for the configured hand.
    Returns None if construction fails — visualization should never hard-break
    the serve loop.
    """
    if STATE.vis is not None:
        return STATE.vis
    try:
        from src.utils.vis_plotly import Vis
        kwargs: Dict[str, Any] = {"robot_name": STATE.hand_config.name}
        # Vis gripper path doesn't read the URDF; dex path does. Pass paths
        # along when we have them so dex-hand rendering works.
        if STATE.hand_family != "gripper":
            kwargs["urdf_path"] = str(STATE.hand_config.urdf_path)
            kwargs["meta_path"] = str(STATE.hand_config.meta_path)
        STATE.vis = Vis(**kwargs)
    except Exception as exc:
        logger.warning(f"Failed to build Vis for HTML dumps: {exc}")
        STATE.vis = False   # sentinel: tried and failed, don't retry
    return STATE.vis or None


def _dump_html(
    request_id: str,
    point_cloud: np.ndarray,
    grasps: List[GraspPose],
    scene_points: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """
    Write a Plotly HTML visualization of the point cloud + top-K grasp poses
    to `<log_dir>/predictions/<request_id>.html`. Only the first
    `STATE.top_k_dump` grasps are rendered to keep the HTML small and fast.
    If scene_points was supplied, it is rendered alongside the target cloud
    in a distinct color so the clutter context is visible.
    Returns the file path on success, None on failure.
    """
    if STATE.log_dir is None:
        return None
    vis = _get_vis()
    if vis is None:
        return None

    try:
        from src.utils.robot_info import GRIPPER_NEW_DEPTH
    except Exception:
        GRIPPER_NEW_DEPTH = 0.04

    try:
        pc_t = torch.as_tensor(point_cloud, dtype=torch.float32)
        pc_traces = vis.pc_plotly(pc_t)

        # Add the clutter (non-target scene) points if present.
        if scene_points is not None and scene_points.size > 0:
            try:
                scene_t = torch.as_tensor(scene_points, dtype=torch.float32)
                # `color` kwarg is accepted by Vis.pc_plotly in the paper's
                # codebase; if it's ever removed, the call still works
                # without the highlight.
                pc_traces = pc_traces + vis.pc_plotly(scene_t, color="#444")
            except Exception as exc:
                logger.debug(f"Scene-cloud trace failed (non-fatal): {exc}")

        # Color convention matches the USD dump: target grasps in a warm
        # family, clutter grasps in cool. Makes the two visually distinct
        # even when the viewer is toggling the clouds on/off.
        HTML_TARGET_COLOR = "#f28a24"   # warm orange
        HTML_SCENE_COLOR = "#5a6f8f"    # cool blue-grey
        pose_traces: List[Any] = []
        for g in grasps[: STATE.top_k_dump]:
            trans = torch.as_tensor(g.translation, dtype=torch.float32)[None]
            rot = torch.as_tensor(g.rotation, dtype=torch.float32)[None]
            if STATE.hand_family == "gripper":
                width = float(np.asarray(g.joint_angles).flatten()[0])
                qpos = torch.tensor([[width, GRIPPER_NEW_DEPTH]], dtype=torch.float32)
            else:
                qpos = torch.as_tensor(g.joint_angles, dtype=torch.float32)[None]
            is_target = int(getattr(g, "object_id", 1)) == 1
            color = HTML_TARGET_COLOR if is_target else HTML_SCENE_COLOR
            pose_traces += vis.robot_plotly(
                trans=trans, rot=rot, qpos=qpos, color=color
            )

        out_dir = STATE.log_dir / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{request_id}.html"
        vis.show(pc_traces + pose_traces, path=str(out_path))
        return out_path
    except Exception as exc:
        logger.warning(f"HTML dump failed for request {request_id}: {exc}")
        return None


def _dump_npz(
    request_id: str,
    point_cloud: np.ndarray,
    grasps: List[GraspPose],
    scene_points: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """
    Write a raw-data NPZ (point cloud + grasp arrays) for post-hoc replay.
    Saves into `<log_dir>/predictions/<request_id>.npz`. Cheap; safe to leave
    on for every request.
    """
    if STATE.log_dir is None:
        return None
    try:
        out_dir = STATE.log_dir / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{request_id}.npz"

        if grasps:
            translations = np.stack([np.asarray(g.translation) for g in grasps])
            rotations = np.stack([np.asarray(g.rotation) for g in grasps])
            joint_angles = np.stack(
                [np.asarray(g.joint_angles).flatten() for g in grasps]
            )
            scores = np.array([float(g.score) for g in grasps])
        else:
            translations = np.zeros((0, 3), dtype=np.float32)
            rotations = np.zeros((0, 3, 3), dtype=np.float32)
            joint_angles = np.zeros((0, 1), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)

        save_args: Dict[str, Any] = dict(
            point_cloud=point_cloud.astype(np.float32, copy=False),
            translations=translations,
            rotations=rotations,
            joint_angles=joint_angles,
            scores=scores,
            hand_family=np.array(STATE.hand_family),
            hand_name=np.array(STATE.hand_config.name),
        )
        if scene_points is not None and scene_points.size > 0:
            save_args["scene_points"] = scene_points.astype(np.float32, copy=False)
        np.savez(out_path, **save_args)
        return out_path
    except Exception as exc:
        logger.warning(f"NPZ dump failed for request {request_id}: {exc}")
        return None


# --- USD (Isaac Sim) debug dump ---------------------------------------------
#
# Generic parametric parallel-jaw gripper geometry, matching the constants the
# DexGraspNet 2.0 paper uses (src/utils/robot_info.py). We intentionally DO
# NOT render any real-robot mesh (e.g. Panda) — the model wasn't trained on
# any specific gripper, just the generic 4-box shape below, and showing a
# real mesh would falsely imply a geometry the model doesn't know about.
#
# Grasp-frame convention used here, taken verbatim from
# src/utils/vis_plotly.py::robot_plotly (gripper path):
#   +x → approach axis (fingers extend this direction)
#   +y → jaw-opening axis (distance between fingers = `width`)
#   +z → gripper height (thin)

_GRIPPER_NEW_DEPTH = 0.04      # finger length
_GRIPPER_DEPTH_BASE = 0.02     # small offset behind fingertips
_GRIPPER_TAIL_LENGTH = 0.04    # stem behind the crossbar
_GRIPPER_FINGER_WIDTH = 0.004  # visual thickness of each finger rod
_GRIPPER_HEIGHT = 0.004        # gripper "thickness" in z


def _mat4_to_gf(mat4: np.ndarray):
    """Convert numpy 4x4 (column-vector convention) to USD Gf.Matrix4d."""
    from pxr import Gf
    mt = np.ascontiguousarray(mat4.T, dtype=np.float64)
    return Gf.Matrix4d(*mt.flatten().tolist())


def _set_xform(prim, transform_4x4: np.ndarray) -> None:
    """Apply a world-space 4x4 transform to a USD prim via a single matrix op."""
    from pxr import UsdGeom
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()
    op.Set(_mat4_to_gf(transform_4x4))


def _add_gripper_fork(
    stage,
    path: str,
    transform_4x4: np.ndarray,
    width: float,
    color: tuple = (0.9, 0.5, 0.25),
) -> None:
    """Draw a generic parallel-jaw gripper as 4 thin boxes at the given pose."""
    from pxr import Gf, UsdGeom

    depth = _GRIPPER_NEW_DEPTH
    fw = _GRIPPER_FINGER_WIDTH
    db = _GRIPPER_DEPTH_BASE
    tl = _GRIPPER_TAIL_LENGTH
    h = _GRIPPER_HEIGHT

    # (center_in_grasp_frame, scale) per box, matching Vis.robot_plotly.
    specs = [
        # finger on +y
        (np.array([(depth - fw - db) / 2, (width + fw) / 2, 0.0]),
         np.array([fw + db + depth, fw, h])),
        # finger on -y
        (np.array([(depth - fw - db) / 2, -(width + fw) / 2, 0.0]),
         np.array([fw + db + depth, fw, h])),
        # crossbar at base
        (np.array([-db - fw / 2, 0.0, 0.0]),
         np.array([fw, width, h])),
        # tail/stem behind crossbar
        (np.array([-db - fw - tl / 2, 0.0, 0.0]),
         np.array([tl, fw, h])),
    ]

    parent = UsdGeom.Xform.Define(stage, path)
    _set_xform(parent.GetPrim(), transform_4x4)

    for i, (center, scale) in enumerate(specs):
        cube = UsdGeom.Cube.Define(stage, f"{path}/box{i}")
        cube.CreateSizeAttr(1.0)
        xformable = UsdGeom.Xformable(cube)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        xformable.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in scale]))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _add_axis_gizmo(
    stage,
    path: str,
    transform_4x4: np.ndarray,
    arm_length: float = 0.02,
    arm_radius: float = 0.001,
) -> None:
    """Three colored cylinders (X=red, Y=green, Z=blue) at the given pose."""
    from pxr import Gf, UsdGeom
    parent = UsdGeom.Xform.Define(stage, path)
    _set_xform(parent.GetPrim(), transform_4x4)
    z = np.array([0.0, 0.0, 1.0])
    for label, direction, color in [
        ("X", (1.0, 0.0, 0.0), (0.95, 0.15, 0.15)),
        ("Y", (0.0, 1.0, 0.0), (0.15, 0.95, 0.15)),
        ("Z", (0.0, 0.0, 1.0), (0.15, 0.35, 0.95)),
    ]:
        d = np.array(direction, dtype=float)
        cyl = UsdGeom.Cylinder.Define(stage, f"{path}/Axis_{label}")
        cyl.CreateRadiusAttr(arm_radius)
        cyl.CreateHeightAttr(arm_length)
        cyl.CreateAxisAttr(UsdGeom.Tokens.z)
        cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if np.allclose(d, z):
            R = np.eye(3)
        else:
            axis = np.cross(z, d)
            n = np.linalg.norm(axis)
            if n < 1e-9:
                R = Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()
            else:
                R = Rotation.from_rotvec((axis / n) * np.arccos(float(np.dot(z, d)))).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = d * (arm_length / 2.0)
        _set_xform(cyl.GetPrim(), T)


def _dump_usd(
    request_id: str,
    point_cloud: np.ndarray,
    grasps: List[GraspPose],
    scene_points: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """
    Write an Isaac Sim-ready USD containing:
      - the target-object point cloud as UsdGeomPoints (warm color)
      - the surrounding scene points if supplied (cool color)
      - the top-K predicted grasps, each as a parametric 4-box gripper fork
      - a small axis gizmo at each grasp origin for orientation
    Output lands at `<log_dir>/predictions/<request_id>.usd`.
    """
    if STATE.log_dir is None:
        return None
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
    except Exception as exc:
        logger.warning(f"pxr (USD) not available; skipping USD dump: {exc}")
        return None

    try:
        out_dir = STATE.log_dir / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{request_id}.usd"
        if out_path.exists():
            out_path.unlink()

        stage = Usd.Stage.CreateNew(str(out_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        # Target-object point cloud (warm color to distinguish from scene).
        pts_prim = UsdGeom.Points.Define(stage, "/World/TargetPointCloud")
        pts_prim.CreatePointsAttr(
            [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in point_cloud]
        )
        pts_prim.CreateWidthsAttr([0.003] * len(point_cloud))
        pts_prim.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.65, 0.20)])

        # Surrounding scene points, if provided — rendered smaller and cooler
        # so it's visually obvious what's the grasp target vs. what's clutter
        # context for the model.
        if scene_points is not None and scene_points.size > 0:
            scene_prim = UsdGeom.Points.Define(stage, "/World/ScenePointCloud")
            scene_prim.CreatePointsAttr(
                [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
                 for p in scene_points]
            )
            scene_prim.CreateWidthsAttr([0.0015] * len(scene_points))
            scene_prim.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.40, 0.50)])

        # World-origin axis gizmo — orient at a glance in Isaac Sim.
        _add_axis_gizmo(stage, "/World/OriginGizmo", np.eye(4), arm_length=0.03)

        # Top-K grasps. `grasps` is already sorted descending by score.
        # Color-code by object_id so target vs. scene-context grasps are
        # visually distinct — matches the target/scene point-cloud color
        # conventions applied above.
        is_gripper = STATE.hand_family == "gripper"
        COLOR_TARGET = (0.95, 0.55, 0.20)   # warm orange — same family as target cloud
        COLOR_SCENE = (0.35, 0.45, 0.65)    # cool blue-grey — same family as scene cloud
        for i, g in enumerate(grasps[: STATE.top_k_dump]):
            T = np.eye(4)
            T[:3, :3] = np.asarray(g.rotation, dtype=np.float64)
            T[:3, 3] = np.asarray(g.translation, dtype=np.float64)

            is_target = int(getattr(g, "object_id", 1)) == 1
            # Preserve rank information via a small hue shift: top-ranked
            # grasps of each family are slightly brighter; tail grasps slightly
            # darker. Keeps colors clearly target-vs-scene while also
            # communicating "this was the first/highest-scoring."
            rank_shade = 1.0 - 0.35 * (
                i / max(len(grasps[: STATE.top_k_dump]) - 1, 1)
            )
            base = COLOR_TARGET if is_target else COLOR_SCENE
            color = tuple(float(c * rank_shade) for c in base)

            grasp_root = f"/World/Grasp_{i:02d}_{'target' if is_target else 'scene'}"

            if is_gripper:
                width = float(np.asarray(g.joint_angles).flatten()[0])
                _add_gripper_fork(stage, grasp_root, T, width=width, color=color)
            else:
                # Dex hand fallback: just show a coordinate gizmo + sphere at
                # the wrist. The model's joint_angles aren't rendered as a
                # mesh here (would need URDF FK); NPZ carries the full data.
                _add_axis_gizmo(stage, grasp_root + "_axes", T, arm_length=0.04)

            # Always add a small gizmo at grasp origin.
            _add_axis_gizmo(
                stage,
                grasp_root + "/OriginGizmo",
                T,
                arm_length=0.015,
                arm_radius=0.0008,
            )

        light = UsdLux.DistantLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(3000.0)

        stage.GetRootLayer().Save()
        return out_path
    except Exception as exc:
        logger.warning(f"USD dump failed for request {request_id}: {exc}")
        return None


def _append_request_log(record: Dict[str, Any]) -> None:
    """Append a single request-summary JSON line to requests.jsonl."""
    if STATE.requests_jsonl is None:
        return
    try:
        with STATE.requests_jsonl.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to append to requests.jsonl: {exc}")


def _compute_approach_axis_wrist(hand_config: HandConfig) -> np.ndarray:
    """TCP +Z expressed in the wrist frame. Matches GraspSimulator's convention."""
    if hand_config.tcp_rotation_rpy is not None:
        R_wrist_tcp = Rotation.from_euler(
            "xyz", hand_config.tcp_rotation_rpy, degrees=True
        ).as_matrix()
        return R_wrist_tcp @ np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 0.0, 1.0])


def _grasp_to_response(
    g: GraspPose,
    hand_family: str,
    joint_names: Optional[List[str]],
    approach_axis_wrist: np.ndarray,
) -> Dict[str, Any]:
    """
    Convert a GraspPose into the spec-compliant dict. `approach_axis` is
    returned in the input point-cloud (world) frame — the client uses it
    directly for trajectory planning without needing to rotate a grasp-local
    vector.
    """
    R = np.asarray(g.rotation, dtype=np.float64)

    # (x, y, z, w) per ROS convention
    quat_xyzw = Rotation.from_matrix(R).as_quat().tolist()

    approach_world = (R @ approach_axis_wrist).tolist()

    # Hand-family dispatch for hand-specific fields.
    is_gripper = hand_family == "gripper"
    joints = np.asarray(g.joint_angles, dtype=np.float64).flatten()

    if is_gripper:
        # For a 1-DoF parallel gripper, joint_angles[0] is the target jaw width.
        gripper_width = float(joints[0]) if joints.size >= 1 else 0.0
        joint_angles_out: Optional[List[float]] = None
        joint_names_out: Optional[List[str]] = None
    else:
        gripper_width = None
        joint_angles_out = joints.tolist()
        joint_names_out = list(joint_names) if joint_names else None

    return {
        "translation": np.asarray(g.translation, dtype=np.float64).tolist(),
        "rotation_quat": quat_xyzw,
        "rotation_matrix": R.tolist(),
        "score": float(g.score),
        "gripper_width": gripper_width,
        "joint_angles": joint_angles_out,
        "joint_names": joint_names_out,
        "approach_axis": approach_world,
        # Matches the segmentation mask the server built from the request:
        #   1 = grasp is anchored on target-object points (those sent as
        #       `point_cloud`)
        #   0 = grasp is anchored on clutter (those sent as `scene_points`,
        #       or any point when `scene_points` was omitted — in which case
        #       all returned grasps have object_id == 1).
        "object_id": int(g.object_id),
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _build_app() -> FastAPI:
    app = FastAPI(
        title="DexGraspNet2 Grasp Predictor",
        version=API_VERSION,
        description="HTTP interface over a pretrained DexGraspNet2 model.",
    )

    # -- /healthz -----------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> Any:
        if STATE.loaded:
            return {"status": "ok"}
        return JSONResponse(status_code=503, content={"status": "loading"})

    # -- /config ------------------------------------------------------------

    @app.get("/config")
    def config() -> Any:
        if not STATE.loaded:
            raise HTTPException(
                status_code=503,
                detail={"error": "not_ready", "retry_after_seconds": 5},
            )
        joint_names = (
            None
            if STATE.hand_family == "gripper"
            else list(STATE.hand_config.joint_names)
        )
        return {
            "hand": STATE.hand_config.name,
            "model_name": STATE.model_name,
            "checkpoint": STATE.checkpoint_name,
            "joint_names": joint_names,
            "grasp_reference": STATE.grasp_reference,
            "default_num_grasps": 20,
            "min_points": STATE.min_points,
            "max_points": STATE.max_points,
            "server_version": SERVER_VERSION,
        }

    # -- /version -----------------------------------------------------------

    @app.get("/version")
    def version() -> Any:
        return {
            "server_version": SERVER_VERSION,
            "api_version": API_VERSION,
            "model_commit": STATE.model_commit,
            "built_at": BUILT_AT,
        }

    # -- /predict -----------------------------------------------------------

    @app.post("/predict")
    def predict(req: PredictRequest) -> Any:
        if not STATE.loaded or STATE.predictor is None:
            raise HTTPException(
                status_code=503,
                detail={"error": "not_ready", "retry_after_seconds": 5},
            )

        # Decode + validate target-object point cloud.
        pc = _decode_array(req.point_cloud, _ALLOWED_PC_DTYPES)

        if pc.ndim != 2 or pc.shape[1] != 3:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "bad_shape",
                    "message": f"point_cloud must be (N, 3); got {pc.shape}",
                },
            )

        n = pc.shape[0]
        if n < STATE.min_points:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_geometry",
                    "message": f"need at least {STATE.min_points} points for "
                               f"the target object; got {n}",
                },
            )
        if not np.isfinite(pc).all():
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_geometry",
                    "message": "point cloud contains NaN or Inf values",
                },
            )
        pc = pc.astype(np.float32, copy=False)

        # Optionally decode scene_points for clutter-aware inference.
        # This reproduces the paper's training/evaluation mode: the whole
        # scene is fed to the backbone, with a per-point segmentation mask
        # telling the model which points are the grasp target (=1) vs
        # surrounding clutter (=0). Context improves grasp quality in
        # cluttered scenes; without it the backbone has no way to know
        # which approaches are blocked by neighboring geometry.
        scene_pts: Optional[np.ndarray] = None
        if req.scene_points is not None:
            scene_pts = _decode_array(req.scene_points, _ALLOWED_PC_DTYPES)
            if scene_pts.ndim != 2 or scene_pts.shape[1] != 3:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "bad_shape",
                        "message": f"scene_points must be (M, 3); "
                                   f"got {scene_pts.shape}",
                    },
                )
            if not np.isfinite(scene_pts).all():
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "invalid_geometry",
                        "message": "scene_points contains NaN or Inf values",
                    },
                )
            scene_pts = scene_pts.astype(np.float32, copy=False)

        # Enforce max_points on the total that reaches the model.
        total_points = n + (scene_pts.shape[0] if scene_pts is not None else 0)
        if total_points > STATE.max_points:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "payload_too_large",
                    "max_points": STATE.max_points,
                    "message": f"point_cloud ({n}) + scene_points "
                               f"({total_points - n}) = {total_points} > "
                               f"{STATE.max_points}",
                },
            )

        # Build the array the predictor actually sees. Concatenation order
        # matters: target-object points first, then clutter — the segmentation
        # mask is built from that order below.
        if scene_pts is not None:
            full_pc = np.concatenate([pc, scene_pts], axis=0).astype(np.float32)
            segmentation = np.concatenate(
                [np.ones(n, dtype=np.int64), np.zeros(scene_pts.shape[0], dtype=np.int64)]
            )
        else:
            full_pc = pc
            segmentation = None

        # Predict. We oversample a bit internally (model's `num_grasps` arg is
        # candidates, not top-K) and then truncate to the requested size.
        # Resolve category sampling: per-request override wins, else the
        # server's startup default.
        category_sampling = (
            req.category_sampling
            if req.category_sampling is not None
            else STATE.default_category_sampling
        )
        model_num = max(req.num_grasps * 8, 64)
        request_id = uuid.uuid4().hex[:12]
        t0 = time.time()
        try:
            result = STATE.predictor.predict(
                full_pc,
                segmentation=segmentation,
                num_grasps=model_num,
                use_category_sampling=bool(category_sampling),
            )
        except Exception as exc:
            logger.exception("Inference failed")
            _append_request_log(
                {
                    "request_id": request_id,
                    "time": _dt.datetime.utcnow().isoformat() + "Z",
                    "num_input_points": int(n),
                    "num_scene_points": (
                        int(scene_pts.shape[0]) if scene_pts is not None else 0
                    ),
                    "num_grasps_requested": req.num_grasps,
                    "error": str(exc),
                }
            )
            raise HTTPException(
                status_code=500,
                detail={"error": "internal_error", "message": str(exc)},
            )
        inference_ms = (time.time() - t0) * 1000.0

        # Convert, filter by score, clip to requested count. `result.grasps` is
        # already sorted descending by score.
        joint_names = (
            None
            if STATE.hand_family == "gripper"
            else list(STATE.hand_config.joint_names)
        )
        kept_grasps: List[GraspPose] = []
        grasps_out: List[Dict[str, Any]] = []
        for g in result.grasps:
            if g.score < req.min_score:
                continue
            grasps_out.append(
                _grasp_to_response(
                    g,
                    hand_family=STATE.hand_family,
                    joint_names=joint_names,
                    approach_axis_wrist=STATE.approach_axis_wrist,
                )
            )
            kept_grasps.append(g)
            if len(grasps_out) >= req.num_grasps:
                break

        # Optional per-request dumps + request-summary log. The dumps always
        # visualize the target-object points; scene_points is passed through
        # too so the viz can render clutter in a distinct color when present.
        html_path: Optional[Path] = None
        usd_path: Optional[Path] = None
        npz_path: Optional[Path] = None
        if kept_grasps:
            if STATE.dump_html:
                html_path = _dump_html(request_id, pc, kept_grasps, scene_pts)
            if STATE.dump_usd:
                usd_path = _dump_usd(request_id, pc, kept_grasps, scene_pts)
            if STATE.dump_npz:
                npz_path = _dump_npz(request_id, pc, kept_grasps, scene_pts)
        num_scene = int(scene_pts.shape[0]) if scene_pts is not None else 0
        _append_request_log(
            {
                "request_id": request_id,
                "time": _dt.datetime.utcnow().isoformat() + "Z",
                "num_input_points": int(n),
                "num_scene_points": num_scene,
                "num_grasps_requested": req.num_grasps,
                "num_grasps_returned": len(grasps_out),
                "top_score": float(grasps_out[0]["score"]) if grasps_out else None,
                "inference_ms": round(inference_ms, 3),
                "html_path": str(html_path) if html_path else None,
                "usd_path": str(usd_path) if usd_path else None,
                "npz_path": str(npz_path) if npz_path else None,
            }
        )

        return {
            "grasps": grasps_out,
            "meta": {
                "hand": STATE.hand_config.name,
                "model_name": STATE.model_name,
                "checkpoint": STATE.checkpoint_name,
                "frame_id": "unchanged_from_input",
                "num_input_points": int(n),
                "num_scene_points": num_scene,
                "context_mode": "scene_masked" if num_scene > 0 else "object_only",
                "category_sampling": bool(category_sampling),
                "inference_ms": round(inference_ms, 3),
                "server_version": SERVER_VERSION,
                "request_id": request_id,
                "debug_html": str(html_path) if html_path else None,
                "debug_usd": str(usd_path) if usd_path else None,
                "debug_npz": str(npz_path) if npz_path else None,
            },
        }

    return app


# ---------------------------------------------------------------------------
# Startup / wiring
# ---------------------------------------------------------------------------


def _guess_hand_family(hand_config: HandConfig) -> tuple:
    """
    Return (family, grasp_reference) based on hand_config.
    family ∈ {"gripper", "dex"}.
    """
    name_lower = (hand_config.name or "").lower()
    if "gripper" in name_lower or hand_config.num_dofs == 1:
        return "gripper", "tcp_between_jaws"
    return "dex", "wrist_link"


def _setup_log_dir(
    log_dir_arg: Optional[Path],
    hand_name_hint: str,
    dump_html: bool,
    dump_usd: bool,
    dump_npz: bool,
    top_k_dump: int,
    cli_args: Dict[str, Any],
) -> None:
    """
    Create `.logs/<timestamp>_server_<hand>/` (or whatever was passed), attach
    a FileHandler so every logger.info() also lands in run.log, and prepare
    the requests.jsonl path for per-request summaries.
    """
    if log_dir_arg is None:
        ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        log_dir = Path(".logs") / f"{ts}_server_{hand_name_hint}"
    else:
        log_dir = Path(log_dir_arg)
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_dir / "run.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logging.getLogger().addHandler(fh)

    STATE.log_dir = log_dir
    STATE.requests_jsonl = log_dir / "requests.jsonl"
    STATE.dump_html = bool(dump_html)
    STATE.dump_usd = bool(dump_usd)
    STATE.dump_npz = bool(dump_npz)
    STATE.top_k_dump = int(top_k_dump)

    logger.info(f"Server log directory: {log_dir}")
    if dump_html or dump_usd or dump_npz:
        enabled = []
        if dump_html:
            enabled.append("HTML")
        if dump_usd:
            enabled.append("USD")
        if dump_npz:
            enabled.append("NPZ")
        logger.info(
            f"Per-request dumps enabled ({', '.join(enabled)}); "
            f"landing in {log_dir}/predictions/"
        )


def _initialise_state(
    checkpoint: Path,
    hand_config_path: Optional[Path],
    device: str,
    min_points: int,
    max_points: int,
    model_commit: str,
    default_category_sampling: bool = False,
) -> None:
    """Load the model and populate STATE. Called before uvicorn.run()."""

    if hand_config_path is not None:
        if not hand_config_path.exists():
            raise FileNotFoundError(f"Hand config not found: {hand_config_path}")
        hand_config = HandConfig.from_yaml(hand_config_path)
    else:
        hand_config = None  # GraspPredictor will try to infer.

    predictor = GraspPredictor(
        checkpoint_path=checkpoint,
        hand_config=hand_config,
        device=device,
    )

    if hand_config is None:
        hand_config = predictor.hand_config

    family, grasp_ref = _guess_hand_family(hand_config)

    # Checkpoint name structure: <exp_dir>/ckpt/<file>.pth  →  exp_dir name
    # is a decent model_name; the filename itself is the checkpoint field.
    ckpt_path = Path(checkpoint)
    model_name = ckpt_path.parent.parent.name
    checkpoint_name = ckpt_path.name

    STATE.predictor = predictor
    STATE.hand_config = hand_config
    STATE.hand_family = family
    STATE.grasp_reference = grasp_ref
    STATE.checkpoint_name = checkpoint_name
    STATE.model_name = model_name
    STATE.min_points = min_points
    STATE.max_points = max_points
    STATE.approach_axis_wrist = _compute_approach_axis_wrist(hand_config)
    STATE.model_commit = model_commit
    STATE.default_category_sampling = bool(default_category_sampling)
    STATE.loaded = True

    logger.info(
        f"Loaded {model_name} ({hand_config.name}, family={family}) "
        f"from {checkpoint_name}. "
        f"approach_axis_wrist={STATE.approach_axis_wrist.tolist()}."
    )

    # Persist the resolved config for this run so a human can see exactly what
    # the server is answering with.
    if STATE.log_dir is not None:
        try:
            with (STATE.log_dir / "config.json").open("w") as f:
                json.dump(
                    {
                        "server_version": SERVER_VERSION,
                        "checkpoint": str(ckpt_path),
                        "hand": hand_config.name,
                        "hand_family": family,
                        "num_dofs": hand_config.num_dofs,
                        "joint_names": list(hand_config.joint_names),
                        "approach_axis_wrist": STATE.approach_axis_wrist.tolist(),
                        "min_points": min_points,
                        "max_points": max_points,
                        "dump_html": STATE.dump_html,
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:
            logger.warning(f"Failed to write config.json: {exc}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve a DexGraspNet2 grasp predictor over HTTP.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to .pth checkpoint file. The parent directory must also "
             "contain a config.yaml (standard DexGraspNet2 layout).",
    )
    parser.add_argument(
        "--hand-config",
        type=Path,
        default=None,
        help="Optional path to a hand YAML (dexgraspnet2/configs/hands/*.yaml). "
             "If omitted, GraspPredictor attempts to infer the hand from the "
             "checkpoint config. Required for gripper/Inspire.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--min-points",
        type=int,
        default=128,
        help="Reject point clouds with fewer than this many points (HTTP 422).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=65536,
        help="Reject point clouds with more than this many points (HTTP 413).",
    )
    parser.add_argument(
        "--model-commit",
        type=str,
        default="",
        help="Optional git commit / build tag reported via /version.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="uvicorn log level (debug|info|warning|error).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory to write run.log, config.json, requests.jsonl, "
             "and per-request HTML dumps (if --dump-html). "
             "Default: .logs/<timestamp>_server_<hand>/",
    )
    parser.add_argument(
        "--dump-html",
        action="store_true",
        help="For every /predict call, write a Plotly HTML visualization "
             "of the input point cloud + top-K grasps to "
             "<log_dir>/predictions/<request_id>.html. Browser-friendly; "
             "same 4-box gripper geometry as tests/visualize_gripper_pred.py.",
    )
    parser.add_argument(
        "--dump-usd",
        action="store_true",
        help="For every /predict call, write an Isaac Sim-ready USD file "
             "of the input point cloud + top-K grasps to "
             "<log_dir>/predictions/<request_id>.usd. Uses the paper's "
             "generic parallel-jaw geometry (4 thin boxes + origin gizmo) "
             "— NOT any real gripper mesh — because the DexGraspNet2 model "
             "was trained on that generic shape, not on Panda / other real hardware.",
    )
    parser.add_argument(
        "--dump-npz",
        action="store_true",
        help="For every /predict call, save the raw input point cloud and "
             "output grasp arrays to <log_dir>/predictions/<request_id>.npz. "
             "Useful for offline replay / custom visualization.",
    )
    parser.add_argument(
        "--top-k-dump",
        type=int,
        default=10,
        help="How many grasps to render into HTML / USD dumps (NPZ always "
             "saves all returned grasps).",
    )
    parser.add_argument(
        "--category-sampling",
        action="store_true",
        help="Default value for the `category_sampling` request field. "
             "When True, the model samples seed points with per-instance "
             "balancing (≈ equal counts for target vs. scene in the top-K). "
             "When False (default), sampling is global, weighted by "
             "graspness — scene often dominates in cluttered inputs. Clients "
             "can override per request via the `category_sampling` field.",
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return 2

    # Log dir must be set BEFORE _initialise_state so config.json lands there.
    # Use the hand-config filename stem as a tag if available, else "unknown".
    hand_tag = (
        args.hand_config.stem if args.hand_config is not None else "unknown"
    )
    _setup_log_dir(
        log_dir_arg=args.log_dir,
        hand_name_hint=hand_tag,
        dump_html=args.dump_html,
        dump_usd=args.dump_usd,
        dump_npz=args.dump_npz,
        top_k_dump=args.top_k_dump,
        cli_args=vars(args),
    )

    _initialise_state(
        checkpoint=args.checkpoint,
        hand_config_path=args.hand_config,
        device=args.device,
        min_points=args.min_points,
        max_points=args.max_points,
        model_commit=args.model_commit,
        default_category_sampling=args.category_sampling,
    )

    app = _build_app()
    logger.info(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
