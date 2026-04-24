# Inference — running the pretrained model

Two ways to get grasps out of a pretrained checkpoint:

1. **FastAPI server** (`scripts/serve_grasp_predictor.py`) — load once, serve grasps over HTTP. Recommended for robot integrations (ROS 2, etc.).
2. **One-shot scripts** (`tests/visualize_dex_pred.py`, `tests/evaluate_predicted_grasps.py`, ...) — useful for ad-hoc inspection, debugging, and the [Quick test](../README.md#quick-test-that-everything-works) in the README.

---

## The FastAPI server

Loads one checkpoint at startup, exposes `POST /predict` + `/healthz` / `/config` / `/version`. Full API schema lives in [`docs/api/predict.md`](api/predict.md) — this page is the operator's-eye overview.

### Launch (LEAP hand)

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  --network host \
  -v $(pwd):/workspace \
  -v /path/to/data:/workspace/data \
  -w /workspace -e PYTHONPATH=/workspace \
  dexgraspnet2:latest \
  bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py38 && \
    python scripts/serve_grasp_predictor.py \
      --checkpoint data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth \
      --host 0.0.0.0 --port 8000"
```

No `--hand-config` flag needed: `GraspPredictor` reads `data.robot` from the checkpoint's sibling `config.yaml` and auto-loads `HandConfig.leap_hand()`.

`--user $(id -u):$(id -g)` + `HOME=/tmp` makes any debug dumps (`--dump-html`, `--dump-usd`, `--dump-npz`) under `.logs/` land owned by your host user. Applies to every `docker run` in this doc.

### Launch (parallel-jaw gripper)

The gripper checkpoint's `config.yaml` doesn't identify its hand family, so the server needs an explicit `--hand-config`:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  --network host \
  -v $(pwd):/workspace \
  -v /path/to/data:/workspace/data \
  -w /workspace -e PYTHONPATH=/workspace \
  dexgraspnet2:latest \
  bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py38 && \
    python scripts/serve_grasp_predictor.py \
      --checkpoint data/DexGraspNet2.0-ckpts/OURS_gripper/ckpt/ckpt_50000.pth \
      --hand-config dexgraspnet2/configs/hands/gripper.yaml \
      --host 0.0.0.0 --port 8000"
```

### Optional debug dumps

Add any combination of:

| Flag | What it saves per `/predict` call |
|---|---|
| `--dump-html` | Plotly HTML (browser-openable) of point cloud + top-K grasps |
| `--dump-usd` | Isaac Sim-ready USD with the same scene (useful for cross-checking against a ROS viz — same hand/scene content as the HTML) |
| `--dump-npz` | Raw input cloud + output grasp arrays — for offline replay |
| `--top-k-dump N` | How many grasps to render into HTML / USD (default 10) |

Files land under `.logs/<timestamp>_server_<hand>/predictions/<request_id>.{html,usd,npz}`. Each response's `meta.request_id` / `meta.debug_*` fields point back at the dumped files so the ROS / client side can correlate.

### Operational knobs

| Flag | Default | Notes |
|---|---|---|
| `--host`, `--port` | `0.0.0.0`, `8000` | Pair with `--network host` for local ROS 2 access. |
| `--min-points` | 128 | Reject requests with fewer target points (HTTP 422). |
| `--max-points` | 65536 | Reject total (target + scene) points above this (HTTP 413). |
| `--category-sampling` | off | Per-instance balanced seed sampling. Clients can also override per-request via the `category_sampling` field. |
| `--log-dir` | `.logs/<ts>_server_<hand>/` | Override the per-run log directory. |

---

## Calling the API

See [`docs/api/predict.md`](api/predict.md) for the full schema. Short version: `POST /predict` with a base64-encoded `(N, 3)` float32 point cloud; get back a ranked list of grasps.

Minimal body (object-only mode):

```json
{
  "point_cloud": { "dtype": "float32", "shape": [N, 3], "data_b64": "..." },
  "num_grasps":  20
}
```

Clutter-aware body (pass scene context — matches the paper's native inference mode):

```json
{
  "point_cloud":       { "dtype": "float32", "shape": [N, 3], "data_b64": "..." },
  "scene_points":      { "dtype": "float32", "shape": [M, 3], "data_b64": "..." },
  "num_grasps":        20,
  "category_sampling": true
}
```

### Three API semantics worth knowing

1. **Frame**: the server does no frame transformations. Input points are interpreted verbatim and returned grasps are in the same frame. Your ROS 2 node handles any `tf2` work.

2. **`scene_points`**: optional. When present, the server concatenates `[target; scene]` and builds a segmentation mask (target = 1, scene = 0) for the backbone. This lets the model's features see surrounding geometry, matching how DexGraspNet 2.0 was trained. Recommended for cluttered scenes (bowl of apples, objects in a bin).

3. **`object_id` per grasp**: the response tags each grasp with its source instance. `1` = grasped on target points, `0` = grasped on scene/clutter points. Meaningful only when `category_sampling=true` — in global-sampling mode the model doesn't track seed origins and returns `-1`. For target-only filtering client-side:

    ```python
    target_grasps = [g for g in response["grasps"] if g["object_id"] == 1]
    ```

---

## One-shot prediction + visualization (no server)

### LEAP hand (used by the [Quick test](../README.md#quick-test-that-everything-works))

```bash
python tests/visualize_dex_pred.py \
  --ckpt_path data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth \
  --scene scene_0090 \
  --output_path outputs/dex_pred.html
```

Loads the checkpoint, runs the model on the scene's ground-truth point cloud, selects the top-scoring grasp per instance, writes an interactive Plotly HTML.

### Parallel-jaw gripper

```bash
python tests/visualize_gripper_pred.py \
  --ckpt_path data/DexGraspNet2.0-ckpts/OURS_gripper/ckpt/ckpt_50000.pth \
  --scene scene_0090 \
  --output_path outputs/gripper_pred.html
```

### Full perception + inference (synthetic camera capture)

Spawns the scene object in Isaac Gym, captures a simulated RGB-D frame, unprojects to a point cloud, runs the model. Heavier but validates the whole pipeline from sensor to grasp.

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  -v /path/to/data:/workspace/data \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  dexgraspnet2:latest \
  bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py38 && \
    python tests/evaluate_predicted_grasps.py \
      --ckpt_path data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth \
      --scene_id scene_0220 \
      --num_grasps 10 \
      --output_vis tests/output/eval_vis.html"
```

### Legacy batch inference (paper authors' script)

```bash
python src/eval/predict_dexterous_all_cates.py \
  --ckpt experiments/my_experiment/ckpt/ckpt_50000.pth
```

Kept for reproducing the paper's original evaluation flow. The FastAPI server is the preferred entry point for new code.

---

## Grasp-frame axis convention (parallel gripper)

Every grasp the model emits is a pose `(translation, rotation)` in the same frame as the input point cloud. The *grasp-local* frame — the one the rotation matrix rotates into — is the paper's convention, documented by the ASCII diagram in `src/utils/vis_plotly.py`:

```
        +y  (jaw-opening axis — distance between fingers = `width`)
         |
         O———— +x  (approach axis — fingers extend this direction,
        /                        length ≈ GRIPPER_NEW_DEPTH = 0.04 m)
       /
      +z  (gripper thickness / "height", thin)
```

Concrete mapping:

| Axis | Colour in USD gizmo | Role | Paper constant |
|---|---|---|---|
| **+X** | red   | approach — fingers extend along +X; hand moves +X to reach the object | `GRIPPER_NEW_DEPTH = 0.04` |
| **+Y** | green | jaw-opening axis — the two fingers sit at `±width/2` in Y | `GRIPPER_MAX_WIDTH = 0.1` |
| **+Z** | blue  | gripper plate thickness — thin, irrelevant to grasp planning | `GRIPPER_HEIGHT ≈ 0.004` |

Implications:

- The **approach vector** in world frame is `rotation_matrix @ [1, 0, 0]` (first column of the rotation). The inference server already exposes this as `approach_axis` per grasp — no client-side conversion needed.
- `gripper_width` in the API response is measured along the **grasp-local +Y**; at the grasp moment the jaws sit at `translation ± (width/2) * (rotation_matrix @ [0, 1, 0])` in world frame.
- The TCP reference point for gripper grasps is the **midpoint between the jaws at the base of the fingers** (`grasp_reference: tcp_between_jaws` in `/config`).

This convention is consistent across `Vis.robot_plotly` (the paper's Plotly viz), `tests/visualize_gripper_pred.py`, and the server's USD / HTML / NPZ dumps — the same rotation matrix plugs into any of them without remapping.

For dex hands, the grasp-local frame is defined by that hand's `tcp_rotation_rpy` in its YAML. The server computes the approach axis per hand-config and still returns it as `approach_axis` per grasp, so downstream code doesn't need to special-case.

---

## ROS 2 integration

See [`docs/api/predict.md`](api/predict.md#handling-the-response-in-your-ros-2-node) for a full PoseArray-publishing snippet and the filter-by-`object_id` idiom. Summary:

- Convert `sensor_msgs/PointCloud2` → `(N, 3)` float32 → base64.
- Send target cloud as `point_cloud`. Send everything else as `scene_points`.
- Response grasps are in the *same frame* as the request — respect the `header.frame_id` when publishing as `PoseArray`.
- Filter `object_id == 1` unless you want scene-context grasps for debugging.
- Log `meta.request_id` — it correlates directly with the server's `.html` / `.usd` / `.npz` dumps for post-hoc cross-check.
