# `POST /predict` — Grasp prediction API

Contract between the ROS 2 client and the inference server running inside the Docker container. This document is the source of truth for both sides.

---

## TL;DR

Send the point cloud of the object you want to grasp. **Optionally** also send the surrounding-scene points for clutter-aware inference. Get a ranked list of grasp poses back, in the same coordinate frame as the input. All geometry in meters / radians / unit quaternions.

```
POST /predict
Content-Type: application/json

Body (minimal):
  { "point_cloud": ..., "num_grasps": 20 }

Body (clutter-aware):
  { "point_cloud": ..., "scene_points": ..., "num_grasps": 20 }

→ 200 OK
   { "grasps": [ {translation, rotation_quat, ...}, ... ], "meta": { ... } }
```

**Segmentation is the client's responsibility.** The client separates target-object points from everything else. The target goes in `point_cloud`; optionally, the rest of the scene goes in `scene_points` so the model's backbone can see the clutter context (this is the paper's native training/eval mode).

---

## Endpoint

```
POST /predict
Content-Type: application/json
Accept:       application/json
```

No authentication assumed (the server is expected to live on a local network or `localhost` only). If auth is ever added, it'll be a `X-Api-Key` header; plan for that column being absent for now.

---

## Server assumptions

- The server is **configured at startup** for one specific hand type (gripper, LEAP, Inspire, etc.). Clients do not specify the hand per-request; the server answers with whatever it was loaded with.
- `GET /config` is provided so a client can introspect which hand/model is active (see below).
- Frame: the server does **no frame transformations**. All input points are interpreted verbatim, and all returned grasp poses are in the **same frame** as the input point cloud. The client is responsible for any TF gymnastics.
- Units: positions in meters, angles in radians, quaternions in `(x, y, z, w)` ordering (ROS convention), rotation matrices (if requested) are row-major 3×3.
- **Input expectation**: every point in `point_cloud` belongs to the object you want grasps on. Optional `scene_points` supplies the *rest* of the scene (non-target objects, table, bowl, etc.) as clutter context for the model's backbone. If `scene_points` is absent the model sees only the object in isolation — fine for single-object setups, but suboptimal for cluttered scenes where the paper's context-aware training actively helps.

## Grasp-frame axis convention (parallel gripper)

The `rotation_quat` / `rotation_matrix` in each grasp rotates the world frame into the **grasp-local** frame. Per the paper's convention (see `src/utils/vis_plotly.py::robot_plotly`):

| Axis | Role | Paper constant |
|---|---|---|
| **+X** | approach direction — fingers extend along +X, hand moves +X to contact the object | `GRIPPER_NEW_DEPTH = 0.04 m` |
| **+Y** | jaw-opening axis — the two fingers sit at `±gripper_width/2` along Y | `GRIPPER_MAX_WIDTH = 0.1 m` |
| **+Z** | plate thickness (thin, irrelevant to planning) | `GRIPPER_HEIGHT ≈ 0.004 m` |

Derived quantities clients frequently want:

```
approach_vec_world  = rotation_matrix @ [1, 0, 0]   # also returned as `approach_axis`
left_finger_world   = translation + rotation_matrix @ [0, +gripper_width/2, 0]
right_finger_world  = translation + rotation_matrix @ [0, -gripper_width/2, 0]
```

`translation` refers to the **midpoint between the jaws at the base of the fingers** (`grasp_reference: tcp_between_jaws` in the `/config` response). See the root `README.md` for the same convention with an ASCII diagram.

---

## Request body

JSON object with the following fields.

| Field | Type | Required | Description |
|---|---|---|---|
| `point_cloud` | object (see below) | ✓ | `(N, 3)` array of XYZ points in meters, filtered to only the target object. |
| `scene_points` | object (see below) | ✗ | `(M, 3)` array of XYZ points for the *rest* of the scene (non-target objects, table, container, etc.). When present, the server concatenates `[point_cloud; scene_points]`, builds a per-point segmentation mask (target=1, scene=0), and feeds that to the model. This matches the paper's training/eval mode and improves grasp quality in cluttered scenes. Same dtype/encoding as `point_cloud`. `N + M` must not exceed `max_points`. |
| `num_grasps` | int | ✗ | Max number of grasps to return. Default: `20`. Server may return fewer if the model produces fewer ranked candidates. |
| `min_score` | float | ✗ | Filter: drop grasps whose internal quality score is below this value. Default: `0.0` (keep all). |
| `category_sampling` | bool / null | ✗ | Override the server's seed-sampling strategy. `true` = per-instance balanced (roughly equal target / scene grasps in the top-K); `false` = global sampling weighted by predicted graspness (scene typically dominates in cluttered inputs). `null` (default) = use whatever the server was started with (`--category-sampling` CLI flag; off unless set). |

### When to use `scene_points`

- **Isolated object on table / bench / gripper jaws** → omit it. Cleaner request, model still gets a well-posed input.
- **Object in a bowl / stacked pile / adjacent to other clutter** → include it. The model's backbone features for the target will then "know" which approach directions are blocked by surrounding geometry, biasing the top-K toward physically achievable grasps. Same model, different inference mode.
- **You already have a scene PC from your perception stack (e.g. RGBD fusion)** → include everything that's NOT the target object. Minimal extra client code.

### When to flip `category_sampling`

Only relevant when you're supplying `scene_points`. Two regimes:

| `category_sampling` | What the server does | Top-K typically contains |
|---|---|---|
| `false` (default) | Global seed-point sampling weighted by predicted graspness | Mostly scene grasps when the scene has more / more-graspable geometry than the target |
| `true` | Per-instance balanced seed sampling (`k/2` target, `k/2` scene-ish) | Roughly equal target and scene grasps |

Rule of thumb: leave it at `false` unless you're seeing very few target grasps in the response (< ~3 per 20). Flipping to `true` is the cheapest way to guarantee target coverage without raising `num_grasps`.

### Encoded-array subschema for `point_cloud`

The point cloud is packed as a numpy array in JSON without ballooning the payload:

```json
{
  "dtype":    "float32" | "float64",
  "shape":    [N, 3],
  "data_b64": "AAAA..."
}
```

Semantics:
- `dtype`: numpy dtype string. Server MUST accept `float32` and `float64`.
- `shape`: declared shape, must be `[N, 3]` with `N >= 1`.
- `data_b64`: raw little-endian bytes, base64-encoded. Standard `base64.b64encode(array.tobytes())` on the client side.
- Server validates `prod(shape) * dtype_size == len(decoded_bytes)` and rejects mismatches with `400`.

### Example request (minimal, no scene context)

```json
{
  "point_cloud": {
    "dtype": "float32",
    "shape": [4096, 3],
    "data_b64": "..."
  },
  "num_grasps": 10,
  "min_score": 0.3
}
```

### Example request (clutter-aware)

```json
{
  "point_cloud": {
    "dtype": "float32",
    "shape": [2048, 3],
    "data_b64": "..."
  },
  "scene_points": {
    "dtype": "float32",
    "shape": [8192, 3],
    "data_b64": "..."
  },
  "num_grasps": 10,
  "category_sampling": true
}
```

---

## Response body (success, 200)

```json
{
  "grasps": [
    {
      "translation":      [x, y, z],
      "rotation_quat":    [qx, qy, qz, qw],
      "rotation_matrix":  [[r00,r01,r02], [r10,r11,r12], [r20,r21,r22]],
      "score":            0.82,
      "gripper_width":    0.057,
      "joint_angles":     null,
      "joint_names":      null,
      "approach_axis":    [0.0, 0.0, 1.0],
      "object_id":        1
    }
  ],
  "meta": {
    "hand":             "gripper",
    "model_name":       "OURS_gripper",
    "checkpoint":       "ckpt_50000.pth",
    "frame_id":         "unchanged_from_input",
    "num_input_points": 4096,
    "inference_ms":     138.4,
    "server_version":   "0.1.0"
  }
}
```

### Grasp object — field-by-field

| Field | Type | Always present? | Description |
|---|---|---|---|
| `translation` | `[float, float, float]` | ✓ | Position of the grasp reference point in the **input point-cloud frame**, meters. For grippers this is typically the TCP between the jaws; for dexterous hands it's the wrist link. Exact convention is server-configured and reported via `/config`. |
| `rotation_quat` | `[qx, qy, qz, qw]` | ✓ | Orientation as a unit quaternion, ROS `(x, y, z, w)` ordering. |
| `rotation_matrix` | `[[float]*3]*3` | ✓ | Same orientation as a row-major 3×3. Redundant with `rotation_quat` but both are returned so the client doesn't have to convert. |
| `score` | `float` | ✓ | Model's ranking/quality score. Higher is better; the list is sorted descending. Not calibrated to a probability — use as a relative ordering. |
| `gripper_width` | `float` or `null` | Only for gripper models | Target jaw opening at the grasp moment, meters. `null` for dexterous hands. |
| `joint_angles` | `[float, ...]` or `null` | Only for dexterous hands | Target joint angles at the grasp moment, radians, ordered by `joint_names`. `null` for grippers. |
| `joint_names` | `[string, ...]` or `null` | Only for dexterous hands | URDF joint names matching `joint_angles` element-for-element. Allows the client to remap without knowing the order in advance. |
| `approach_axis` | `[float, float, float]` | ✓ | Unit vector **in the input point-cloud frame** indicating which direction the hand moves to reach the grasp from its pregrasp pose. Computed as `rotation_matrix @ tcp_local_z`, so the client can use it directly for trajectory planning without any rotation math. |
| `object_id` | `int` | ✓ | Which input region this grasp is anchored on. Matches the segmentation mask the server built from the request: `1` = grasp is on the **target object** (points sent as `point_cloud`); `0` = grasp is on **clutter / scene context** (points sent as `scene_points`). When `scene_points` is omitted, all returned grasps have `object_id == 1`. Clients that want "target-only" grasps should filter `g["object_id"] == 1`. |

### Meta object

| Field | Type | Description |
|---|---|---|
| `hand` | string | Hand family: `"gripper"`, `"leap_hand"`, `"inspire_hand"`, etc. |
| `model_name` | string | Human-readable checkpoint label from the server config. |
| `checkpoint` | string | Filename of the loaded weights. |
| `frame_id` | string | Always `"unchanged_from_input"` — reminder that grasps are in the input frame. |
| `num_input_points` | int | Count of target-object points (N) the server received. |
| `num_scene_points` | int | Count of `scene_points` (M) the server received; `0` if omitted. |
| `context_mode` | string | `"object_only"` if `scene_points` was absent, `"scene_masked"` if it was provided. Quick sanity check that the server saw what the client meant to send. |
| `category_sampling` | bool | The effective value applied for this request (after resolving per-request override + server default). Log this alongside your grasp evaluations when comparing runs. |
| `inference_ms` | float | Wall-clock ms spent in model forward + sampling (excluding HTTP overhead). |
| `server_version` | string | Semver-ish version of the server build. |
| `request_id` | string | Short UUID for this request; matches filenames under `<log_dir>/predictions/`. |
| `debug_html` / `debug_usd` / `debug_npz` | string or null | Paths to per-request dumps if the server was started with `--dump-html` / `--dump-usd` / `--dump-npz`. |

---

## Errors

| Status | Condition | Body |
|---|---|---|
| 400 | Malformed JSON, missing `point_cloud`, encoded-array shape/dtype mismatch, `num_grasps < 1`, `point_cloud.shape` not `[N, 3]`, `scene_points.shape` not `[M, 3]`. | `{"error": "<code>", "message": "<human description>"}` |
| 413 | Total `N + M` exceeds `max_points` (default limit: 65536). | `{"error": "payload_too_large", "max_points": 65536}` |
| 422 | Any of `point_cloud` or `scene_points` contains NaN or Inf values, or `point_cloud` has fewer than `min_points` target-object points (default: 128; `scene_points` is unconstrained on the low end). | `{"error": "invalid_geometry", "message": "..."}` |
| 503 | Server starting up, checkpoint not yet loaded, or GPU unavailable. | `{"error": "not_ready", "retry_after_seconds": 5}` |
| 500 | Unhandled model exception. | `{"error": "internal_error", "message": "..."}` |

Error codes are stable machine-readable strings; messages are human-readable and may change.

---

## Auxiliary endpoints

### `GET /healthz` — liveness

- 200 `{"status": "ok"}` when the server is up and the model is loaded.
- 503 `{"status": "loading"}` during startup / checkpoint load.
- Intended for supervisor probes / Kubernetes liveness. Cheap; does not run inference.

### `GET /config` — introspection

Returns what the server is configured with. Call once at client startup.

```json
{
  "hand":               "gripper",
  "model_name":         "OURS_gripper",
  "checkpoint":         "ckpt_50000.pth",
  "joint_names":        null,
  "grasp_reference":    "tcp_between_jaws",
  "default_num_grasps": 20,
  "min_points":         128,
  "max_points":         65536,
  "server_version":     "0.1.0"
}
```

- `joint_names`: for dexterous hands, ordered list matching the per-grasp `joint_angles`. `null` for grippers.
- `grasp_reference`: string describing what point `translation` refers to on the hand (`"tcp_between_jaws"`, `"wrist_link"`, etc.). Informational.
- `min_points` / `max_points`: accepted input size bounds.

### `GET /version` — build metadata

```json
{
  "server_version": "0.1.0",
  "model_commit":   "eb3d685",
  "built_at":       "2026-04-22T10:15:00Z"
}
```

---

## Example end-to-end call from Python

```python
import base64, numpy as np, requests

def _encode(arr: np.ndarray) -> dict:
    return {
        "dtype":    str(arr.dtype),
        "shape":    list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }

# Target-object points — e.g. from your SAM segmentation.
target_pc = np.random.randn(2048, 3).astype(np.float32)

# Everything else in the scene — table, other objects, etc.
# Omit this field entirely if you only have / want the target.
scene_pc = np.random.randn(8192, 3).astype(np.float32)

body = {
    "point_cloud":  _encode(target_pc),
    "scene_points": _encode(scene_pc),   # optional; remove for "object-only" mode
    "num_grasps":   10,
}

r = requests.post("http://localhost:8000/predict", json=body, timeout=10.0)
r.raise_for_status()
data = r.json()

all_grasps = data["grasps"]
target_grasps = [g for g in all_grasps if g["object_id"] == 1]
scene_grasps  = [g for g in all_grasps if g["object_id"] == 0]

print(f"context_mode: {data['meta']['context_mode']}")
print(f"Inference: {data['meta']['inference_ms']:.1f} ms")
print(f"Total {len(all_grasps)} grasps  "
      f"(target={len(target_grasps)}, scene={len(scene_grasps)})")
if target_grasps:
    print(f"Best target grasp: "
          f"translation={target_grasps[0]['translation']}, "
          f"width={target_grasps[0]['gripper_width']}")
```

---

## ROS 2 integration notes

Conversion from `sensor_msgs/PointCloud2` to the request body:

```python
from sensor_msgs_py import point_cloud2

# Convert PointCloud2 → (N, 3) float32
xyz = point_cloud2.read_points_numpy(
    msg, field_names=("x", "y", "z"), skip_nans=True
).astype(np.float32)
```

The `frame_id` of the ROS message must match the frame you expect the returned grasps in. The server does not transform frames. Typical flow:

- Upstream perception (e.g. SAM + RGBD fusion, geometric clustering, PointNet++ segmentation) produces a per-object point cloud in `camera_depth_optical_frame`.
- Optionally transform to `base_link` using `tf2_ros.Buffer` before sending.
- Server returns grasps in that same frame.
- Node publishes the grasps as `geometry_msgs/PoseArray` with `header.frame_id` matching.

**Segmentation is done entirely client-side.** If your scene has a bowl of apples and you want to grasp one apple, publish ONE apple's point cloud as `point_cloud` per `/predict` call; put everything else (the bowl, the other apples, the table) in `scene_points`. Multiple apples = multiple calls.

Typical ROS 2 construction of `scene_points` when you already have per-object SAM masks:

```python
# `full_cloud` is (N_total, 3) float32 from PointCloud2
# `target_mask` is a bool array of shape (N_total,) marking the object you want
target_pc  = full_cloud[target_mask]
scene_pc   = full_cloud[~target_mask]
# Optionally subsample scene_pc to keep request under max_points:
if len(scene_pc) > 16000:
    idx = np.random.choice(len(scene_pc), 16000, replace=False)
    scene_pc = scene_pc[idx]
```

### Handling the response in your ROS 2 node

The response contains grasps for **both** the target and the clutter. You almost certainly want target-only. The `object_id` field on each grasp is the filter:

```python
data = response.json()
target_grasps = [g for g in data["grasps"] if g["object_id"] == 1]

# Publish as PoseArray. Keep them ordered by score (response is already sorted).
pose_array = PoseArray()
pose_array.header.frame_id = input_cloud_frame_id  # same frame as you sent
pose_array.header.stamp = self.get_clock().now().to_msg()
for g in target_grasps:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = g["translation"]
    qx, qy, qz, qw = g["rotation_quat"]
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    pose_array.poses.append(pose)
publisher.publish(pose_array)

# Gripper-width per grasp (for a parallel gripper) — publish on a side topic
widths = [g["gripper_width"] for g in target_grasps]
```

A few things to be aware of in your ROS node:

- If you see `object_id == 0` grasps in the response, that's **normal** when `scene_points` was supplied — the model reports grasps on clutter too. Filter them out unless you explicitly want to debug what the model thinks about the surrounding geometry.
- If `context_mode == "object_only"` (i.e., you did *not* send `scene_points`), every returned grasp has `object_id == 1`, so the filter is a no-op.
- `data["meta"]["request_id"]` is worth logging — it pins each grasp set to the corresponding `.usd` / `.html` / `.npz` dump on the server when `--dump-*` flags are on, which is invaluable for post-hoc cross-check against your RViz/Isaac Sim view.

If you want to also visualize the scene-context grasps in RViz for debugging (e.g., to see what the model considers graspable on the surrounding objects), publish them on a separate topic/namespace with a distinct color — do **not** feed them into your motion planner.

---

## Versioning

- This API is at version **0.1.0**. Breaking changes bump minor version until 1.0; additive changes bump patch.
- The server advertises its version in every `/predict` response and via `/version`.
- The client SHOULD log a warning if `meta.server_version` mismatches the version it was developed against.

---

## Deliberately out of scope (for now)

- **Streaming / websocket / gRPC bidirectional**: single request/response only. If throughput becomes a bottleneck (>10 Hz inference) we'll add a streaming variant.
- **Multi-hand servers**: one hand per server process. Run multiple containers if needed.
- **Segmentation / scene parsing**: the server has no object detector, no instance classifier, no foreground/background heuristic. The client produces the target/scene split; the server only acts on what it's given. (The model internally benefits from a per-point target-vs-scene mask, which is why `scene_points` exists as an input — but the *decision* of what-is-what is still the client's.)
- **Grasp filtering by accessibility / collision**: the client is responsible for checking reachability, collision with other scene elements, etc. The server only knows what was in the point cloud.
