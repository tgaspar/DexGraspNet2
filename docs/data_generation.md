# Label generation for new hands

Generate grasp-label datasets for hands the paper didn't ship labels for (e.g. Inspire). The pipeline: sample candidate hand poses with a heuristic, simulate each one in Isaac Gym through a 5-waypoint trajectory, keep the ones that successfully lift the object.

---

> ⚠️ **Work in progress — not a reliable pipeline yet.**
> Current Inspire-hand yield averages ~3% stable grasps per candidate (10% on friendly shapes, 0% on several harder ones). The paper's own pipeline sits on top of a **force-closure optimization** step that refines candidates before physics validation — that code was never open-sourced, so we're running with heuristic sampling (surface-normal / dome) only. Next step is implementing the force-closure refinement; see `.claude/todos/force_closure_label_generation.md` for the parked plan.

---

## Generate labels

Runs headless by default. Uses the first scene's objects as the grasp targets. Outputs `.npz` per object under `data/dex_grasps_new/scene_XXXX/<hand>/<obj_id>.npz`.

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $(pwd):/workspace \
  -v /path/to/data:/workspace/data \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  dexgraspnet2:latest \
  bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py38 && \
    python scripts/generate_inspire_dataset.py \
      --scene_ids scene_0000 \
      --num_grasps 100 \
      --headless"
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--scene_ids` | One or more scene IDs to process. Default: `scene_0000`. |
| `--obj_ids` | Subset of object IDs within the scene. Default: all objects. |
| `--num_grasps` | Target stable-grasp count per object (the sampler generates 10× this and filters). |
| `--headless` | Headless physics. Omit to open the Isaac Gym viewer and watch the candidates execute. |
| `--output_root` | Where to write `.npz` files. Default: `data/dex_grasps_new`. |

### Pipeline stages (what the script actually does)

1. **Load scene meta** — reads `scenes/<scene>/realsense/annotations/*.xml` for object identities and per-object mesh paths.
2. **Compute per-mesh spawn z** — for each object mesh, solve `spawn_z = -mesh.bounds[0][2] + 0.005` so the mesh's lowest vertex sits 5 mm above the ground plane. Without this, meshes whose local origin isn't at their geometric bottom spawn partially inside the ground and PhysX violently resolves the intersection.
3. **Settle** — spawn the object, park the hand at z = 0.5 m (out of the way), step physics for 60 ticks under gravity. The settled pose becomes the authoritative target for sampling.
4. **Sample candidates** — `SurfaceNormalStrategy` or `DomeSamplingStrategy` (picked via `sampling_strategy` in the hand YAML) produces candidate TCP poses + closing preshape.
5. **Validate** — for each batch, teleport the hand to the pregrasp pose, then execute the 5-waypoint trajectory (pregrasp → approach → close fingers → squeeze → lift). Keep candidates where the object lifts ≥ 3 cm.
6. **Persist** — write stable grasps to `.npz`, plus (when `--dump-*` flags are on) per-candidate trajectory traces.

### Per-run logs

Every invocation creates `.logs/<timestamp>_generate_inspire_<scene>/` with:
- `run.log` — full Python logging output
- `config.json` — CLI args + resolved hand config
- `trajectory_<scene>_<obj>.jsonl` — per-candidate 5-waypoint pose trace (hand + object at each milestone), useful for debugging why a particular candidate failed

See [`grasp_validation.md`](grasp_validation.md) for the deeper story on the 5-waypoint protocol, coordinate frames, and validation semantics.

---

## Visualize a single candidate before running the full generator

`scripts/visualize_grasp_candidate.py` writes a USD file showing the object + one sampled candidate's hand pose. Opens in Isaac Sim. Useful for verifying the TCP calibration and approach direction before spending compute on 500 candidates.

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $(pwd):/workspace \
  -v /path/to/data:/workspace/data \
  -w /workspace -e PYTHONPATH=/workspace \
  dexgraspnet2:latest \
  bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py38 && \
    python scripts/visualize_grasp_candidate.py \
      --hand dexgraspnet2/configs/hands/inspire_hand.yaml \
      --scene_id scene_0000 --obj_id 14 \
      --strategy normals --preshape power --seed 42"
```

Writes `.logs/visualize/<scene>_<obj>_<strategy>_<preshape>_seed<N>.usd`. The USD contains:
- Object mesh at spawn pose
- 2 mm thin cylinder along the surface normal at the sampled grasp point
- Hand at pregrasp pose (via `yourdfpy` FK, pre-applied preshape)
- X/Y/Z axis gizmos at world origin, wrist, and TCP

The TCP gizmo is parented under the hand — grabbing and moving the hand in Isaac Sim carries the TCP with it, which is the workflow we use for calibrating `tcp_position` / `tcp_rotation_rpy` in `dexgraspnet2/configs/hands/<hand>.yaml`.

---

## Validate existing labels in Isaac Gym

Separate from generation — given existing `.npz` labels (e.g. the LEAP labels from the paper's HuggingFace release), run the physics validation on them and report stable/total.

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  --shm-size=8g \
  -v $(pwd):/workspace \
  -v /path/to/data:/workspace/data \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  dexgraspnet2:latest \
  conda run -n py38 python tests/demo_batch_grasp_validation.py \
    --num-envs 32 \
    --scene-id 1 \
    --obj-id 0
```

5-waypoint sequence:
1. **Pregrasp** — position hand at approach pose
2. **Approach** — interpolate to grasp position
3. **Grasp** — close fingers
4. **Squeeze** — extra finger force
5. **Lift** — move up; success if object rises ≥ 3 cm

Deeper walkthrough (coordinate transforms, success criteria, success-rate expectations on paper labels) lives in [`grasp_validation.md`](grasp_validation.md).

---

## Inspect the dataset itself

**Visualize a scene's point cloud**:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $(pwd):/workspace \
  -v /path/to/data:/workspace/data \
  -w /workspace \
  dexgraspnet2:latest \
  conda run -n py38 python tests/visualize_scene.py --scene_id 0000
```

**Visualize grasps in the shipped dataset** (LEAP hand, with optional graspness heatmap overlay):

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $(pwd):/workspace \
  -v /path/to/data:/workspace/data \
  -w /workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  dexgraspnet2:latest \
  conda run -n py38 python tests/visualize_dex_grasp.py \
    --scene scene_0001 \
    --view 0000 \
    --grasp_num 5 \
    --with_graspness True
```

Both scripts use the paper's `src/utils/vis_plotly.Vis` helper and write interactive Plotly HTML.
