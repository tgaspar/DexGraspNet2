# CLAUDE.md

Persistent context for this repository. See @.claude/CLAUDE.md for personal dev preferences, and @README.md for user-facing docs.

## Project summary

Docker-based fork of **DexGraspNet 2.0** (CoRL 2024): diffusion-based dexterous grasp generation from point clouds, trained and validated in Isaac Gym. Supports LEAP (16-DoF), parallel gripper, and Inspire (6-DoF) hands.

## Repository layout (what lives where)

- `dexgraspnet2/` — refactored package. **Use this for training, inference, and new features.**
- `src/` — original paper code. Kept only for inference against the provided `OURS` checkpoint.
- `configs/network/` — YAML training configs (legacy/original, consumed by both code paths).
- `dexgraspnet2/configs/hands/` — hand-specific YAML (`leap_hand.yaml`, `inspire_hand.yaml`) loaded into `HandConfig` dataclass.
- `scripts/generate_inspire_dataset.py` — physics-based grasp dataset generator (new-hand support).
- `tests/` — demos, visualizers, and validation scripts (not pytest units).
- `data/` — **symlinks only** to `/mnt/datasets/dexgraspnet2/data/`. Never commit.
- `third_party/` — vendored deps (isaacgym, MinkowskiEngine, nflows, TorchSDF, torchprimitivesdf, diffusers). Do not modify.
- `experiments/` — training outputs (`<exp_name>/ckpt/ckpt_*.pth`).

## Environment — everything runs in Docker

Host has no Python deps. Always use the `dexgraspnet2:latest` image. Data is at `/mnt/datasets/dexgraspnet2/data` on host.

Standard run template:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $(pwd):/workspace \
  -v /mnt/datasets/dexgraspnet2/data:/workspace/data \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  dexgraspnet2:latest \
  conda run -n py38 <command>
```

Why the `--user` / `HOME=/tmp` / `/workspace` dance: the base image's `/root` is mode 700, so running as non-root there fails with `PermissionError: '/root/DexGraspNet2'`. Mounting at `/workspace` (world-accessible) fixes that. `HOME=/tmp` redirects PyTorch's ninja extension cache (`gymtorch.so` build) to a path the non-root user can write. Net effect: anything the container writes into the bind mount (`experiments/`, `outputs/`, `.logs/`) is owned by your host user — no `sudo chown` needed afterwards.

For **Isaac Gym with GUI**, additional flags are mandatory (see `dexgraspnet2/run_isaac_gym.md`):

```
-e DISPLAY=:1 \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
-e __GLX_VENDOR_LIBRARY_NAME=nvidia \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /dev/dri:/dev/dri \
```

The host may need `xhost +local:docker` first. The actual `$DISPLAY` is usually `:1` here — check with `ls /tmp/.X11-unix/`.

## Code style (enforced)

- **Python 3.8** (Isaac Gym constraint). Do not use 3.9+ syntax (`X | Y`, `list[int]`, PEP 604).
- Use `logging` (via `dexgraspnet2/utils/logging.py`), **not** `print`.
- Google-style docstrings for public methods.
- Use design patterns where appropriate (e.g., `sampling_strategies.py` uses Strategy pattern — extend it rather than branching inside `GraspGenerator`).
- Keep the package structure meaningful; new generation-time code goes in `dexgraspnet2/generation/`, new inference-time in `dexgraspnet2/inference/`, simulator/Isaac-Gym code in `dexgraspnet2/simulation/` or `dexgraspnet2/generation/grasp_simulator.py`.

## Critical gotchas

### 1. `isaacgym` must be imported before `torch`
Any script touching Isaac Gym must `import isaacgym` (or `from isaacgym import ...`) **before** importing `torch`, or it will crash with opaque errors. Vulkan/GLX env vars must also be set before the import:

```python
import os
os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
from isaacgym import gymapi, gymtorch   # before torch
import torch
```

### 2. Checkpoint ↔ code compatibility
Training-time and inference-time configs must match **exactly**, or joint predictions go haywire (out-of-range values, extended fingers). The trap: `src/` reads `beta_schedule`/`clip_sample` only from `model.diffusion.scheduler`; `dexgraspnet2/` reads them from `model.diffusion` directly. Correct pairing:

| Checkpoint | Use this inference code |
|---|---|
| `data/DexGraspNet2.0-ckpts/OURS/...` | `src/` |
| `experiments/<ours>/...` (our training) | `dexgraspnet2/` |

If you see predicted joints outside `[-1.25, 2.10]` rad (LEAP), or wildly extended fingers, suspect a config mismatch.

### 3. `joint_mlp` is a `nflows.nn.nets.resnet.ResidualNet`, not a plain MLP
`ConditionalTransform` in `dexgraspnet2/models/graspness_model.py` must use `nflows` ResidualNet directly. A naive re-implementation breaks pre/post-activation ordering and silently produces bad joint angles. Don't "clean this up."

### 4. Hand TCP / sampling strategy
Inspire-hand grasp sampling is pluggable via `SamplingStrategy` (`dexgraspnet2/generation/sampling_strategies.py`). `HandConfig` fields `tcp_position`, `tcp_rotation_rpy`, `sampling_strategy`, `sampling_params`, `preshapes` drive it. Current Inspire yield on `scene_0000`/obj 14 is ~0% — tuning, not architecture, is the blocker.

### 5. Scene grasps are in scene-world frame
Grasps in `data/dex_grasps_new/scene_XXXX/<hand>/<obj_id>.npz` are relative to the scene's object placement (often on a table at z≈0.45). When re-simulating on ground, transform with `our_obj_pose @ inv(scene_obj_pose)` and filter `z > 0.02`. See `docs/grasp_validation.md`.

### 6. URDF DOF order ≠ grasp-file joint order
Isaac Gym's DOF enumeration may scramble `j0..j15`. `grasp_simulator.py` handles the remap; don't assume ordinal alignment.

## Common commands

### Training (refactored pipeline)
```bash
conda run -n py38 python -m dexgraspnet2.train \
  --config configs/network/train_dex_ours.yaml \
  --exp_name <name> --max_iter 50000 --batch_size 8 [--wandb]
```
Outputs: `experiments/<name>/{ckpt/ckpt_*.pth, training_config.yaml}`. ~4h for 50k iters on an L40S.

### Visualize predictions
```bash
conda run -n py38 python tests/visualize_dex_pred.py \
  --ckpt_path experiments/<name>/ckpt/ckpt_50000.pth \
  --scene scene_0090 --output_path outputs/vis.html
```

### Grasp validation (batch, needs X11 flags)
```bash
conda run -n py38 python -u tests/demo_batch_grasp_validation.py \
  --num-envs 32 --timeout 10
# single grasp: swap --num-envs with --grasp-idx <IDX>
```

### Generate grasps for a new hand
```bash
conda run -n py38 python scripts/generate_inspire_dataset.py \
  --scene_ids scene_0000 --num_grasps 100
```

## Verification before declaring success

- For model/inference changes: run `tests/visualize_dex_pred.py` or `tests/demo_batch_grasp_validation.py` and inspect output; don't rely on it "loading without error."
- Predicted LEAP joint ranges must fall in `[-1.25, 2.10]` rad (mean ~0.38). Outside that range → something is wrong.
- For Isaac Gym changes: run a short viewer session (`--timeout 10`) and confirm the hand is visible, not black-screened, and the object doesn't fall through the floor.

## Housekeeping

- Branch for work: `feat/refactor` is the active feature branch; `main` is the PR target.
- **Never** commit `data/`, `experiments/`, `outputs/`, `*.html`, `*.tar`, `*.tar.gz` (see `.gitignore`).
- Keep the README terse — details go here or in `docs/`.
- The many scratch `CLAUDE-REPORT.md`, `session-*.md`, `agent_isaac_eval_chkpt.md` files are historical logs, not canonical docs.

## Parked work — pick up later

- **Force-closure–based label generation** — @.claude/todos/force_closure_label_generation.md. Parked 2026-04-22 while we pivot to model-usage inside the robot application. Contains exploration notes, history of fixes, and a staged implementation plan for when we resume dataset expansion.
