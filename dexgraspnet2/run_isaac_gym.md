# Running Isaac Gym with GUI in Docker

This document describes how to run Isaac Gym simulations with GPU-accelerated physics and proper GUI rendering inside the DexGraspNet2 Docker container.

## Quick Start

```bash
docker run --rm --gpus all \
  -e DISPLAY=:1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/dri:/dev/dri \
  -v $(pwd):/root/DexGraspNet2 \
  -v /mnt/datasets/DexGraspNet2.0-data:/root/DexGraspNet2/data \
  -w /root/DexGraspNet2 dexgraspnet2 \
  bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py38 && python -u tests/demo_grasp_verification.py"
```

## Docker Flags Explained

| Flag | Purpose |
|------|---------|
| `--gpus all` | Enable GPU access for CUDA/PhysX |
| `-e DISPLAY=:1` | X11 display for GUI window |
| `-e NVIDIA_DRIVER_CAPABILITIES=all` | Enable all NVIDIA capabilities (compute, graphics, display) |
| `-v /tmp/.X11-unix:/tmp/.X11-unix` | Mount X11 socket for window display |
| `-v /dev/dri:/dev/dri` | Direct Rendering Infrastructure for GPU graphics |

## Finding Your Display

The display value (`:1`) may vary by system. To find yours:

```bash
ls -la /tmp/.X11-unix/
```

If you see `X0`, use `DISPLAY=:0`. If you see `X1`, use `DISPLAY=:1`.

## Troubleshooting

### Window opens but shows content from behind (transparency glitch)

**Cause:** Missing GPU rendering capabilities or DRI access.

**Fix:** Ensure these flags are present:
- `-e NVIDIA_DRIVER_CAPABILITIES=all`
- `-v /dev/dri:/dev/dri`
- Correct `DISPLAY` value

### Viewer creation fails (GLFW initialization failed)

**Cause:** Missing graphics libraries or X11 connection issues.

**Fix:** The Dockerfile includes required libraries:
```
libglfw3 libglfw3-dev libxcursor1 libxinerama1 libxi6 libxrandr2
libegl1 libglvnd0 libgl1-mesa-glx
```

### GPU pipeline tensor API errors

**Cause:** Using per-actor APIs after simulation starts with GPU pipeline.

**Fix:** Use tensor APIs instead:
```python
# Instead of:
gym.get_actor_rigid_body_states(env, handle, ...)
gym.set_actor_dof_position_targets(env, handle, targets)

# Use:
root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
gym.set_actor_root_state_tensor_indexed(sim, ...)
gym.set_dof_position_target_tensor(sim, ...)
```

## Verifying GPU Rendering

Check that direct rendering is working:

```bash
docker run --rm --gpus all -e DISPLAY=:1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/dri:/dev/dri \
  dexgraspnet2 bash -c 'glxinfo | grep -E "direct rendering|OpenGL renderer"'
```

Expected output:
```
direct rendering: Yes
OpenGL renderer string: NVIDIA L40S/PCIe/SSE2
```

## Python Script Requirements

For Isaac Gym scripts to work with GPU pipeline and GUI:

```python
import os
# Set BEFORE importing isaacgym
os.environ['VK_ICD_FILENAMES'] = '/etc/vulkan/icd.d/nvidia_icd.json'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

from isaacgym import gymapi, gymtorch
import torch
```

## Available Demos

| Demo | Description |
|------|-------------|
| `tests/demo_isaac_gym_gui.py` | Simple box falling (minimal test) |
| `tests/demo_grasp_verification.py` | LEAP hand grasping a cube |
| `tests/demo_inspire_hand_grasp.py` | Inspire Hand (6 DoF) with 5-waypoint grasp |
| `tests/demo_batch_grasp_validation.py` | **16 LEAP hands validating grasps in parallel** |

## Inspire Hand Demo

The Inspire Hand demo uses the `_free` URDF variant with a virtual 6-DOF joint chain
for floating base control (following DexGraspNet2.0 paper approach).

```bash
docker run --rm --gpus all \
  -e DISPLAY=:1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/dri:/dev/dri \
  -v $(pwd):/root/DexGraspNet2 \
  -v /mnt/datasets/DexGraspNet2.0-data:/root/DexGraspNet2/data \
  -w /root/DexGraspNet2 dexgraspnet2 \
  bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py38 && python -u tests/demo_inspire_hand_grasp.py"
```

## Batch Grasp Validation Demo (Label Generation)

This demo shows how the label generation pipeline works: 16 LEAP hands validate
pre-generated grasps in parallel. Uses real grasp candidates from the dataset.

```bash
docker run --rm --gpus all \
  -e DISPLAY=:1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/dri:/dev/dri \
  -v $(pwd):/root/DexGraspNet2 \
  -v /mnt/datasets/DexGraspNet2.0-data:/root/DexGraspNet2/data \
  -w /root/DexGraspNet2 dexgraspnet2 \
  bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py38 && python -u tests/demo_batch_grasp_validation.py"
```

You should see a 4x4 grid of LEAP hands, each attempting to grasp the same object
with different grasp poses. The output shows which grasps succeeded vs failed.
