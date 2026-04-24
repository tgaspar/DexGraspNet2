# Training

How to train a grasp-prediction model from scratch. The repo inherits the paper's diffusion + sparse-conv architecture and adds a dataclass-based configuration layer around it.

---

## Command

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  --shm-size=32g \
  -v $(pwd):/workspace \
  -v /path/to/data:/workspace/data \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e WANDB_API_KEY=your_wandb_key \
  dexgraspnet2:latest \
  bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py38 && \
    python -u -m dexgraspnet2.train \
      --config configs/network/train_dex_ours.yaml \
      --exp_name my_experiment \
      --batch_size 16 \
      --max_iter 50000 \
      --wandb"
```

> `--user $(id -u):$(id -g)` + `HOME=/tmp` runs the container as your host user so `experiments/<exp_name>/ckpt/*.pth` lands owned by you, not root.

### Outputs

- **Checkpoints**: `experiments/<exp_name>/ckpt/ckpt_*.pth`
- **Config backup**: `experiments/<exp_name>/training_config.yaml`
- **W&B dashboard**: real-time loss curves and metrics (if `--wandb` is set and `WANDB_API_KEY` is in env)

### Typical runtime

~4 hours for 50,000 iterations on a single L40S (48 GB). Runs OOM under 16 GB; either reduce `batch_size` in the YAML or train on bigger hardware.

---

## Configuration

Training is YAML-driven. Configs live in `configs/network/`:

| File | Purpose |
|---|---|
| `train_dex_ours.yaml` | Main LEAP-hand training config |
| `train_dex_isagrasp.yaml` | ISAGrasp baseline |
| `train_dex_grasptta.yaml` | GraspTTA baseline |
| `train_gripper_ours.yaml` | Parallel-jaw gripper variant |

Key parameters you'll touch:

```yaml
# configs/network/train_dex_ours.yaml
batch_size: 8           # Scenes per batch
max_iter: 50000         # Total training iterations
lr: 0.001               # Learning rate

data:
  robot: leap_hand      # Hand type (leap_hand | gripper | inspire_hand)
  num_points: 40000     # Points per scene
  voxel_size: 0.005     # Sparse-conv voxel size

model:
  type: graspness_diffusion
  backbone: sparseconv
  joint_num: 16         # LEAP hand DOF (1 for gripper, 6 for Inspire)
  trans_scale: 25       # Translation scaling factor
```

### Swapping hands

To train on a non-LEAP hand, change `data.robot` and `model.joint_num` to match. The dataset loader then pulls from `data/dex_grasps_new/scene_XXXX/<robot>/*.npz` and expects the label files to exist there. LEAP labels ship in the paper's HuggingFace release; gripper labels ship as a separate tarball; Inspire labels you generate yourself (see [`data_generation.md`](data_generation.md)).

---

## Pre-trained checkpoints

After extracting `DexGraspNet2.0-ckpts.tar` you get the following checkpoint directories under `data/DexGraspNet2.0-ckpts/`:

| Checkpoint | Description |
|---|---|
| `OURS/` | Main model — LEAP hand, 50k iterations. This is what the quick-test command uses. |
| `OURS_gripper/` | Parallel-jaw gripper variant (1 DoF = gripper width) |
| `BASELINE_ISAGrasp/` | ISAGrasp baseline |
| `BASELINE_GraspTTA/` | GraspTTA baseline |
| `ABLATION_*/` | Ablation studies: rotation representation, local features, scene composition, etc. |
| `SCALING_GRASP_*` / `SCALING_SCENE_*` | Data-scaling experiments |
| `REBUTTAL_*/` | Follow-up experiments from paper rebuttal (88 objects, friction-1, ...) |

### Checkpoint layout

Each checkpoint dir follows the convention the inference tooling expects:

```
<exp_name>/
├── config.yaml              # the training config used
└── ckpt/
    ├── ckpt_5000.pth
    ├── ckpt_10000.pth
    └── ckpt_50000.pth       # final
```

`GraspPredictor` (and therefore the FastAPI server) looks for `config.yaml` at `<ckpt_path>/../..` so this layout matters. If you rename directories, keep the two-level structure intact.

### Checkpoint `.pth` contents

```python
{
    "model":     OrderedDict(...),   # model state dict
    "optimizer": {...},              # Adam state dict
    "iter":      50000,              # iteration number
}
```

`GraspPredictor` handles both this wrapped format and a raw state-dict fallback.

---

## Troubleshooting

- **Training starts but immediately OOMs** → reduce `batch_size` in the YAML; 16 is safe on 48 GB, 8 is safe on 24 GB.
- **`MinkowskiEngine` import errors mid-training** → the Docker image pins `MinkowskiEngine==0.5.4` against CUDA 11.8 + PyTorch 2.0. Rebuilding the image against a different CUDA/PyTorch combo requires reinstalling the engine.
- **Graspness loss is NaN after step 1** → check that `data.voxel_size` matches what the dataset was preprocessed at. Mismatches produce all-zero sparse tensors.
- **W&B runs don't log** → `WANDB_API_KEY` must be passed via `-e` to `docker run` *and* `--wandb` must be on the train command. One without the other silently disables logging.
