# DexGraspNet 2.0 (Docker Fork)

A Docker-based fork of **DexGraspNet 2.0: Learning Generative Dexterous Grasping in Large-scale Synthetic Cluttered Scenes** *(CoRL 2024)*.

[Original Repository](https://github.com/PKU-EPIC/DexGraspNet2.0) | [Project Page](https://pku-epic.github.io/DexGraspNet2.0/) | [Paper](https://arxiv.org/pdf/2410.23004)

![image](./figure/teaser.png)

## Why This Fork?

The original DexGraspNet2.0 requires a complex environment setup with specific versions of CUDA, PyTorch, MinkowskiEngine, Isaac Gym, and numerous other dependencies. This fork provides:

- **Docker-first approach**: Single container with all dependencies pre-configured
- **Refactored training pipeline**: Modular, configurable training with W&B integration
- **Vectorized simulation**: Optimized Isaac Gym batch validation (32 envs run as fast as 1)
- **Simplified entry points**: Clear examples for visualization, validation, and training

## Repository Structure

```
DexGraspNet2/
├── dexgraspnet2/           # Refactored Python package
│   ├── configs/            # Dataclass-based configuration
│   ├── models/             # Model architectures (backbones, diffusion)
│   ├── training/           # Training loop, callbacks, metrics
│   ├── simulation/         # Isaac Gym grasp validation
│   └── train.py            # Training entry point
├── src/                    # Original codebase (preprocessing, eval)
├── tests/                  # Visualization and validation demos
├── configs/                # YAML configuration files
├── robot_models/           # URDF files for hands (LEAP, Inspire, etc.)
├── Dockerfile              # Docker image definition
└── data/                   # Symlinks to dataset (see Data Setup)
```

## Quick Start

### 1. Build Docker Image

```bash
docker build -t dexgraspnet2:latest .
```

### 2. Data Setup

Download data from [HuggingFace](https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0):

```bash
# Create data directory
mkdir -p /path/to/dexgraspnet2-data

# Download required files
wget https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/scenes.tar.gz
wget https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/meshdata.tar.gz
wget https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/dex_grasps_new.tar.gz
wget https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/dex_graspness_new.tar.gz
wget https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/DexGraspNet2.0-ckpts.tar

# Extract
tar -xzf scenes.tar.gz
tar -xzf meshdata.tar.gz
tar -xzf dex_grasps_new.tar.gz
tar -xzf dex_graspness_new.tar.gz
tar -xf DexGraspNet2.0-ckpts.tar

# Create symlinks in project
ln -s /path/to/dexgraspnet2-data/scenes data/scenes
ln -s /path/to/dexgraspnet2-data/meshdata data/meshdata
ln -s /path/to/dexgraspnet2-data/dex_grasps_new data/dex_grasps_new
ln -s /path/to/dexgraspnet2-data/dex_graspness_new data/dex_graspness_new
```

## Usage

All commands run inside Docker. The general pattern:

```bash
docker run --rm --gpus all \
  -v $(pwd):/root/DexGraspNet2 \
  -v /path/to/data:/root/DexGraspNet2/data \
  -w /root/DexGraspNet2 \
  dexgraspnet2:latest \
  conda run -n py38 <command>
```

### Visualize Grasps from Dataset

Visualize dexterous hand grasps with graspness heatmap overlay:

```bash
docker run --rm --gpus all \
  -v $(pwd):/root/DexGraspNet2 \
  -v /path/to/data:/root/DexGraspNet2/data \
  -w /root/DexGraspNet2 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  dexgraspnet2:latest \
  conda run -n py38 python tests/visualize_dex_grasp.py \
    --scene scene_0001 \
    --view 0000 \
    --grasp_num 5 \
    --with_graspness True
```

Visualize scene point cloud:

```bash
docker run --rm --gpus all \
  -v $(pwd):/root/DexGraspNet2 \
  -v /path/to/data:/root/DexGraspNet2/data \
  -w /root/DexGraspNet2 \
  dexgraspnet2:latest \
  conda run -n py38 python tests/visualize_scene.py
```

### Validate Grasps in Isaac Gym Simulation

Run batch grasp validation with physics simulation:

```bash
docker run --rm --gpus all \
  --shm-size=8g \
  -v $(pwd):/root/DexGraspNet2 \
  -v /path/to/data:/root/DexGraspNet2/data \
  -w /root/DexGraspNet2 \
  -e PYTHONPATH=/root/DexGraspNet2 \
  dexgraspnet2:latest \
  conda run -n py38 python tests/demo_batch_grasp_validation.py \
    --num_envs 32 \
    --scene scene_0001 \
    --robot leap_hand
```

The validation runs a 5-stage grasp trajectory:
1. **Pregrasp**: Position hand at approach pose
2. **Approach**: Move toward object
3. **Grasp**: Close fingers
4. **Squeeze**: Apply additional force
5. **Lift**: Lift object and check success (object rises ≥3cm)

### Train Model

Train the diffusion-based grasp prediction model:

```bash
docker run --rm --gpus all \
  --shm-size=32g \
  -v $(pwd):/root/DexGraspNet2 \
  -v /path/to/data:/root/DexGraspNet2/data \
  -w /root/DexGraspNet2 \
  -e PYTHONPATH=/root/DexGraspNet2 \
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

Training outputs:
- Checkpoints: `experiments/<exp_name>/ckpt/ckpt_*.pth`
- Config backup: `experiments/<exp_name>/training_config.yaml`
- W&B dashboard: Real-time loss curves and metrics

### Run Inference

Generate grasp predictions on test scenes:

```bash
docker run --rm --gpus all \
  -v $(pwd):/root/DexGraspNet2 \
  -v /path/to/data:/root/DexGraspNet2/data \
  -w /root/DexGraspNet2 \
  dexgraspnet2:latest \
  conda run -n py38 python src/eval/predict_dexterous_all_cates.py \
    --ckpt experiments/my_experiment/ckpt/ckpt_50000.pth
```

Visualize predictions:

```bash
docker run --rm --gpus all \
  -v $(pwd):/root/DexGraspNet2 \
  -v /path/to/data:/root/DexGraspNet2/data \
  -w /root/DexGraspNet2 \
  dexgraspnet2:latest \
  conda run -n py38 python tests/visualize_dex_pred.py \
    --ckpt_path experiments/my_experiment/ckpt/ckpt_50000.pth
```

## Pre-trained Checkpoints

After extracting `DexGraspNet2.0-ckpts.tar`, available checkpoints include:

| Checkpoint | Description |
|------------|-------------|
| `OURS/` | Main model (LEAP hand, 50k iterations) |
| `OURS_gripper/` | Parallel-jaw gripper variant |
| `BASELINE_ISAGrasp/` | ISAGrasp baseline |
| `ABLATION_*/` | Ablation studies (rotation representations, etc.) |
| `SCALING_*/` | Data scaling experiments |

## Configuration

Training is configured via YAML files in `configs/network/`. Key parameters:

```yaml
# configs/network/train_dex_ours.yaml
batch_size: 8           # Scenes per batch
max_iter: 50000         # Total training iterations
lr: 0.001               # Learning rate

data:
  robot: leap_hand      # Hand type
  num_points: 40000     # Points per scene
  voxel_size: 0.005     # Sparse convolution voxel size

model:
  type: graspness_diffusion
  backbone: sparseconv
  joint_num: 16         # LEAP hand DOF
  trans_scale: 25       # Translation scaling factor
```

## Supported Hands

| Hand | DOF | Config |
|------|-----|--------|
| LEAP Hand | 16 | `robot: leap_hand` |
| Parallel Gripper | 1 | `robot: gripper` |
| Inspire Hand | 6 | `robot: inspire_hand` (requires grasp generation) |

## Citation

```bibtex
@inproceedings{zhang2024dexgraspnet,
  title={DexGraspNet 2.0: Learning Generative Dexterous Grasping in Large-scale Synthetic Cluttered Scenes},
  author={Zhang, Jialiang and Liu, Haoran and Li, Danshi and Yu, XinQiang and Geng, Haoran and Ding, Yufei and Chen, Jiayi and Wang, He},
  booktitle={8th Annual Conference on Robot Learning},
  year={2024}
}
```

## License

This work and the dataset are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)
