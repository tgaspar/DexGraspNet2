# DexGraspNet 2.0 (Docker Fork)

A Docker-based fork of **DexGraspNet 2.0: Learning Generative Dexterous Grasping in Large-scale Synthetic Cluttered Scenes** *(CoRL 2024)*.

[Original Repository](https://github.com/PKU-EPIC/DexGraspNet2.0) | [Project Page](https://pku-epic.github.io/DexGraspNet2.0/) | [Paper](https://arxiv.org/pdf/2410.23004)

![image](./figure/teaser.png)

## Why This Fork?

The original DexGraspNet 2.0 requires a complex environment setup with specific versions of CUDA, PyTorch, MinkowskiEngine, Isaac Gym, and numerous other dependencies. This fork provides:

- **Docker-first approach** — single container with all dependencies pre-configured.
- **Refactored training pipeline** — modular, configurable, with W&B integration.
- **FastAPI inference server** — run the pretrained model behind HTTP, with optional per-request HTML / USD / NPZ debug dumps.
- **Label-generation pipeline for new hands** *(work in progress)* — Isaac Gym-based, currently wired for the Inspire hand. Yield is ~3% on easy shapes, 0% on harder ones; the paper's force-closure optimisation step hasn't been re-implemented yet. Don't rely on it for production labels — see caveat in [`docs/data_generation.md`](docs/data_generation.md).
- **Simplified entry points** — clear examples for visualization, validation, training, and inference.

> ⚠️ **This repository ships no model weights and no datasets.**
> The repo contains *code* only. Pretrained checkpoints, scenes, meshes, and labels must be downloaded separately from the paper authors' HuggingFace release — see [Full Data Setup](#full-data-setup) below. The quick test only needs the LEAP checkpoint; everything else beyond that requires the full data download.

## Repository Structure

```
DexGraspNet2/
├── dexgraspnet2/           # Refactored Python package
│   ├── configs/            # Dataclass + YAML configuration
│   ├── models/             # Model architectures (backbones, diffusion)
│   ├── generation/         # Candidate sampling + Isaac Gym validation
│   ├── inference/          # GraspPredictor (loads checkpoint, runs inference)
│   ├── training/           # Training loop, callbacks, metrics
│   └── train.py            # Training entry point
├── src/                    # Original paper codebase (preprocessing, eval, vis)
├── scripts/                # Entry-point CLIs (serve / generate / visualize)
├── tests/                  # Visualization and validation demos
├── configs/                # YAML configuration files
├── robot_models/           # URDF files for hands (LEAP, Inspire)
├── docs/                   # Topic-specific documentation (see below)
├── Dockerfile              # Docker image definition
└── data/                   # Symlinks to dataset (see Full Data Setup)
```

## Quick Start

### 1. Build Docker Image

```bash
docker build -t dexgraspnet2:latest .
```

### 2. Quick test that everything works

Before downloading multi-GB datasets, smoke-test the whole pipeline first. A bundled scene (~1.4 MB, `examples/quick_test_data/`) ships with the repo, so the only thing you need to grab is the LEAP checkpoint.

```bash
# Fetch the checkpoints tarball, extract only the LEAP files we need
mkdir -p data/DexGraspNet2.0-ckpts
cd data/DexGraspNet2.0-ckpts
wget https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/DexGraspNet2.0-ckpts.tar
tar -xf DexGraspNet2.0-ckpts.tar OURS/config.yaml OURS/ckpt/ckpt_50000.pth
cd -

mkdir -p outputs
docker run --rm --gpus all \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $(pwd):/workspace \
  -w /workspace -e PYTHONPATH=/workspace \
  dexgraspnet2:latest \
  bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py38 && \
    python scripts/quick_test.py \
      --ckpt_path data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth \
      --data_root examples/quick_test_data \
      --output_path outputs/quicktest_vis.html"
```

The script prints a `QUICK TEST PASSED` banner when done and writes `outputs/quicktest_vis.html`. Open it in any browser — you should see the scene's point cloud (blue) with the LEAP hand rendered at the top-scoring predicted grasp. First run takes a minute or two (CUDA kernels compile, checkpoint loads); subsequent runs are fast.

`--user $(id -u):$(id -g)` + `HOME=/tmp` runs the container as your host user so the output HTML lands owned by you — openable in Firefox without a trip through `sudo chown`.

What this verifies: Docker image + CUDA, MinkowskiEngine, `dexgraspnet2.inference.GraspPredictor`, checkpoint loading, diffusion sampling, Plotly HTML output. If the quick test passes, the refactored inference code path (the same one behind the FastAPI server) is live.

Common failures:
- **Checkpoint not found** → `tar -xf` didn't run or extracted elsewhere. Re-check `data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth` exists.
- **GPU out of memory** → the LEAP model loads around 2 GB; close other GPU apps or pick a smaller GPU with `--gpus device=<N>`.

## Usage

Detailed, topic-specific docs live under `docs/`:

| Doc | What's inside |
|---|---|
| [`docs/inference.md`](docs/inference.md) | **Running the pretrained model.** FastAPI server, debug dumps, one-shot prediction + visualization scripts, grasp-frame axis convention, ROS 2 integration notes. |
| [`docs/api/predict.md`](docs/api/predict.md) | **Full API schema** for `POST /predict` — request / response fields, error codes, `scene_points` + `category_sampling` + `object_id` semantics. |
| [`docs/training.md`](docs/training.md) | **Training a new model.** Command, YAML configs, pre-trained checkpoint catalog, checkpoint layout, W&B setup. |
| [`docs/data_generation.md`](docs/data_generation.md) | **Generating grasp labels for new hands.** Pipeline stages (settle → sample → validate), candidate visualizer, Isaac Gym validation, dataset inspection. Note: Inspire-hand pipeline is under active development (~3% yield — see caveat inside). |
| [`docs/grasp_validation.md`](docs/grasp_validation.md) | **Deep dive into 5-waypoint validation.** Coordinate frames, success criteria, label transform math. Reference for the label-generation and validation flows. |

## Full Data Setup

Only needed for training, label generation, or running inference against arbitrary scenes from the dataset — the quick test above doesn't need any of this.

Download the remaining assets from [HuggingFace](https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0):

```bash
# Create data directory
mkdir -p /path/to/dexgraspnet2-data
cd /path/to/dexgraspnet2-data

# Download required files
wget https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/scenes.tar.gz
wget https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/meshdata.tar.gz
wget https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/dex_grasps_new.tar.gz
wget https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/dex_graspness_new.tar.gz
wget https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/DexGraspNet2.0-ckpts.tar  # skip if you already have it from the quick test

# Extract
tar -xzf scenes.tar.gz
tar -xzf meshdata.tar.gz
tar -xzf dex_grasps_new.tar.gz
tar -xzf dex_graspness_new.tar.gz
tar -xf DexGraspNet2.0-ckpts.tar

# Create symlinks in project (skip the DexGraspNet2.0-ckpts link if you already populated that directory)
cd /path/to/DexGraspNet2
ln -s /path/to/dexgraspnet2-data/scenes data/scenes
ln -s /path/to/dexgraspnet2-data/meshdata data/meshdata
ln -s /path/to/dexgraspnet2-data/dex_grasps_new data/dex_grasps_new
ln -s /path/to/dexgraspnet2-data/dex_graspness_new data/dex_graspness_new
ln -s /path/to/dexgraspnet2-data/DexGraspNet2.0-ckpts data/DexGraspNet2.0-ckpts
```

## Supported Hands

| Hand | DOF | Config YAML | Pretrained checkpoint |
|---|---|---|---|
| LEAP Hand | 16 | `configs/network/train_dex_ours.yaml` | `OURS/` |
| Parallel Gripper | 1 | `configs/network/train_gripper_ours.yaml` | `OURS_gripper/` |

Inspire Hand support is a work in progress — see [`docs/data_generation.md`](docs/data_generation.md) for the caveat and current state.

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

Inherited from the [upstream repository](https://github.com/PKU-EPIC/DexGraspNet2.0): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Non-commercial use only; attribution required.
