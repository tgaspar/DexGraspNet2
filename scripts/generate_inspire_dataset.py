import argparse
import json
import logging
import os
import sys
from datetime import datetime

# ISAAC GYM SETUP - MUST BE FIRST
# Set environment variables for Vulkan/GLX rendering
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

# Import Isaac Gym before ANY torch imports
try:
    import isaacgym
except ImportError as e:
    print(f"Failed to import isaacgym: {e}")
    sys.exit(1)

from pathlib import Path
from tqdm import tqdm
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import generator
from dexgraspnet2.generation.grasp_generator import GraspGenerator
from dexgraspnet2.configs.hand_config import HandConfig
from dexgraspnet2.utils.scene_loader import load_scene_data


def _attach_file_log(run_dir: Path) -> None:
    """Attach a FileHandler to the root logger so all loggers write to run.log."""
    fh = logging.FileHandler(run_dir / "run.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logging.getLogger().addHandler(fh)


def _hand_config_summary(cfg: HandConfig) -> dict:
    """Serialize the subset of HandConfig fields useful for run reproducibility."""
    return {
        "name": cfg.name,
        "urdf_path": str(cfg.urdf_path),
        "num_dofs": cfg.num_dofs,
        "joint_names": cfg.joint_names,
        "joint_stiffness": cfg.joint_stiffness,
        "joint_damping": cfg.joint_damping,
        "tcp_position": cfg.tcp_position,
        "tcp_rotation_rpy": cfg.tcp_rotation_rpy,
        "sampling_strategy": cfg.sampling_strategy,
        "sampling_params": cfg.sampling_params,
        "preshapes": list(cfg.preshapes.keys()) if cfg.preshapes else [],
    }


def _label_to_jsonl_record(
    label,  # GraspLabel
    run_id: str,
    scene_id: str,
    obj_id: int,
) -> dict:
    """Flatten a GraspLabel into the JSONL schema documented in the plan."""
    traj = label.trajectory or {}
    return {
        "run_id": run_id,
        "scene_id": scene_id,
        "obj_id": f"{obj_id:03d}",
        "candidate_index": traj.get("candidate_index"),
        "preshape": traj.get("preshape"),
        "sampling_strategy": traj.get("sampling_strategy"),
        "sampled_point": traj.get("sampled_point"),
        "snapshots": traj.get("snapshots"),
        "result": {
            "is_stable": bool(label.is_stable),
            "lift_height": float(label.lift_height),
            "final_finger_joints": (
                label.joint_angles.tolist()
                if label.joint_angles is not None
                else None
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate grasp dataset for Inspire Hand"
    )
    parser.add_argument(
        "--scene_ids",
        nargs="+",
        default=["scene_0000"],
        help="List of scene IDs to process",
    )
    parser.add_argument(
        "--num_grasps", type=int, default=100, help="Number of stable grasps per object"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--headless", action="store_true", help="Run simulation headless"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/dex_grasps_new",
        help="Root directory for saving grasps",
    )
    parser.add_argument(
        "--data_root", type=str, default="data", help="Data root directory"
    )
    parser.add_argument(
        "--obj_ids",
        nargs="+",
        type=int,
        default=None,
        help="Specific object IDs to process; defaults to all objects in the scene.",
    )
    args = parser.parse_args()

    # Per-run log directory. Contains run.log (all python logging output),
    # config.json (CLI + hand config), and one trajectory_<scene>_<obj>.jsonl
    # per object processed.
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_label = f"generate_inspire_{args.scene_ids[0]}"
    run_dir = Path(".logs") / f"{run_id}_{run_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _attach_file_log(run_dir)
    logger.info(f"Run log directory: {run_dir}")

    # Initialize generator
    # Note: We initialize it once and reuse it for efficiency (simulator startup is slow)
    logger.info("Initializing GraspGenerator for Inspire Hand...")

    # Load the full hand config from YAML so TCP, preshapes, sampling strategy,
    # and per-joint stiffness/damping all reach the simulator. The hardcoded
    # HandConfig.inspire_hand() factory silently drops all of those.
    config_path = Path("dexgraspnet2/configs/hands/inspire_hand.yaml")
    try:
        hand_config = HandConfig.from_yaml(config_path)
        if not Path(hand_config.urdf_path).exists():
            logger.error(f"Inspire Hand URDF not found: {hand_config.urdf_path}")
            return
    except Exception as e:
        logger.error(f"Failed to load hand config from {config_path}: {e}")
        return

    # Snapshot CLI args + resolved hand config for reproducibility.
    with (run_dir / "config.json").open("w") as f:
        json.dump(
            {
                "run_id": run_id,
                "args": vars(args),
                "hand_config": _hand_config_summary(hand_config),
            },
            f,
            indent=2,
            default=str,
        )

    try:
        generator = GraspGenerator(
            hand_config=hand_config, device=args.device, headless=args.headless
        )
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        import traceback

        traceback.print_exc()
        return

    for scene_id in args.scene_ids:
        logger.info(f"Processing scene: {scene_id}")

        try:
            # Load scene data to identify objects
            scene_data = load_scene_data(scene_id=scene_id, data_root=args.data_root)
            objects = scene_data["objects"]

            logger.info(f"Found {len(objects)} objects in scene {scene_id}")

            # Setup output directory: data/dex_grasps_new/scene_XXXX/inspire_hand/
            output_dir = Path(args.output_root) / scene_id / "inspire_hand"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Filter by --obj_ids if the user specified a subset.
            if args.obj_ids is not None:
                wanted = set(args.obj_ids)
                objects = [o for o in objects if int(o["id"]) in wanted]
                logger.info(f"Filtered to {len(objects)} object(s): "
                            f"{[int(o['id']) for o in objects]}")

            # Process each object
            for obj_data in objects:
                obj_id = obj_data["id"]
                mesh_path = obj_data["mesh_path"]
                output_path = output_dir / f"{obj_id:03d}.npz"

                if output_path.exists():
                    logger.info(f"Grasps already exist for object {obj_id}, skipping.")
                    continue

                logger.info(f"Generating grasps for object {obj_id}...")

                try:
                    # Setup scene with just this object (canonical generation)
                    generator.setup_scene(object_meshes=[mesh_path])

                    # Generate grasps
                    # Note: These are in world frame where object is at origin (setup_scene default)
                    # We might need to adjust if setup_scene moves object
                    labels = generator.generate(num_grasps=args.num_grasps)

                    # Write per-candidate trajectory trace (includes failed
                    # candidates, which are the interesting ones for debugging).
                    traj_path = run_dir / f"trajectory_{scene_id}_{obj_id:03d}.jsonl"
                    with traj_path.open("w") as fh:
                        for lbl in generator._all_labels:
                            record = _label_to_jsonl_record(
                                lbl, run_id=run_id, scene_id=scene_id, obj_id=obj_id
                            )
                            fh.write(json.dumps(record, default=str) + "\n")
                    logger.info(
                        f"Wrote trajectory trace: {traj_path} "
                        f"({len(generator._all_labels)} candidates)"
                    )

                    if labels:
                        generator.save_labels(str(output_path))
                    else:
                        logger.warning(f"No stable grasps found for object {obj_id}")

                except Exception as e:
                    logger.error(f"Failed to generate grasps for object {obj_id}: {e}")
                    import traceback

                    traceback.print_exc()

        except Exception as e:
            logger.error(f"Failed to process scene {scene_id}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
