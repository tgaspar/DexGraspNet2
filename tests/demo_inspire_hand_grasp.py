#!/usr/bin/env python3
"""
Demo: Inspire Hand grasp simulation with virtual 6-DOF joint chain.

Demonstrates the grasp simulator with the Inspire Hand (6 DoF) grasping a box.
Uses the "_free" URDF variant with virtual 6-DOF joints for floating base control.
"""

import os
import sys
from pathlib import Path

# Set environment variables FIRST, before any imports
os.environ['VK_ICD_FILENAMES'] = '/etc/vulkan/icd.d/nvidia_icd.json'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

# Add project root to path BEFORE importing isaacgym
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# IMPORTANT: Import isaacgym BEFORE torch (Isaac Gym requirement)
from isaacgym import gymapi, gymtorch
import torch

# Now import other dependencies
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules AFTER isaacgym
from dexgraspnet2.configs.hand_config import HandConfig
from dexgraspnet2.generation.grasp_simulator import GraspSimulator, SimulationConfig

print("=" * 60, flush=True)
print("INSPIRE HAND GRASP SIMULATION DEMO", flush=True)
print("Using _free URDF with virtual 6-DOF joint chain", flush=True)
print("=" * 60, flush=True)


def main():
    # Create Inspire Hand configuration
    hand_config = HandConfig.inspire_hand()
    print(f"Hand: {hand_config.name} ({hand_config.num_dofs} DoF)", flush=True)
    print(f"Joints: {hand_config.joint_names}", flush=True)

    # Simulation config with 5-waypoint system
    sim_config = SimulationConfig(
        pregrasp_steps=30,
        approach_steps=60,
        grasp_steps=60,
        squeeze_steps=30,
        lift_steps=90,
        pregrasp_distance=0.10,
        lift_height=0.10,
        success_threshold=0.03,
    )

    # Create simulator (with viewer for visualization)
    print("\nInitializing simulation...", flush=True)
    simulator = GraspSimulator(
        hand_config=hand_config,
        device="cuda:0",
        headless=False,  # Show GUI
        config=sim_config,
    )

    # Setup scene with a box object
    print("Setting up scene...", flush=True)
    simulator.setup(
        object_mesh_path=None,  # Use default box
        num_envs=1,
    )

    # Define a test grasp
    # The Inspire Hand's palm faces -Y in its local frame (based on URDF orientation)
    # Object (5cm box) is at z=0.05 (center), so top is at z=0.075

    # Grasp from above - palm facing down
    # Translation: above the object center
    translation = np.array([0.0, 0.0, 0.10])  # 10cm above ground

    # Rotation: palm facing down (-Z), fingers pointing along +X
    # This rotation aligns the hand to approach from above
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])

    # Joint angles: close the fingers
    # Inspire hand joints: thumb_yaw, thumb_pitch, index, middle, ring, pinky
    # Higher values = more closed
    joint_angles = np.array([0.8, 0.5, 1.2, 1.2, 1.2, 1.2])

    print("\nTesting grasp:", flush=True)
    print(f"  Translation: {translation}", flush=True)
    print(f"  Joint angles: {joint_angles}", flush=True)

    # Validate the grasp
    print("\nExecuting grasp sequence (5 waypoints):", flush=True)
    print("  1. Pregrasp (fingers open, 10cm back)", flush=True)
    print("  2. Cover (approach to grasp position)", flush=True)
    print("  3. Grasp (close fingers)", flush=True)
    print("  4. Squeeze (additional grip)", flush=True)
    print("  5. Lift (move up, check if object follows)", flush=True)

    is_stable, final_height = simulator.validate_single_grasp(
        translation=translation,
        rotation=rotation,
        joint_angles=joint_angles,
        visualize=True,
    )

    print("\n" + "=" * 60, flush=True)
    print("RESULT:", flush=True)
    print(f"  Grasp stable: {is_stable}", flush=True)
    print(f"  Final object height: {final_height:.4f} m", flush=True)
    print(f"  Height gained: {final_height - 0.05:.4f} m", flush=True)
    print("=" * 60, flush=True)

    # Let user observe the result
    print("\nPress Ctrl+C or close the viewer to exit...", flush=True)
    try:
        while True:
            simulator.step(render=True)
    except KeyboardInterrupt:
        pass

    # Cleanup
    print("\nCleaning up...", flush=True)
    simulator.cleanup()
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
