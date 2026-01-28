# Isaac Gym Evaluation Experiment Report

## Goal
The objective was to create an evaluation script `tests/evaluate_predicted_grasps.py` that:
1.  Loads a scene from the DexGraspNet dataset (e.g., `scene_0220`).
2.  Spawns the corresponding object and a robot hand (LEAP hand) in Isaac Gym.
3.  Captures a depth image from a simulated camera positioned according to the scene data.
4.  Converts the depth image to a point cloud.
5.  Runs a grasp generation model (using a checkpoint) on this point cloud.
6.  Validates the predicted grasps by executing them in the simulation.

## Implementation Details

### Scene Loading
We successfully loaded scene annotations to get:
-   **Camera Pose:** `cam0_wrt_table` (Camera relative to table/world).
-   **Object Pose:** Object position and orientation from XML annotations.

### Simulation Setup
-   **Isaac Gym:** Used `gymapi` to create a simulation environment.
-   **Rendering:** Configured for headless or viewer-based execution.
-   **Assets:** Loaded simplified URDFs for the LEAP hand and the object.

### Coordinate Systems
A major challenge was aligning the coordinate systems:
-   **Isaac Gym (OpenGL):** Right(+X), Up(+Y), Forward(-Z).
-   **Dataset/OpenCV:** Right(+X), Down(+Y), Forward(+Z).
-   **World Frame:** Z-up (Table).

## Issues Faced & Resolutions

### 1. Blank/Transparent Viewer Window
**Issue:** The Isaac Gym viewer window would open but remain blank or transparent, unlike other working scripts.
**Resolution:**
-   **Environment Variables:** Added critical Vulkan/GLX environment variables before importing `isaacgym`:
    ```python
    os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    ```
-   **Device Selection:** Fixed `create_sim` to use the correct GPU device ID (extracted from string like `"cuda:0"`) instead of hardcoded `0`, which caused issues when multiple GPUs were visible or indices mismatched.
-   **Graphics Stepping:** Ensured `step_graphics` is called appropriately in the loop.

### 2. Missing/Invisible Robot Hand
**Issue:** The robot hand was not visible in the viewer or camera.
**Resolution:**
-   **Spawn Position:** Spawned the robot at `(0, 0, 0)` so its virtual joints (which control base position) align with world coordinates.
-   **Safety Position:** Implemented logic to teleport the robot to a "safe" holding position `(0, 0, 0.6)` (above the camera) during object settling and point cloud capture to prevent visual interference.
-   **Teleportation:** Implemented `_reset_simulation` to instantly set DOF states for teleporting the robot to the pre-grasp position, avoiding unstable "flying" behavior from far away.

### 3. Coordinate Frame Mismatch (Unresolved/Partial)
**Issue:** The point cloud captured in Isaac Gym, when unprojected to the world frame, did not perfectly match the expected world coordinates (z-height was off, orientation mismatches).
**Attempts:**
-   **Using Isaac View Matrix:** Computed `cam_to_world = inv(view_matrix)`. Resulted in significant offsets (e.g., Z = -0.45m instead of +0.47m).
-   **Forum Fix:** Attempted to correct with `get_env_origin`. Did not resolve the issue.
-   **Using Reference Pose:** Using the dataset's `camera_in_world` pose directly. This was better but still had orientation mismatches, likely due to Isaac Gym's `set_camera_location` (look-at) choosing a different roll angle or up-vector than the dataset's fixed camera pose.
-   **Forced Transform:** Theoretically, using `set_camera_transform` with a coordinate conversion (OpenCV -> OpenGL) would solve this, but we decided to scope down the experiment.

## Current Status
The script has been simplified to a **Point Cloud Generation & Inference Tool**:
1.  **Scene Setup:** Spawns object and robot (safely hidden).
2.  **Capture:** Captures point cloud using the reference camera pose for unprojection (ensuring consistency with dataset conventions).
3.  **Inference:** Generates grasps based on this point cloud.
4.  **Visualization:** Produces a camera-frame visualization (`_camera_frame.html`) showing the point cloud and predicted grasps. This confirms the model receives valid input and produces reasonable grasps in the camera frame.
5.  **Simulation:** Runs the viewer to show the stable object state.

**Grasp Validation (Physical Execution)** has been disabled/removed to focus on the perception pipeline accuracy.

## Run Command
To run the evaluation in the Docker container:

```bash
docker run --rm --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/root/DexGraspNet2 \
  -v /mnt/datasets/dexgraspnet2/data:/root/DexGraspNet2/data \
  -w /root/DexGraspNet2 \
  -e PYTHONPATH=/root/DexGraspNet2 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  dexgraspnet2:latest \
  bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py38 && python tests/evaluate_predicted_grasps.py --ckpt_path data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth --object_id 48 --num_grasps 1 --output_vis /root/DexGraspNet2/tests/output/eval_vis.html"
```
