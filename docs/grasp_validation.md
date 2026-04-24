# Grasp Validation in Isaac Gym

This document explains how grasp validation works in the DexGraspNet2 project, including data sources, coordinate transformations, and how to run your own validations.

## Overview

The grasp validation pipeline tests pre-generated grasp candidates by simulating them in Isaac Gym. Each grasp is executed through 5 phases: settle → pregrasp → approach → grasp → squeeze → lift. Success is determined by whether the object is lifted off the ground while maintaining finger contact.

## Data Sources

### Directory Structure

```
data/
├── meshdata/                    # Object meshes
│   └── {obj_id}/               # e.g., 000, 001, ...
│       ├── simplified.obj       # Visual/collision mesh
│       ├── nontextured_simplified.urdf  # Physics URDF
│       └── surface_points_1000.npy
│
├── scenes/                      # Scene configurations
│   └── scene_{xxxx}/           # e.g., scene_0001
│       └── kinect/
│           └── annotations/
│               └── 0000.xml    # Object poses in scene
│
└── dex_grasps_new/             # Generated grasps
    └── scene_{xxxx}/
        └── leap_hand/
            └── {obj_id}.npz    # Grasps for object in scene
```

### Grasp Data Format (`{obj_id}.npz`)

Each grasp file contains arrays for N grasps:

| Key | Shape | Description |
|-----|-------|-------------|
| `translation` | (N, 3) | Wrist position in **scene world frame** |
| `rotation` | (N, 3, 3) | Wrist rotation matrix |
| `j0` - `j15` | (N,) | Joint angles for 16 finger joints |
| `point` | (N, 3) | Contact point on object surface |

### Scene Annotations (`0000.xml`)

XML file containing object poses within the scene:

```xml
<scene>
  <obj>
    <obj_id>0</obj_id>
    <pos_in_world>-0.0539 0.1199 0.4524</pos_in_world>
    <ori_in_world>0.26 0.33 0.56 0.72</ori_in_world>  <!-- wxyz quaternion -->
  </obj>
  ...
</scene>
```

## Selecting Scene and Object

In `tests/demo_batch_grasp_validation.py`, modify these paths:

```python
# Line ~701-705
scene_dir = Path("data/scenes/scene_0001")      # Change scene here
obj_id = 0                                       # Object ID within scene
grasp_path = Path(f"data/dex_grasps_new/scene_0001/leap_hand/{obj_id:03d}.npz")
object_urdf = Path(f"data/meshdata/{obj_id:03d}/nontextured_simplified.urdf")
object_mesh = Path(f"data/meshdata/{obj_id:03d}/simplified.obj")
```

To find available scenes and objects:

```bash
# List all scenes
ls data/scenes/

# List objects with grasps in a scene
ls data/dex_grasps_new/scene_0001/leap_hand/

# Check object IDs in scene annotation
cat data/scenes/scene_0001/kinect/annotations/0000.xml | grep obj_id
```

## Coordinate Transformations

### The Problem

Grasps are stored in the **scene world frame**, designed for where the object was originally placed (e.g., on a table at z=0.45). When we spawn the object at a different location (e.g., on ground at z=0.1), we must transform the grasps accordingly.

### Step 1: Load Scene Object Pose

```python
# From demo_batch_grasp_validation.py:load_object_pose_from_scene()
def load_object_pose_from_scene(scene_dir: Path, obj_id: int) -> np.ndarray:
    """Returns 4x4 pose matrix from scene annotations."""
    ann_file = scene_dir / "kinect/annotations/0000.xml"
    # Parse XML to get pos_in_world and ori_in_world
    # Convert wxyz quaternion to rotation matrix
    # Return 4x4 homogeneous transform
```

Example scene object pose for scene_0001, object 0:
```
Position: (-0.054, 0.12, 0.45)
Rotation: Non-trivial rotation (object tilted on table)
```

### Step 2: Calculate Spawn Height

```python
# From demo_batch_grasp_validation.py:calculate_spawn_height()
def calculate_spawn_height(mesh_path, rotation_matrix, margin=0.005):
    """
    Calculate z-height so rotated mesh rests on ground (z=0).

    1. Load mesh vertices
    2. Apply rotation to vertices
    3. Find minimum z of rotated mesh
    4. Return: margin - min_z (so bottom is at z=margin)
    """
```

This ensures the object spawns just above ground with correct orientation.

### Step 3: Define Our Object Pose

```python
# Our object: same rotation as scene, positioned above ground
our_obj_pose = np.eye(4)
our_obj_pose[:3, :3] = scene_obj_pose[:3, :3]  # Same rotation
our_obj_pose[:3, 3] = [0.0, 0.0, spawn_z]      # At calculated height
```

### Step 4: Transform Grasps

```python
# From demo_batch_grasp_validation.py:transform_grasps()
def transform_grasps(translations, rotations, scene_obj_pose, our_obj_pose):
    """
    Transform grasps from scene frame to our object frame.

    Mathematical relationship:
        grasp_our = our_obj_pose @ scene_obj_pose^(-1) @ grasp_scene

    For translations:
        new_trans = R_transform @ old_trans + p_transform

    For rotations:
        new_rot = R_transform @ old_rot
    """
```

**Key insight**: When rotations match (`R_our = R_scene`), the transform simplifies to a translation:
```
new_position = old_position + (our_position - scene_position)
```

### Step 5: Filter Valid Grasps

Grasps designed for an object on a table (z~0.45) may end up below ground when the object is at z~0.1. We filter to keep only grasps with `transformed_z > 0.02`:

```python
valid_mask = transformed_trans[:, 2] > 0.02  # Above ground
valid_indices = np.where(valid_mask)[0]
# Only ~3% of grasps are valid (5776 / 182787)
```

## Approach Vector Calculation

The approach direction is derived from the grasp rotation matrix:

```python
# In validate_grasps() method, line ~452
approach_dir = -rotations[i][:, 2]  # Negative Z-axis of hand frame
```

The LEAP hand convention: the palm faces along +Z, so approaching means moving in -Z direction.

### Pregrasp Position

```python
pregrasp_pos = grasp_translation - approach_dir * pregrasp_distance
# pregrasp_distance = 0.10m (10cm back from grasp)
```

### Grasp Trajectory

1. **Pregrasp**: 10cm back from grasp position, fingers open
2. **Approach**: Linear interpolation to grasp position (60 steps)
3. **Grasp**: Close fingers to target angles (60 steps)
4. **Squeeze**: Tighten fingers by 20% (30 steps)
5. **Lift**: Move hand up by 10cm (90 steps)

## Running Validation

### Basic Command

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  -e DISPLAY=:1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/dri:/dev/dri \
  -v $(pwd):/workspace \
  -v /mnt/datasets/DexGraspNet2.0-data:/workspace/data \
  -w /workspace dexgraspnet2 \
  bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py38 && \
    python -u tests/demo_batch_grasp_validation.py --num-envs 16 --timeout 10"
```

### Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--num-envs N` | Number of parallel environments (grasps to test) |
| `--timeout T` | Viewer timeout in seconds (None = interactive) |
| `--grasp-idx I` | Test specific grasp index |

### Success Criteria

A grasp succeeds if ALL conditions are met:
1. **Object lifted**: `final_height - before_lift_height > 1.5cm`
2. **Hand lifted**: `final_hand_height - before_hand_height > 5cm`
3. **Has contact**: At least 2 finger links in contact with object

## Visualizing Scenes

### Method 1: GraspNet API Visualization

```python
from graspnetAPI import GraspNet

# Load the dataset
g = GraspNet(root='data', camera='kinect', split='all')

# Visualize scene with objects
g.showObjGrasp(sceneId=1, camera='kinect', objId=0, numGrasp=10)
```

### Method 2: Open3D Point Cloud

```python
import open3d as o3d
import numpy as np

# Load scene point cloud
scene_dir = "data/scenes/scene_0001/kinect"
points = np.load(f"{scene_dir}/points/0000.npy")  # If available

# Or load from depth image
depth = np.load(f"{scene_dir}/depth/0000.npy")
# Convert depth to point cloud using camera intrinsics

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
```

### Method 3: Visualize Object Mesh with Grasps

```python
import trimesh
import numpy as np

# Load object mesh
mesh = trimesh.load("data/meshdata/000/simplified.obj")

# Load scene object pose
import xml.etree.ElementTree as ET
tree = ET.parse("data/scenes/scene_0001/kinect/annotations/0000.xml")
# ... parse pose ...

# Apply transform to mesh
mesh.apply_transform(scene_obj_pose)

# Load grasps
grasps = np.load("data/dex_grasps_new/scene_0001/leap_hand/000.npz")

# Visualize (add grasp frames as coordinate axes)
scene = trimesh.Scene([mesh])
for i in range(min(10, len(grasps['translation']))):
    frame = trimesh.creation.axis(origin_size=0.01, transform=...)
    scene.add_geometry(frame)
scene.show()
```

### Method 4: Isaac Gym Viewer

The validation script opens an Isaac Gym viewer window showing:
- Ground plane
- Object (with physics)
- 16 LEAP hands in a 4x4 grid

Controls:
- **Mouse drag**: Rotate view
- **Scroll**: Zoom
- **W/A/S/D**: Pan camera
- **ESC**: Close viewer

## Debugging Tips

### Object Falls Through Ground

Check spawn height calculation:
```python
print(f"Spawn height: {spawn_z}")
print(f"Mesh min z (rotated): {rotated_vertices[:, 2].min()}")
```

### Hands Start Inside Object

The pregrasp position may be wrong. Check:
```python
print(f"Grasp pos: {grasp_translation}")
print(f"Approach dir: {approach_dir}")
print(f"Pregrasp pos: {pregrasp_pos}")
```

### All Grasps Below Ground

The scene object was on a table. Either:
1. Spawn object higher (float above ground)
2. Filter grasps with `transformed_z > threshold`
3. Use different scene with ground-level object

### Joint Mapping Issues

URDF DOF order may differ from grasp data order. Check:
```python
print(f"URDF DOF names: {dof_names}")
print(f"Grasp to DOF map: {grasp_to_dof_map}")
```

## File Reference

| File | Purpose |
|------|---------|
| `tests/demo_batch_grasp_validation.py` | Main validation script |
| `robot_models/urdf/leap_hand_simplified_free.urdf` | LEAP hand with virtual 6-DOF base |
| `dexgraspnet2/utils/geometric_conversions.py` | Rotation/quaternion utilities |
| `dexgraspnet2/run_isaac_gym.md` | Docker run instructions |

## Known Limitations

1. **~3% valid grasps**: Most grasps in the dataset approach from below (designed for tabletop). Only grasps approaching from above/side work when object is on ground.

2. **Settling offset**: Objects drop ~2-3cm during physics settling, requiring adjustment of grasp targets.

3. **Joint order mismatch**: URDF loads joints in different order than grasp data (j1, j0, j2... vs j0, j1, j2...). Mapping is handled automatically.
