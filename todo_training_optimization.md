# Training Optimization TODO

## Current Bottleneck Analysis

**Date**: 2026-01-26
**Training**: LEAP hand baseline (50k iterations)

### Resource Utilization
| Resource | Usage | Bottleneck? |
|----------|-------|-------------|
| GPU VRAM | 7GB / 46GB (15%) | No |
| GPU Compute | 42-72% | No - waiting for data |
| CPU | 8 workers × 100% | **Yes** |
| Disk I/O | 0.5% wait | No |
| RAM | 34GB / 127GB | No |

### Per-Sample I/O Operations
| File | Size | Operation |
|------|------|-----------|
| `depth.png` | ~1MB | Image.open() |
| `label.png` | ~100KB | Image.open() |
| `meta.mat` | ~10KB | scipy.io.loadmat() |
| `camera_poses.npy` | ~50KB | np.load() |
| `graspness.npy` | ~3MB | np.load() |
| `*.npz` (grasps) | ~5MB | Multiple np.load() |
| **Total** | **~10MB** | **6-10 file reads** |

---

## Optimization Options

### Option 1: Increase DataLoader Prefetch (Easy)

**Effort**: Low
**Impact**: Low-Medium

```python
# In dexgraspnet2/data/dataset.py, modify create_data_loaders()
DataLoader(
    ...,
    prefetch_factor=4,  # Default is 2, increase to 4-8
    persistent_workers=True,  # Keep workers alive between epochs
)
```

**Pros**: Simple one-line change, no data modification
**Cons**: Limited improvement, uses slightly more RAM

---

### Option 2: Cache Metadata in Memory (Easy)

**Effort**: Low
**Impact**: Low

```python
# In GraspNetDataset.__init__()
class GraspNetDataset(Dataset):
    def __init__(self, ...):
        # Pre-load all camera poses and metadata into RAM
        self._camera_poses_cache = {}
        self._align_mat_cache = {}

        for scene in self.scene_id:
            path = self._data_root / "scenes" / scene / camera
            self._camera_poses_cache[scene] = np.load(str(path / "camera_poses.npy"))
            self._align_mat_cache[scene] = np.load(str(path / "cam0_wrt_table.npy"))

# In _load_sample(), use cache instead of loading
camera_poses = self._camera_poses_cache[scene]
align_mat = self._align_mat_cache[scene]
```

**Saves**: 2 file reads per sample
**RAM cost**: ~50MB total (negligible)

---

### Option 3: RAM Disk (tmpfs) for Hot Data (Medium)

**Effort**: Medium
**Impact**: High for I/O

```bash
# Create 80GB RAM disk
sudo mount -t tmpfs -o size=80G tmpfs /mnt/ramdisk

# Copy graspness data (accessed every sample)
cp -r /mnt/datasets/dexgraspnet2/data/dex_graspness_new /mnt/ramdisk/

# Update symlink
rm data/dex_graspness_new
ln -s /mnt/ramdisk/dex_graspness_new data/dex_graspness_new
```

**Pros**: Eliminates disk I/O for graspness data
**Cons**: Uses ~80GB RAM, need to remount after reboot
**Note**: Graspness data is 282GB total, so only subset fits

---

### Option 4: Pre-compute Point Clouds (Medium)

**Effort**: Medium
**Impact**: Medium (eliminates CPU-intensive depth conversion)

```python
# One-time preprocessing script: scripts/precompute_clouds.py
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def depth_to_point_cloud(depth, intrinsics, factor_depth):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32) / factor_depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)

data_root = Path("data")
output_root = Path("data/precomputed_clouds")
camera = "realsense"

for scene_id in tqdm(range(100)):
    scene = f"scene_{str(scene_id).zfill(4)}"
    scene_path = data_root / "scenes" / scene / camera
    output_path = output_root / scene / camera
    output_path.mkdir(parents=True, exist_ok=True)

    meta = scio.loadmat(str(scene_path / "meta" / "0000.mat"))
    intrinsics = meta["intrinsic_matrix"]
    factor_depth = meta["factor_depth"]

    for view in range(256):
        str_view = str(view).zfill(4)
        depth = np.array(Image.open(scene_path / "depth_gt" / f"{str_view}.png"))
        cloud = depth_to_point_cloud(depth, intrinsics, factor_depth)
        np.save(output_path / f"{str_view}.npy", cloud.astype(np.float16))
```

**Pros**: Eliminates depth→cloud conversion each iteration
**Cons**: ~100GB additional storage, one-time preprocessing needed

---

### Option 5: LMDB Database (High effort, best for I/O)

**Effort**: High
**Impact**: High

```python
# One-time conversion script: scripts/convert_to_lmdb.py
import lmdb
import pickle
import numpy as np
from tqdm import tqdm

env = lmdb.open("data/graspnet.lmdb", map_size=500*1024**3)

with env.begin(write=True) as txn:
    for scene_id in tqdm(range(100)):
        scene = f"scene_{str(scene_id).zfill(4)}"
        for view in range(256):
            # Load all data for this sample
            data = {
                "depth": load_depth(scene, view),
                "seg": load_seg(scene, view),
                "meta": load_meta(scene, view),
                "camera_poses": load_camera_poses(scene),
                "graspness": load_graspness(scene, view),
            }
            key = f"{scene}_{view}".encode()
            txn.put(key, pickle.dumps(data))

# New dataset class using LMDB
class LMDBGraspNetDataset(Dataset):
    def __init__(self, lmdb_path, ...):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)

    def __getitem__(self, idx):
        scene, view = self.views[idx]
        key = f"{scene}_{view}".encode()
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(key))
        # Process data...
        return processed_data
```

**Pros**: Single sequential read instead of 6-10 random reads
**Cons**: Requires data conversion, ~500GB LMDB file, code changes

---

### Option 6: Optimize `_match_grasps_to_cloud` with KD-Tree (Easy, High Impact)

**Effort**: Low
**Impact**: High (10-100x faster for this operation)

```python
# In dexgraspnet2/data/dataset.py

# Add import at top
from scipy.spatial import cKDTree

# Replace _match_grasps_to_cloud method
def _match_grasps_to_cloud(
    self,
    cloud: np.ndarray,
    grasp_points: np.ndarray,
    k: int,
    max_point_dis: float,
) -> Tuple[np.ndarray, List[int]]:
    """
    Match grasp points to nearest cloud points using KD-tree.

    O(K × log N) instead of O(K × N).
    """
    # Build KD-tree once
    tree = cKDTree(cloud)

    # Query all grasp points at once
    distances, nearest_indices = tree.query(grasp_points, k=1)

    # Filter by max distance
    valid_mask = distances <= max_point_dis
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return np.zeros(len(grasp_points)), []

    # Limit to k grasps
    if len(valid_indices) > k:
        selected = np.random.choice(valid_indices, k, replace=False)
    else:
        selected = np.random.choice(valid_indices, k, replace=True)

    centers = np.zeros(len(grasp_points))
    centers[valid_indices] = nearest_indices[valid_indices]

    return centers, selected.tolist()
```

**Current complexity**: O(K × N) where K=256 grasps, N=40000 points
**New complexity**: O(K × log N)
**Expected speedup**: 10-100x for this function

---

## Implementation Priority

### Immediate (do now)
1. **Option 6** - KD-tree optimization (~10 lines, biggest CPU win)
2. **Option 2** - Cache metadata (~20 lines, easy win)
3. **Option 1** - Prefetch factor (1 line change)

### Short-term (next training run)
4. **Option 3** - RAM disk for graspness data

### Long-term (if needed)
5. **Option 4** - Pre-compute point clouds
6. **Option 5** - LMDB conversion

---

## Estimated Impact

| Optimization | Implementation Time | Expected Speedup |
|--------------|---------------------|------------------|
| KD-tree | 15 min | 1.2-1.5x |
| Cache metadata | 10 min | 1.05x |
| Prefetch factor | 2 min | 1.05x |
| RAM disk | 30 min | 1.3-1.5x |
| Pre-compute clouds | 2 hours | 1.2x |
| LMDB | 4 hours | 1.5-2x |

**Combined (options 1-3 + 6)**: Expected ~1.5-2x speedup with minimal effort.
