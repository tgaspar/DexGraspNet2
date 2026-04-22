import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 4x4 transform matrix."""
    w, x, y, z = quat_wxyz
    # scipy uses (x,y,z,w) order
    rot = Rotation.from_quat([x, y, z, w])
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    return T


def load_scene_data(
    scene_id: str = "scene_0220", view_idx: int = 0, data_root: str = "data"
):
    """
    Load scene data including camera and all objects.

    Returns:
        Dictionary containing camera transforms and a list of object data.
    """
    import xml.etree.ElementTree as ET

    scene_dir = Path(data_root) / "scenes" / scene_id / "realsense"

    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    # Load transforms
    cam0_wrt_table = np.load(scene_dir / "cam0_wrt_table.npy")
    camera_poses = np.load(scene_dir / "camera_poses.npy")
    camera_pose_wrt_cam0 = camera_poses[view_idx]

    # Camera pose in table/world frame
    camera_in_world = cam0_wrt_table @ camera_pose_wrt_cam0

    # Load all objects from annotation
    ann_file = scene_dir / "annotations" / f"{view_idx:04d}.xml"
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    tree = ET.parse(ann_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall("obj"):
        obj_id = int(obj.find("obj_id").text)
        pos = np.array([float(x) for x in obj.find("pos_in_world").text.split()])
        ori_wxyz = [float(x) for x in obj.find("ori_in_world").text.split()]

        # Build transform matrix (in camera frame)
        obj_pose_in_camera = quat_wxyz_to_matrix(np.array(ori_wxyz))
        obj_pose_in_camera[:3, 3] = pos

        # Transform to world/table frame
        obj_pose_in_world = cam0_wrt_table @ obj_pose_in_camera

        # Get mesh path
        mesh_path = Path(data_root) / "meshdata" / f"{obj_id:03d}" / "simplified.obj"
        urdf_path = (
            Path(data_root)
            / "meshdata"
            / f"{obj_id:03d}"
            / "nontextured_simplified.urdf"
        )

        objects.append(
            {
                "id": obj_id,
                "pose_in_camera": obj_pose_in_camera,
                "pose_in_world": obj_pose_in_world,
                "mesh_path": mesh_path,
                "urdf_path": urdf_path,
            }
        )

    return {
        "cam0_wrt_table": cam0_wrt_table,
        "camera_pose_wrt_cam0": camera_pose_wrt_cam0,
        "camera_in_world": camera_in_world,
        "objects": objects,
    }
