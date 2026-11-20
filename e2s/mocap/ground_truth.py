"""Ground-truth object trajectories for a clip, sampled at its video frame times."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from e2s.geometry import interp_poses, mat
from e2s.mocap import motive


def rb_to_mesh(spec):
    """{"translation": [x,y,z], "rotation_deg_xyz": [rx,ry,rz]} -> 4x4 (mesh pose in the rigid-body frame)."""
    R = Rot.from_euler("xyz", spec.get("rotation_deg_xyz", [0, 0, 0]), degrees=True).as_matrix()
    return mat(R, spec.get("translation", [0, 0, 0]))


def object_trajectories(manifest, root, video_times):
    """{body name: (N,4,4) world <- mesh} at the given video frame times (s), using the manifest's sync offset."""
    gt = manifest["ground_truth"]
    data = motive.load(Path(root) / gt["mocap_csv"])
    lag = float(gt["sync"]["video_to_mocap_lag_s"])
    out = {}
    for name, spec in gt["bodies"].items():
        b = data["bodies"][spec.get("rigid_body", name)]
        ok = ~np.isnan(b["pos"]).any(1)
        T = interp_poses(data["time"][ok], Rot.from_quat(b["quat_xyzw"][ok]).as_matrix(), b["pos"][ok],
                         np.asarray(video_times) + lag)
        out[name] = T @ rb_to_mesh(spec["rb_to_mesh"])
    return out
