"""tools/camera_pose_graph.optimise on a synthetic clip: drifting odometry is pulled back by table planes and resting
constraints."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import camera_pose_graph as pg  # noqa: E402

N = 120


def truth():
    """Camera 0.6 m above a table (world z up = table normal, table at z = 0), looking down, swaying."""
    c2w = np.tile(np.eye(4), (N, 1, 1))
    s = np.linspace(0, 1, N)
    base = Rotation.from_euler("x", 180 - 35, degrees=True)
    for i in range(N):
        c2w[i, :3, :3] = (Rotation.from_euler("zy", [20 * np.sin(3 * s[i]), 8 * s[i]], degrees=True) * base).as_matrix()
        c2w[i, :3, 3] = [0.2 * s[i], -0.4 + 0.1 * s[i], 0.6 + 0.05 * np.sin(5 * s[i])]
    return c2w


class PoseGraphTest(unittest.TestCase):
    def test_drift_is_reduced(self):
        T = truth()
        T = np.linalg.inv(T[0]) @ T                       # world = camera 0, as the odometry
        nw_true = np.linalg.inv(truth()[0])[:3, :3] @ [0, 0, 1]
        dw_true = -(nw_true @ (np.linalg.inv(truth()[0]) @ [0, 0, 0, 1])[:3])
        # odometry: true relative motions with a constant small rotation bias -> drift
        bias = Rotation.from_rotvec([0.0007, -0.0005, 0.0004]).as_matrix()   # ~0.055 deg/frame, ~6.5 deg over the clip
        vo = T.copy()
        for i in range(1, N):
            rel = np.linalg.inv(T[i - 1]) @ T[i]
            rel[:3, :3] = bias @ rel[:3, :3]
            rel[:3, 3] *= 1.03
            vo[i] = vo[i - 1] @ rel
        # object resting at a fixed world pose for the first and last 30 frames, moving (hand) in between
        Wobj = np.eye(4)
        Wobj[:3, 3] = (np.linalg.inv(truth()[0]) @ [0.1, 0.0, 0.04, 1])[:3]
        Tcm = np.linalg.inv(T) @ Wobj
        Wend = Wobj.copy()
        Wend[:3, 3] += nw_true * 0 + np.array([0.05, 0.02, 0.0])
        Tcm[90:] = np.linalg.inv(T[90:]) @ Wend
        # planes seen from each true camera
        planes = []
        for i in range(N):
            n_c = T[i, :3, :3].T @ nw_true
            d_c = dw_true + nw_true @ T[i, :3, 3]
            planes.append((n_c, d_c, 1.0))
        lost = np.zeros(N, bool)
        out, info = pg.optimise(vo, Tcm, np.zeros(3), planes, [(0, 30), (90, N)], lost)
        err_vo = np.linalg.norm(vo[:, :3, 3] - T[:, :3, 3], axis=1)
        err_pg = np.linalg.norm(out[:, :3, 3] - T[:, :3, 3], axis=1)
        self.assertLess(np.median(err_pg), 0.8 * np.median(err_vo))
        obj_vo = np.linalg.norm((vo[90:] @ Tcm[90:])[:, :3, 3] - Wend[:3, 3], axis=1).mean()
        obj_pg = np.linalg.norm((out[90:] @ Tcm[90:])[:, :3, 3] - Wend[:3, 3], axis=1).mean()
        self.assertLess(obj_pg, obj_vo)


if __name__ == "__main__":
    unittest.main()
