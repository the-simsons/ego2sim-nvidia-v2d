"""HOT3D-Clips (Aria, train split) reader for a ground-truth check of the E1best methodology.

Data: 8 clips from huggingface.co/datasets/bop-benchmark/hot3d (train_aria/clip-*.tar, 150 frames each) in
handtracking/data/hot3d (list in clips.txt). Licence: images CC BY-SA 4.0, hand annotations CC BY-NC-SA 4.0
(non-commercial research only).

Streams: 214-1 RGB 1408x1408, 1201-1 / 1201-2 mono SLAM 640x480, all FISHEYE624 cameras. GT hands are MANO
(15 PCA comps + wrist axis-angle/translation, smplx convention) in the world frame; per-frame T_world_from_camera.
GT joints/verts are computed with HOT3D's own MANO layer (methods/hot3d_repo/hot3d/data_loaders/mano_layer.py,
smplx), so their conventions are reproduced exactly; that needs torch + smplx (run in methods/.venv-hawor).

The benchmark camera here is a PINHOLE view of the RGB stream: same centre and orientation as 214-1, focal F_PIN,
principal point at the image centre (HaWoR assumes it).
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np

from handbench import paths

ROOT = paths.ROOT / "data" / "hot3d"
RGB, SLAM_L, SLAM_R = "214-1", "1201-1", "1201-2"
F_PIN = 600.0
SIZE_PIN = 1408


def clips() -> list[Path]:
    return [ROOT / l.strip() for l in open(ROOT / "clips.txt") if l.strip()]


def quat_t_to_T(d) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot

    w, x, y, z = d["quaternion_wxyz"]
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = d["translation_xyz"]
    return T


def fisheye624_project(X: np.ndarray, params) -> np.ndarray:
    """Aria FISHEYE624 (single focal): params = [f, cx, cy, k0..k5, p0, p1, s0..s3]. X (N,3) camera frame -> (N,2) px."""
    f, cx, cy = params[:3]
    k = params[3:9]
    p0, p1 = params[9:11]
    s = params[11:15]
    a, b = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
    r = np.sqrt(a * a + b * b)
    th = np.arctan(r)
    th2 = th * th
    poly = 1 + th2 * (k[0] + th2 * (k[1] + th2 * (k[2] + th2 * (k[3] + th2 * (k[4] + th2 * k[5])))))
    sc = np.where(r > 1e-9, th * poly / np.maximum(r, 1e-9), 1.0)
    u, v = a * sc, b * sc
    rd2 = u * u + v * v
    ut = u + (2 * u * u + rd2) * p0 + 2 * u * v * p1 + s[0] * rd2 + s[1] * rd2 * rd2
    vt = v + (2 * v * v + rd2) * p1 + 2 * u * v * p0 + s[2] * rd2 + s[3] * rd2 * rd2
    return np.stack([f * ut + cx, f * vt + cy], 1)


class Clip:
    def __init__(self, tar_path: Path):
        self.path = Path(tar_path)
        self.name = self.path.stem
        self.tar = tarfile.open(self.path)
        self.members = {m.name: m for m in self.tar.getmembers()}
        self.n = len([k for k in self.members if k.endswith(".info.json")])
        self.shape = json.loads(self._read("__hand_shapes.json__"))["mano"]

    def _read(self, name):
        return self.tar.extractfile(self.members[name]).read()

    def json(self, i, kind):
        return json.loads(self._read(f"{i:06d}.{kind}.json"))

    def image(self, i, stream):
        buf = np.frombuffer(self._read(f"{i:06d}.image_{stream}.jpg"), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def cameras(self, i):
        c = self.json(i, "cameras")
        return {s: {"T_world_from_camera": quat_t_to_T(v["T_world_from_camera"]),
                    "params": np.array(v["calibration"]["projection_params"], float),
                    "size": (v["calibration"]["image_width"], v["calibration"]["image_height"])} for s, v in c.items()}


def pinhole_K(f=F_PIN, size=SIZE_PIN):
    return np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])


def pinhole_from_fisheye_maps(params, K, size, R_cam_from_pin=np.eye(3)):
    """remap() maps rendering a pinhole view (K, size) from a fisheye image; R_cam_from_pin rotates pinhole rays into
    the fisheye camera's frame (identity = same orientation)."""
    W, H = size
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    rays = rays @ R_cam_from_pin.T
    uv = fisheye624_project(rays, params)
    uv[rays[:, 2] <= 0.05] = -1
    return uv[:, 0].reshape(H, W).astype(np.float32), uv[:, 1].reshape(H, W).astype(np.float32)


class Hot3dMano:
    """HOT3D's hand GT convention, reproduced with handbench's own NumPy MANO (no HOT3D code is executed).

    HOT3D builds MANO with smplx `use_pca=True, num_pca_comps=15` and smplx's default `flat_hand_mean=False`:
        finger pose (45) = hands_mean + thetas(15) @ hands_components[:15]
    with `wrist_xform` = [global_orient (3), transl (3)], translation added after FK (root at J0 + transl). Left hands use
    MANO_LEFT, which is the mirror of MANO_RIGHT: its left-frame axis-angles map to our right-model convention by
    negating the y and z components of every joint (incl. global orient), and the output is mirrored in x before
    `transl` is added, exactly what handbench.mano does for is_right=False. Validated by projecting the GT hands into
    the images and comparing with HOT3D's own 2D hand boxes.
    """

    def __init__(self):
        from handbench.mano import Mano

        self.mano = Mano()
        d = np.load(paths.ROOT / "assets" / "mano_right.npz")
        self.mean_r = d["hands_mean"].reshape(45)
        self.comps_r = d["hands_components"][:15]  # (15, 45)
        flip = np.tile([1.0, -1.0, -1.0], 15)
        self.mean_l, self.comps_l = self.mean_r * flip, self.comps_r * flip  # MANO_LEFT = mirrored right

    def world_joints_verts(self, hand: dict, betas, is_right: bool):
        """GT hand -> (joints21 OpenPose order, verts778) in the WORLD frame, metres."""
        th = np.asarray(hand["mano_pose"]["thetas"], float)
        wx = np.asarray(hand["mano_pose"]["wrist_xform"], float)
        if is_right:
            pose = self.mean_r + th @ self.comps_r
            go = wx[:3]
        else:
            pose_left = self.mean_l + th @ self.comps_l  # in the left model's own frame
            pose = pose_left * np.tile([1.0, -1.0, -1.0], 15)  # -> our right-model-mirrored convention
            go = wx[:3] * np.array([1.0, -1.0, -1.0])
        V, J = self.mano.forward(go, pose, np.asarray(betas, float), wx[3:], is_right, 1.0)
        return J[0], V[0]

# Aria stores the RGB image rotated by 90 deg; the benchmark pinhole view is rotated -90 deg about the optical axis so
# hands come in from the bottom. Rays in the pinhole frame -> rays in the 214-1 camera frame:
R_CAM_FROM_PIN = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
