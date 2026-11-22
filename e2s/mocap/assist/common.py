"""Shared helpers for the assist steps: clip config from the manifest, the mocap object at video times, box drawing."""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

from e2s.mocap import motive
from e2s.mocap.ground_truth import rb_to_mesh

SIGNS = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(SIGNS[a] - SIGNS[b]).sum() == 2]


class Clip:
    """Everything an assist step needs about one clip, from its manifest."""

    def __init__(self, manifest, root, body=None):
        self.m, self.root = manifest, Path(root)
        self.name = manifest["clip"]
        gt, inp = manifest["ground_truth"], manifest["inputs"]
        self.body = body or next(iter(gt["bodies"]))
        spec = gt["bodies"][self.body]
        self.spec = spec
        K = inp["intrinsics"]
        self.fx, self.fy, self.cx, self.cy = K["fx"], K["fy"], K["cx"], K["cy"]
        self.W, self.H = K["width"], K["height"]
        self.extents = np.asarray(spec["box_extents"], float)
        self.corners = SIGNS * self.extents / 2
        self.T_rb_mesh = rb_to_mesh(spec["rb_to_mesh"])
        self.up = "xyz".index(gt.get("up_axis", "z"))
        self.out = self.root / manifest["outputs_dir"]
        self.frames = self.root / inp["frames"]
        self.mask_dir = self.root / inp["object_masks"][self.body]
        self.clip_dir = self.frames.parent
        self.pts = np.asarray(json.load(open(self.root / inp["frame_times"]))["pts"], float)
        d = motive.load(self.root / gt["mocap_csv"])
        b = d["bodies"][spec.get("rigid_body", self.body)]
        ok = ~np.isnan(b["pos"]).any(1) & ~np.isnan(b["quat_xyzw"]).any(1)
        self.tm, self.qm, self.pm = d["time"][ok], b["quat_xyzw"][ok], b["pos"][ok]
        self.mocap_fps = d["fps"]
        self._slerp = Slerp(self.tm, Rot.from_quat(self.qm))

    def world_mesh(self, t):
        """(N,4,4) world <- mesh at mocap times t (clamped to the take)."""
        t = np.clip(np.atleast_1d(t), self.tm[0], self.tm[-1])
        T = np.tile(np.eye(4), (len(t), 1, 1))
        T[:, :3, :3] = self._slerp(t).as_matrix()
        T[:, :3, 3] = np.c_[[np.interp(t, self.tm, self.pm[:, a]) for a in range(3)]].T
        return T @ self.T_rb_mesh

    def mask(self, i):
        return cv2.imread(str(self.mask_dir / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)

    def project(self, T_cam_mesh, scale=1.0):
        X = (T_cam_mesh[:3, :3] @ self.corners.T).T + T_cam_mesh[:3, 3]
        z = np.maximum(X[:, 2], 1e-3)
        return np.c_[self.fx * scale * X[:, 0] / z + self.cx * scale, self.fy * scale * X[:, 1] / z + self.cy * scale], X

    def iou(self, T_cam_mesh, mask, scale=1.0):
        uv, X = self.project(T_cam_mesh, scale)
        if (X[:, 2] <= 0.02).any():
            return 0.0
        p = np.zeros_like(mask)
        cv2.fillConvexPoly(p, cv2.convexHull(uv.astype(np.int32)), 255)
        return np.logical_and(p > 0, mask > 0).sum() / max(np.logical_or(p > 0, mask > 0).sum(), 1)

    def draw(self, frame, T_cam_mesh, label):
        im = cv2.imread(str(self.frames / f"{frame:06d}.jpg"))
        uv, _ = self.project(T_cam_mesh)
        for a, b in EDGES:
            cv2.line(im, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(im, label, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        return cv2.resize(im, (960, 540))


def write_pose_dir(d, T):
    """Per-frame pose JSONs (rotation wxyz, translation, scale) as the object stages use."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for i, Ti in enumerate(T):
        x, y, z, w = Rot.from_matrix(Ti[:3, :3]).as_quat()
        json.dump({"rotation": [float(w), float(x), float(y), float(z)], "translation": [float(v) for v in Ti[:3, 3]],
                   "scale": [1.0, 1.0, 1.0]}, open(d / f"{i:06d}.json", "w"), indent=1)
