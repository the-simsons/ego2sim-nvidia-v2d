"""Ego rig cameras for the published Track 3 videos.

Conventions (checked by handbench/tools/calib_check.py, see refs/calib/rig.json):
  * The factory intrinsics + 14-coef rational distortion fit the published videos as-is ("raw" hypothesis; the
    rotated-K hypothesis gets 5x fewer essential-matrix inliers).
  * Benchmark frame = UNDISTORTED cam_a OpenCV camera: pinhole with the factory cam_a K, no distortion (the same
    pinhole tools/ego_stereo_depth.py writes cam_a_undist.mp4 with). x right, y down, z forward, metres.
  * Physically cam_c is left, cam_a centre, cam_b right (~3.7 cm from cam_a).
  * Each camera carries T_cam_from_a (4x4): X_cam = R X_a + t.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CAMS = ("ego_cam_a", "ego_cam_b", "ego_cam_c")
F3 = np.diag([-1.0, -1.0, 1.0])


@dataclass
class Cam:
    name: str
    K: np.ndarray
    dist: np.ndarray
    size: tuple  # (W, H)
    T_from_a: np.ndarray = field(default_factory=lambda: np.eye(4))

    @property
    def R(self):
        return self.T_from_a[:3, :3]

    @property
    def t(self):
        return self.T_from_a[:3, 3]

    @property
    def center_in_a(self):
        return -self.R.T @ self.t

    def to_cam(self, X_a: np.ndarray) -> np.ndarray:
        return X_a @ self.R.T + self.t

    def project(self, X_a: np.ndarray, distort: bool = True) -> np.ndarray:
        """(..., 3) points in the cam_a frame -> (..., 2) pixels of this camera's published (distorted) image."""
        import cv2

        X = self.to_cam(np.asarray(X_a, np.float64).reshape(-1, 3))
        if not distort:
            uv = X[:, :2] / X[:, 2:3]
            uv = uv @ self.K[:2, :2].T + self.K[:2, 2]
        else:
            uv, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), self.K, self.dist)
            uv = uv.reshape(-1, 2)
        return uv.reshape(np.shape(X_a)[:-1] + (2,))

    def depth(self, X_a: np.ndarray) -> np.ndarray:
        return self.to_cam(np.asarray(X_a, np.float64).reshape(-1, 3))[:, 2].reshape(np.shape(X_a)[:-1])

    def undistort_px(self, uv: np.ndarray) -> np.ndarray:
        """Published-image pixels -> normalised image coords (x/z, y/z) in this camera's frame."""
        import cv2

        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        if not len(uv):  # e.g. a camera with no keypoints at all
            return np.zeros((0, 2))
        n = cv2.undistortPoints(uv, self.K, self.dist).reshape(-1, 2)
        return n


def _factory_T_from_b(c: dict) -> np.ndarray:
    T = np.array(c.get("extrinsics_to_stereo_left", np.eye(4)), np.float64)
    if np.abs(T[:3, 3]).max() > 1.0:
        T[:3, 3] /= 100.0
    return T


def load_factory(calib_json: str | Path, hypothesis: str = "conj") -> dict:
    """Factory rig. hypothesis: 'raw' = extrinsics as stored; 'conj' = conjugated by diag(-1,-1,1) (180 deg device
    rotation applied to the extrinsics only, K raw)."""
    d = json.load(open(calib_json))["cameras"]
    Tb = {n: _factory_T_from_b(d[n]) for n in CAMS}  # X_n = T X_b
    if hypothesis == "conj":
        for n in CAMS:
            T = Tb[n].copy()
            T[:3, :3] = F3 @ T[:3, :3] @ F3
            T[:3, 3] = F3 @ T[:3, 3]
            Tb[n] = T
    Ta_inv = np.linalg.inv(Tb["ego_cam_a"])
    cams = {}
    for n in CAMS:
        c = d[n]
        cams[n] = Cam(n, np.array(c["intrinsic_matrix"], np.float64), np.array(c["distortion_coefficients"], np.float64),
                      tuple(c["resolution"]), Tb[n] @ Ta_inv)
    return cams


def load_rig(rig_json: str | Path | None = None, calib_json: str | Path | None = None) -> dict:
    """Load the benchmark rig (refs/calib/rig.json written by calib_check.py). Falls back to factory 'conj'."""
    if rig_json and Path(rig_json).exists():
        r = json.load(open(rig_json))
        return {n: Cam(n, np.array(v["K"]), np.array(v["dist"]), tuple(v["size"]), np.array(v["T_from_a"]))
                for n, v in r["cams"].items()}
    if calib_json is None:
        raise FileNotFoundError("need rig_json or calib_json")
    return load_factory(calib_json)


def save_rig(cams: dict, path: str | Path, meta: dict | None = None):
    out = {"frame": "undistorted ego_cam_a OpenCV camera (factory K, no distortion), metres",
           "cams": {n: {"K": c.K.tolist(), "dist": c.dist.tolist(), "size": list(c.size), "T_from_a": c.T_from_a.tolist()}
                    for n, c in cams.items()}}
    out.update(meta or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)


def pinhole_a(cams: dict) -> np.ndarray:
    """K of the benchmark camera (undistorted cam_a)."""
    return cams["ego_cam_a"].K.copy()


def triangulate(cams: list, uvs: list) -> np.ndarray:
    """Linear (DLT on normalised coords) triangulation of one point seen in several cams. uvs: list of (2,) published
    pixels. Returns X in the cam_a frame."""
    A = []
    for c, uv in zip(cams, uvs):
        n = c.undistort_px(np.asarray(uv)[None])[0]
        P = c.T_from_a[:3]
        A.append(n[0] * P[2] - P[0])
        A.append(n[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3]


def triangulate_batch(c1: Cam, c2: Cam, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    """Two-view triangulation for many points at once: (N,2),(N,2) published px -> (N,3) cam_a frame."""
    n1 = c1.undistort_px(uv1)
    n2 = c2.undistort_px(uv2)
    P1, P2 = c1.T_from_a[:3], c2.T_from_a[:3]
    A = np.stack([n1[:, :1] * P1[2] - P1[0], n1[:, 1:] * P1[2] - P1[1],
                  n2[:, :1] * P2[2] - P2[0], n2[:, 1:] * P2[2] - P2[1]], 1)  # (N,4,4)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1]
    return X[:, :3] / X[:, 3:4]
