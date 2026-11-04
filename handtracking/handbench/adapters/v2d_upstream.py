"""Geometry of the cached upstream ego pipeline (video_to_data run_ego_reconstruction, --hand_tracking hamer) and
the conversion of its per-frame hand records into the benchmark frame.

The upstream run does not use the factory calibration. Its frames are cam_a undistorted by AnyCalib (opencv_fisheye
model, K_any) and re-projected with K_u (anycalib/undistorted_intrinsics.json); the hands/depth then live in a pinhole
camera with K_stable (intrinsics_stable.json, GeoCalib). We map every predicted 3D point X_m by its PIXEL and keep its
DEPTH:
    p = K_stable X_m / z  ->  n_u = K_u^-1 p  ->  raw px = fisheye_distort(n_u; K_any, D)
      ->  n = factory_undistort(raw px)  ->  X_bench = (n_x z, n_y z, z)
so the image-plane fit the method achieved is preserved exactly and only its depth is judged.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from handbench import paths
from handbench.calib import Cam


def _K(d):
    return np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1.0]])


class UpstreamCamera:
    def __init__(self, ep_dir: Path, cam_a: Cam, anycalib_dir: Path | None = None):
        ep_dir = Path(ep_dir)
        ac = Path(anycalib_dir) if anycalib_dir else ep_dir / "anycalib"
        self.K_stable = _K(json.load(open(ep_dir / "intrinsics_stable.json")))
        self.K_any = _K(json.load(open(ac / "intrinsics.json")))
        self.K_u = _K(json.load(open(ac / "undistorted_intrinsics.json")))
        d = json.load(open(ac / "distortion.json"))
        assert d["model"] == "opencv_fisheye", d
        self.D = np.array(d["params"], np.float64)
        self.cam_a = cam_a
        self.K_bench = cam_a.K

    def cache_px_to_bench_n(self, uv: np.ndarray) -> np.ndarray:
        """Pixels of the cached (AnyCalib-undistorted) frames -> normalised coords of the benchmark camera."""
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        n_u = (uv - self.K_u[:2, 2]) / np.diag(self.K_u)[:2]
        raw = cv2.fisheye.distortPoints(n_u.reshape(-1, 1, 2), self.K_any, self.D).reshape(-1, 2)
        return self.cam_a.undistort_px(raw)

    def points_to_bench(self, X: np.ndarray) -> np.ndarray:
        """(..., 3) points in the upstream camera (K_stable) -> benchmark cam_a frame (pixel preserved, depth kept)."""
        sh = np.shape(X)
        X = np.asarray(X, np.float64).reshape(-1, 3)
        z = X[:, 2]
        p = X[:, :2] / z[:, None] * np.diag(self.K_stable)[:2] + self.K_stable[:2, 2]
        n = self.cache_px_to_bench_n(p)
        return np.c_[n * z[:, None], z].reshape(sh)

    @lru_cache(None)
    def bench_to_cache_maps(self, scale: float = 1.0):
        """remap() maps that sample a cached-frame image at every benchmark pixel (for masks / depth)."""
        W, H = self.cam_a.size
        Wo, Ho = int(round(W * scale)), int(round(H * scale))
        uu, vv = np.meshgrid((np.arange(Wo) + 0.5) / scale - 0.5, (np.arange(Ho) + 0.5) / scale - 0.5)
        n = (np.stack([uu.ravel(), vv.ravel()], 1) - self.K_bench[:2, 2]) / np.diag(self.K_bench)[:2]
        raw, _ = cv2.projectPoints(np.c_[n, np.ones(len(n))], np.zeros(3), np.zeros(3), self.cam_a.K, self.cam_a.dist)
        n_raw = cv2.fisheye.undistortPoints(raw.reshape(-1, 1, 2), self.K_any, self.D).reshape(-1, 2)
        p = n_raw * np.diag(self.K_u)[:2] + self.K_u[:2, 2]
        return p[:, 0].reshape(Ho, Wo).astype(np.float32), p[:, 1].reshape(Ho, Wo).astype(np.float32)

    def warp_to_bench(self, img: np.ndarray, scale: float = 1.0, nearest: bool = True) -> np.ndarray:
        mx, my = self.bench_to_cache_maps(scale)
        return cv2.remap(img, mx, my, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR, borderValue=0)


def upstream_dir(ep) -> Path:
    return paths.CACHE / paths.ep_name(ep)


def anycalib_dir(ep) -> Path:
    """anycalib/ is not copied into the cache (it holds a 2028x1520 video); read it from the original run."""
    local = upstream_dir(ep) / "anycalib"
    if local.exists():
        return local
    return Path("/usr1/e2s/v2d_challenge_data/outputs/track3_handfix") / paths.ep_name(ep) / "anycalib"
