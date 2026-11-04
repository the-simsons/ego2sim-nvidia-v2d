"""Perspective (virtual pinhole) crops of the wide-FOV distorted ego images.

A crop is a virtual pinhole camera sharing the real camera's centre, rotated to look along `center_ray`, with focal
`f` px and size S x S. Pixels of the crop map to exact rays of the real camera, so keypoints detected in a crop are
converted to normalised coords without any undistortion approximation.
"""
from __future__ import annotations

import cv2
import numpy as np

from handbench.calib import Cam


def look_at(ray: np.ndarray) -> np.ndarray:
    """Rotation R (crop <- cam) whose z axis is `ray` and whose x axis stays as horizontal as possible."""
    z = ray / np.linalg.norm(ray)
    x = np.cross([0.0, 1.0, 0.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


class PerspectiveCrop:
    def __init__(self, cam: Cam, center_ray: np.ndarray, f: float, size: int = 256):
        self.cam, self.f, self.S = cam, float(f), int(size)
        self.R = look_at(np.asarray(center_ray, float))
        self.K = np.array([[f, 0, (size - 1) / 2], [0, f, (size - 1) / 2], [0, 0, 1.0]])
        uu, vv = np.meshgrid(np.arange(size), np.arange(size))
        rays_crop = np.stack([(uu - self.K[0, 2]) / f, (vv - self.K[1, 2]) / f, np.ones_like(uu, float)], -1)
        rays_cam = rays_crop.reshape(-1, 3) @ self.R  # crop -> cam: R^T x  (row form: x @ R)
        uv, _ = cv2.projectPoints(rays_cam, np.zeros(3), np.zeros(3), cam.K, cam.dist)
        uv = uv.reshape(size, size, 2).astype(np.float32)
        front = rays_cam[:, 2].reshape(size, size) > 0.05
        uv[~front] = -1
        self.mapx, self.mapy = uv[..., 0], uv[..., 1]

    @staticmethod
    def around(cam: Cam, rays: np.ndarray, size: int = 256, margin: float = 1.3, min_f: float = 50.0):
        """Crop covering a set of rays (N,3) in this camera's frame, with `margin` extra extent."""
        c = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        center = c.mean(0)
        R = look_at(center)
        q = c @ R.T
        q = q[:, :2] / q[:, 2:3]
        half = np.abs(q).max() * margin
        f = max(min_f, (size / 2) / max(half, 1e-3))
        return PerspectiveCrop(cam, center, f, size)

    def sample(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def crop_px_to_norm(self, uv: np.ndarray) -> np.ndarray:
        """Crop pixels (N,2) -> normalised coords (x/z, y/z) in the real camera frame."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        r = np.c_[(uv - self.K[:2, 2]) / self.f, np.ones(len(uv))] @ self.R
        return r[:, :2] / r[:, 2:3]

    def crop_px_to_img(self, uv: np.ndarray) -> np.ndarray:
        """Crop pixels -> published (distorted) image pixels of the real camera."""
        n = self.crop_px_to_norm(uv)
        p, _ = cv2.projectPoints(np.c_[n, np.ones(len(n))], np.zeros(3), np.zeros(3), self.cam.K, self.cam.dist)
        return p.reshape(-1, 2)

    def norm_to_crop_px(self, n: np.ndarray) -> np.ndarray:
        r = np.c_[np.asarray(n, float).reshape(-1, 2), np.ones(len(n))] @ self.R.T
        return r[:, :2] / r[:, 2:3] * self.f + self.K[:2, 2]
