"""CPU MANO silhouettes and depth (OpenCV polygon fill, no GL; Mac friendly).

Vertices are given as image pixels + camera depth; projection (pinhole or distorted) is done by the caller, which is
accurate for MANO's small triangles even under the wide-angle distortion of the mono cameras.
"""
from __future__ import annotations

import cv2
import numpy as np


def silhouette(uv: np.ndarray, faces: np.ndarray, size: tuple) -> np.ndarray:
    """uv (V,2) px, faces (F,3), size (W,H) -> bool mask (H,W)."""
    W, H = size
    m = np.zeros((H, W), np.uint8)
    ok = np.isfinite(uv).all(1) & (np.abs(uv) < 1e5).all(1)
    f = faces[ok[faces].all(1)]
    if len(f):
        pts = np.round(uv[f] * 4).astype(np.int32)  # 2 bits of sub-pixel precision
        cv2.fillPoly(m, list(pts), 1, lineType=cv2.LINE_8, shift=2)
    return m > 0


def depth_map(uv: np.ndarray, z: np.ndarray, faces: np.ndarray, size: tuple) -> np.ndarray:
    """z-buffered depth (m, 0 = empty) by painting triangles far -> near with their mean depth. Error vs exact
    per-pixel interpolation is ~1 mm for MANO-sized triangles."""
    W, H = size
    d = np.zeros((H, W), np.float32)
    ok = np.isfinite(uv).all(1) & (z > 1e-3) & (np.abs(uv) < 1e5).all(1)
    f = faces[ok[faces].all(1)]
    if not len(f):
        return d
    zf = z[f].mean(1)
    order = np.argsort(-zf)
    pts = np.round(uv[f[order]] * 4).astype(np.int32)
    for p, zz in zip(pts, zf[order]):
        cv2.fillConvexPoly(d, p, float(zz), lineType=cv2.LINE_8, shift=2)
    return d


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")
