"""Per-frame cues from segmentation masks (video only)."""
from pathlib import Path

import cv2
import numpy as np


def mask_dirs(spec):
    """'DIR[,DIR...]' -> list of dirs; a dir of per-track subdirs (SAM 2 layout) expands to its subdirs."""
    out = []
    for d in [Path(x) for x in (spec.split(",") if isinstance(spec, str) else spec) if x]:
        subs = [s for s in sorted(d.iterdir()) if s.is_dir()] if d.is_dir() else []
        out += subs if subs else [d]
    return out


def union(dirs, i, shape):
    m = np.zeros(shape, np.uint8)
    for d in dirs:
        x = cv2.imread(str(Path(d) / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if x is not None:
            m |= (cv2.resize(x, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    return m


def hand_contact(object_dir, hand_dirs, n, px=15, min_overlap=50):
    """(n,) bool: the hand mask dilated by px overlaps the object mask. Frames without an object mask count as contact
    (unknown), so they never start a still stretch."""
    dirs = mask_dirs(hand_dirs)
    ker = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    out = np.ones(n, bool)
    for i in range(n):
        o = cv2.imread(str(Path(object_dir) / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if o is None or not o.any():
            continue
        h = union(dirs, i, o.shape)
        out[i] = ((cv2.dilate(h, ker) > 0) & (o > 0)).sum() > min_overlap
    return out
