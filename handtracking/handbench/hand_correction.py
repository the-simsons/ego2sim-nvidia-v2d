"""Correct single-camera hand tracks with triangulated keypoints (shared by the pipeline stage and the benchmark).

Per hand, over runs of consecutive frames where the hand is tracked:
  shift_to_keypoints   frames with >= MIN_JOINTS triangulated joints: translate the hand by the median joint offset;
                       rejected when the median residual after the shift exceeds MAX_RESIDUAL_M (wrong-hand matches)
  carry_offsets        offsets cleaned against a 9-frame running median, interpolated between frames that have one and
                       held for `hold` frames past the first / last (depth errors of single-camera trackers vary slowly)
Stereo fallback constants (MIN_STEREO_FRAC, SCALE_RANGE) are applied by the callers, which own the depth source.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

MIN_JOINTS = 4  # triangulated joints needed to shift a frame
MAX_RESIDUAL_M = 0.03  # median joint residual after the shift above which the shift is rejected
HOLD_S = 1.0  # how long (s) a shift is carried past the last frame that has one
OUTLIER_M = 0.04  # offsets further than this from their running median are dropped
MIN_STEREO_FRAC = 0.3  # stereo fallback: share of silhouette pixels that need a stereo depth
SCALE_RANGE = (0.4, 2.0)  # stereo fallback: accepted scale about the camera


def runs_of(mask: np.ndarray) -> list[np.ndarray]:
    """Index runs of consecutive True values."""
    idx = np.nonzero(mask)[0]
    return np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1) if len(idx) else []


def shift_to_keypoints(J: np.ndarray, T: np.ndarray, usable: np.ndarray, tracked: np.ndarray) -> np.ndarray:
    """J (n,21,3) tracked joints, T (n,21,3) triangulated joints, usable (n,21) bool -> offsets (n,3), NaN = none."""
    off = np.full((len(J), 3), np.nan)
    for i in np.nonzero(tracked)[0]:
        u = usable[i]
        if u.sum() < MIN_JOINTS:
            continue
        o = np.median(T[i, u] - J[i, u], 0)
        if np.median(np.linalg.norm(T[i, u] - (J[i, u] + o), axis=1)) <= MAX_RESIDUAL_M:
            off[i] = o
    return off


def carry_offsets(off: np.ndarray, tracked: np.ndarray, hold: int) -> np.ndarray:
    """off (n,3) NaN where no shift; returns offsets interpolated / held inside each tracked run (NaN = unreached)."""
    out = np.full_like(off, np.nan)
    for run in runs_of(tracked):
        o = off[run].copy()
        have = np.isfinite(o[:, 0])
        if have.sum() >= 3:
            med = np.stack([median_filter(np.interp(run, run[have], o[have, c]), 9, mode="nearest") for c in range(3)], 1)
            have &= np.linalg.norm(np.nan_to_num(o - med), axis=1) < OUTLIER_M
        if not have.any():
            continue
        t = run[have]
        filled = np.stack([np.interp(run, t, o[have, c]) for c in range(3)], 1)
        reach = (run >= t[0] - hold) & (run <= t[-1] + hold)
        out[run[reach]] = filled[reach]
    return out
