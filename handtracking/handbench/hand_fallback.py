"""Per-frame fallback between two tracks of the same hand (shared by the pipeline stage and the benchmark).

A tracked hand is implausible in a frame when, in cam_a, its joints fall outside that hand's SAM2 mask or far from the
cam_a hand keypoints (e.g. a clip-level tracker that latched onto another person's hand). Those frames take the
fallback track's hand instead, if the fallback is plausible there; frames the primary track misses are also taken from
a plausible fallback. Only cam_a signals are used.

  plausibility   per frame: share of projected joints inside the (dilated) mask, mean px distance to the keypoints
  choose_fallback  per frame: True where the fallback hand replaces the primary one (a primary that fails the test
                   is replaced when the fallback passes or is closer to the keypoints / more inside the mask)
  seam_offsets   per frame: translations that spread the jump at each switch between the two tracks over a few frames
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_dilation

MIN_MASK_HIT = 0.2  # share of projected joints that must land inside the hand's mask (good hands: 0.3-1)
MASK_GROW_PX = 15  # mask dilation before the test (mask edges, thin fingers)
MAX_KP_PX = 150.0  # mean distance to the cam_a keypoints above which the hand is implausible
MIN_KP = 5  # confident keypoints needed for the keypoint test
KP_MIN_SCORE = 0.5
GROW_FRAMES = 2  # fallback stretches are widened by this many frames (failures ramp in and out)
SEAM_FRAMES = 3  # a switch's jump is spread over this many frames on each side
MAX_SEAM_M = 0.05  # larger jumps are left alone (one side is a failure, not a seam)


def plausibility(uv: np.ndarray, mask: np.ndarray | None, kp: np.ndarray, kp_score: np.ndarray):
    """uv (21,2) projected joints, mask (H,W) bool or None, kp (21,2) cam_a keypoints in the same pixels.
    Returns (mask_hit, kp_px); NaN where that test has no data."""
    hit = np.nan
    if mask is not None and mask.any():
        m = cv2.dilate(mask.astype(np.uint8), np.ones((2 * MASK_GROW_PX + 1,) * 2, np.uint8)) > 0
        p = np.round(uv).astype(int)
        inside = np.isfinite(uv).all(1) & (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < m.shape[1]) & (p[:, 1] < m.shape[0])
        hit = float(m[p[inside, 1], p[inside, 0]].sum() / len(uv))
    kp_px = np.nan
    k = (kp_score >= KP_MIN_SCORE) & np.isfinite(kp[:, 0]) & np.isfinite(uv[:, 0])
    if k.sum() >= MIN_KP:
        kp_px = float(np.linalg.norm(uv[k] - kp[k], axis=1).mean())
    return hit, kp_px


def is_plausible(hit: np.ndarray, kp_px: np.ndarray) -> np.ndarray:
    """Per frame; a test without data does not count against the hand."""
    return ~(hit < MIN_MASK_HIT) & ~(kp_px > MAX_KP_PX)


def better(hit_p, kp_p, hit_f, kp_f) -> np.ndarray:
    """Per frame: the fallback is closer to the keypoints, or (without keypoints) more inside the mask."""
    have_kp = np.isfinite(kp_p) & np.isfinite(kp_f)
    return np.where(have_kp, np.nan_to_num(kp_f, nan=np.inf) < np.nan_to_num(kp_p, nan=np.inf),
                    np.nan_to_num(hit_f, nan=-1) > np.nan_to_num(hit_p, nan=-1))


def choose_fallback(has_p, ok_p, has_f, ok_f, f_better, fill_missing: bool = True) -> np.ndarray:
    """All (n,) bool. True where the fallback hand is used; fill_missing also takes it where the primary has no hand."""
    bad_p = has_p & ~ok_p
    use = has_f & ((~has_p & ok_f & fill_missing) | (bad_p & (ok_f | f_better)))
    if GROW_FRAMES and bad_p.any():  # the frames around a failure are usually already drifting
        use |= binary_dilation(bad_p, iterations=GROW_FRAMES) & has_f & ok_f & f_better
    return use


def seam_offsets(C: np.ndarray, tracked: np.ndarray, use: np.ndarray) -> np.ndarray:
    """C (n,3) hand centroids after the swap, tracked (n,) bool, use (n,) bool -> (n,3) translations to add.
    At each switch inside a tracked run, the jump beyond the local motion is ramped out linearly over 2*SEAM_FRAMES
    frames (half on each side); jumps above MAX_SEAM_M are left alone."""
    from handbench.hand_correction import runs_of

    out = np.zeros_like(C)
    S = SEAM_FRAMES
    for run in runs_of(tracked):
        u = use[run]
        for k in np.nonzero(u[1:] != u[:-1])[0]:  # switch between run[k] and run[k + 1]
            v = [C[run[j + 1]] - C[run[j]] for j in (k - 1, k + 1) if 0 <= j and j + 1 < len(run)]
            excess = C[run[k + 1]] - C[run[k]] - (np.mean(v, 0) if v else 0)
            if np.linalg.norm(excess) > MAX_SEAM_M:
                continue
            for p in range(-S + 1, S + 1):
                j = k + p
                if 0 <= j < len(run):
                    f = (p + S - 0.5) / (2 * S)
                    out[run[j]] += excess * (f if p <= 0 else f - 1)
    return out
