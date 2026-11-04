"""Unit tests for the per-frame fallback rules (handbench.hand_fallback) and HaMeR's camera conversion.

  python -m pytest handbench/tests/test_hand_fallback.py   (or: python -m handbench.tests.test_hand_fallback)
"""
from __future__ import annotations

import numpy as np

from handbench.hand_fallback import (GROW_FRAMES, MAX_KP_PX, MAX_SEAM_M, MIN_MASK_HIT, SEAM_FRAMES, better,
                                     choose_fallback, is_plausible, plausibility, seam_offsets)
from handbench.pipeline.hamer_pass import to_camera


def test_plausibility_mask_and_keypoints():
    mask = np.zeros((100, 100), bool)
    mask[40:60, 40:60] = True
    uv = np.full((21, 2), 50.0)
    kp, sc = uv + 3.0, np.ones(21)
    hit, kpx = plausibility(uv, mask, kp, sc)
    assert hit == 1.0 and abs(kpx - np.hypot(3, 3)) < 1e-6
    hit, kpx = plausibility(uv + 40, mask, kp, sc)  # far outside the (dilated) mask, ~60 px from the keypoints
    assert hit == 0.0 and kpx > 50
    hit, kpx = plausibility(uv, None, kp, np.zeros(21))  # no mask, no confident keypoints: no data
    assert np.isnan(hit) and np.isnan(kpx)


def test_is_plausible_missing_data_does_not_count():
    hit = np.array([MIN_MASK_HIT + 0.1, MIN_MASK_HIT - 0.1, np.nan, np.nan])
    kpx = np.array([10.0, 10.0, MAX_KP_PX + 1, np.nan])
    assert is_plausible(hit, kpx).tolist() == [True, False, False, True]


def test_better_prefers_keypoints_then_mask():
    assert better(np.array([0.5]), np.array([80.0]), np.array([0.1]), np.array([20.0]))[0]  # closer to keypoints
    assert not better(np.array([0.5]), np.array([np.nan]), np.array([0.3]), np.array([20.0]))[0]  # no common kp: mask
    assert better(np.array([0.2]), np.array([np.nan]), np.array([0.9]), np.array([np.nan]))[0]


def test_choose_fallback_replaces_bad_and_fills_missing():
    n = 20
    has_p = np.ones(n, bool)
    has_p[15:] = False  # primary misses the last frames
    ok_p = np.ones(n, bool)
    ok_p[8] = False  # one implausible primary frame
    has_f, ok_f = np.ones(n, bool), np.ones(n, bool)
    f_better = np.zeros(n, bool)
    f_better[8 - GROW_FRAMES:8 + GROW_FRAMES + 1] = True
    use = choose_fallback(has_p, ok_p, has_f, ok_f, f_better, fill_missing=True)
    assert use[8] and use[15:].all()
    assert use[8 - GROW_FRAMES:8 + GROW_FRAMES + 1].all()  # failures are widened where the fallback is better
    assert not use[:8 - GROW_FRAMES].any() and not use[8 + GROW_FRAMES + 1:15].any()
    use = choose_fallback(has_p, ok_p, has_f, ok_f, f_better, fill_missing=False)
    assert not use[15:].any()
    ok_f[8] = False  # fallback implausible and not better -> keep the primary
    assert not choose_fallback(has_p, ok_p, has_f, ok_f, np.zeros(n, bool), True)[8]


def test_seam_offsets_spread_small_jump_and_skip_large():
    n = 20
    C = np.zeros((n, 3))
    C[:, 0] = np.arange(n) * 0.001  # steady motion
    use = np.zeros(n, bool)
    use[10:] = True
    jump = 0.02
    C[10:, 0] += jump  # switch to the fallback at frame 10 with a 2 cm jump
    d = seam_offsets(C, np.ones(n, bool), use)
    C2 = C + d
    steps = np.diff(C2[:, 0])
    assert np.abs(steps - 0.001).max() < jump / (2 * SEAM_FRAMES) + 1e-6  # no single step carries the jump
    assert np.allclose(d[:10 - SEAM_FRAMES], 0) and np.allclose(d[10 + SEAM_FRAMES:], 0)
    C[10:, 0] += MAX_SEAM_M  # beyond MAX_SEAM_M: a failure, not a seam -> left alone
    assert np.allclose(seam_offsets(C, np.ones(n, bool), use), 0)


def test_to_camera_keeps_shape_pixel_and_image_size():
    K = np.array([[1150.0, 0, 1010.0], [0, 1150.0, 765.0], [0, 0, 1.0]])
    size, fv = (2028, 1520), 5000 / 256 * 2028
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.03, (50, 3)) + np.array([0.6, -0.3, 25.0])  # a hand-sized blob, far away for the huge focal
    Y = to_camera(X, fv, size, K)
    assert np.allclose(Y - Y.mean(0), X - X.mean(0))  # shape unchanged (pure translation)
    c, d = X.mean(0), Y.mean(0)
    uv_virtual = c[:2] / c[2] * fv + np.array([size[0] / 2, size[1] / 2])
    assert np.allclose(d[:2] / d[2] * np.diag(K)[:2] + K[:2, 2], uv_virtual)  # centroid on the same pixel
    span = lambda P, f: (np.ptp(P[:, 0]) / P[:, 2].mean()) * f  # noqa: E731
    assert abs(span(Y, 1150.0) / span(X, fv) - 1) < 0.01  # same image size under cam_a's focal
    assert abs(d[2] - c[2] * 1150.0 / fv) < 1e-9


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
