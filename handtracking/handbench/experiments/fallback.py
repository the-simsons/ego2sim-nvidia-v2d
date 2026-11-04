"""Per-frame fallback on the benchmark: a primary method's hands, with a fallback method's hands on the frames where
the primary hand is implausible in cam_a or missing (handbench.hand_fallback). Both inputs are already in the benchmark
frame and already depth-corrected, so rows are swapped as they are.

The decision uses the cam_a SAM2 masks and the gated cam_a keypoints only, so cam_b stays an independent judge; the
cam_a IoU / keypoint metrics share those signals and are not independent for this method.

Usage: python -m handbench.experiments.fallback --primary E7tri-hawor --fallback E7tri-e0 --name E8fallback-e0
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal import savgol_filter

from handbench import format as hf
from handbench import paths
from handbench.datasets.v2d import V2DDev
from handbench.experiments.e1 import commit, set_row
from handbench.hand_correction import runs_of
from handbench.mano import Mano
from handbench.hand_fallback import (GROW_FRAMES, SEAM_FRAMES, MAX_KP_PX, MIN_MASK_HIT, better, choose_fallback, is_plausible,
                                     plausibility, seam_offsets)

SIDES = ("L", "R")


def cam_a_checks(tr, ref, n, s, kp_a, v):
    """(has, ok, hit, kp_px) per frame for side s of tracks tr."""
    A = ref.cams["ego_cam_a"]
    rows = np.arange(s, 2 * n, 2)
    has = tr["valid"][rows] & (tr["source"][rows] == "observed")
    J = tr["joints_cam"][rows].astype(float)
    hit, kpx = np.full(n, np.nan), np.full(n, np.nan)
    for i in np.nonzero(has)[0]:
        uv = J[i, :, :2] / J[i, :, 2:3] * np.diag(A.K)[:2] + A.K[:2, 2]
        hit[i], kpx[i] = plausibility(uv, ref.mask("ego_cam_a", i, SIDES[s]), kp_a[i, s], ref.kp2d_score[i, s, v])
    return has, is_plausible(hit, kpx), hit, kpx


def run_episode(args):
    prim, fb, name, fill, smooth, ep = args
    t0 = time.time()
    ref = V2DDev().refs(ep)
    A = ref.cams["ego_cam_a"]
    v = ref.kp_views.index("ego_cam_a")
    kp_a = np.stack([A.undistort_px(np.nan_to_num(ref.kp2d[:, s, v]).reshape(-1, 2)).reshape(-1, 21, 2) for s in
                     range(2)], 1) * np.diag(A.K)[:2] + A.K[:2, 2]
    P = hf.load(paths.RESULTS / prim / ep / "hands.npz")
    F = hf.load(paths.RESULTS / fb / ep / "hands.npz")
    n = min(P.n_frames, F.n_frames, ref.n_frames)
    a = {k: x.copy() for k, x in P.arrays.items()}
    if "verts_cam" in a and "verts_cam" not in F.arrays:
        del a["verts_cam"]  # rebuilt from MANO params by the evaluator
    info, mano = {}, None
    for s, side in enumerate(SIDES):
        has_p, ok_p, hit_p, kp_p = cam_a_checks(P, ref, n, s, kp_a, v)
        has_f, ok_f, hit_f, kp_f = cam_a_checks(F, ref, n, s, kp_a, v)
        use = choose_fallback(has_p, ok_p, has_f, ok_f, better(hit_p, kp_p, hit_f, kp_f), fill)
        for i in np.nonzero(use)[0]:
            r = 2 * i + s
            for k in a:
                if k not in ("frame_idx", "t", "side") and k in F.arrays:
                    a[k][r] = F.arrays[k][r]
        rows = np.arange(s, 2 * n, 2)
        tracked = a["valid"][rows] & (a["source"][rows] == "observed")
        C = np.nan_to_num(a["joints_cam"][rows].astype(float).mean(1))
        d = seam_offsets(C, tracked, use)
        a["transl"][rows] += d
        a["joints_cam"][rows] = (a["joints_cam"][rows] + d[:, None]).astype(a["joints_cam"].dtype)
        if "verts_cam" in a:
            a["verts_cam"][rows] = (a["verts_cam"][rows].astype(float) + d[:, None]).astype(a["verts_cam"].dtype)
        if smooth:  # one Savitzky-Golay pass (w7, o2) over the spliced track, as the last step of the pipeline stage
            mano = mano or Mano()
            for run in runs_of(tracked):
                if len(run) < 7:
                    continue
                r = rows[run]
                V, J = mano.forward(a["global_orient"][r], a["hand_pose"][r], a["betas"][r], a["transl"][r], side == "R",
                                    a["scale"][r])
                V, J = savgol_filter(V, 7, 2, axis=0, mode="interp"), savgol_filter(J, 7, 2, axis=0, mode="interp")
                for k, rr in enumerate(r):
                    set_row(a, rr, V[k], J[k], mano)
        info[side] = {"primary": int(has_p.sum()), "primary_implausible": int((has_p & ~ok_p).sum()),
                      "fallback_replaced": int((use & has_p).sum()), "fallback_added": int((use & ~has_p).sum()),
                      "both_implausible": int((has_p & ~ok_p & ~(has_f & ok_f)).sum())}
    meta = dict(P.meta)
    meta.update(method=name, version=f"{P.meta.get('version')}; per-frame fallback: {F.meta.get('version')}",
                args={"primary": prim, "fallback": fb, "min_mask_hit": MIN_MASK_HIT, "max_kp_px": MAX_KP_PX,
                      "grow_frames": GROW_FRAMES, "seam_frames": SEAM_FRAMES,
                      "fill_missing": fill, "smooth_after_splice": smooth},
                adapter_commit=commit(), info=info,
                consumed_stereo_depth=bool(P.meta.get("consumed_stereo_depth") or F.meta.get("consumed_stereo_depth")),
                uses_cam_b_keypoints=False, derived_wall_time_s=time.time() - t0)
    hf.save(paths.RESULTS / name / ep / "hands.npz", a, meta)
    return ep, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--fallback", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", default="")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--smooth", action="store_true", help="Savitzky-Golay pass over the spliced track")
    ap.add_argument("--replace-only", action="store_true", help="do not add fallback hands where the primary has none")
    a = ap.parse_args()
    eps = [paths.ep_name(e) for e in a.episodes.split(",")] if a.episodes else paths.episodes()
    eps = [e for e in eps if all((paths.RESULTS / m / e / "hands.npz").exists() for m in (a.primary, a.fallback))]
    tot = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for ep, info in ex.map(run_episode, [(a.primary, a.fallback, a.name, not a.replace_only, a.smooth, e) for e in eps]):
            print(ep, json.dumps(info), flush=True)
            for d in info.values():
                for k, x in d.items():
                    tot[k] = tot.get(k, 0) + x
    print("total over episodes and sides:", tot)


if __name__ == "__main__":
    main()
