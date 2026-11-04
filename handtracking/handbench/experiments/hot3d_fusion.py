"""HOT3D ground-truth study: correct HaWoR hands with all three cameras (triangulated keypoints + SGBM stereo).

Inputs, none of them derived from GT:
  hands       HaWoR raw camera-frame hands (hot3d_floor.hawor_hands, as hot3d_check "raw")
  keypoints   build_refs' RTMPose crops + triangulation on the three Aria views (hot3d_floor.detect_clip, source
              "hawor": crops placed and gated with HaWoR's own silhouettes, not GT ones). Only joints whose views include
              the right mono camera b are used (a+b, a+b+c): on Aria the RGB camera is 1.5 cm from the left mono
              camera, so a+c joints carry no depth (hot3d_floor)
  stereo      SGBM on the two mono cameras (13.9 cm), hot3d_check's virtual pinholes and silhouette ratio
Variants (per hand, over runs of consecutive frames where HaWoR observes it):
  raw                   HaWoR as is
  tri-shift             frames with >= MIN_J usable joints: hand translated by the median joint offset to the triangulated
                        joints (offsets with a median residual > MAX_RES after the shift are rejected: wrong-hand matches)
  tri-sim               same frames, similarity fit (umeyama, >= 6 joints; shift with 4-5)
  tri-interp            tri-shift offsets, outliers vs a 9-frame running median dropped, linearly interpolated between
                        frames that have one and held for up to HOLD frames past the first / last
  tri-interp+sgbm       frames tri-interp cannot reach: one stereo scale about the camera per uncovered stretch (median
                        of the frames whose silhouette has >= MIN_STEREO valid depth), if within 0.7-1.4
  ... +savgol           Savitzky-Golay (w7, o2) over runs, as E1best
Scored like hot3d_check: GT hand in front of the camera with its wrist inside the image, and HaWoR observes that side.

Usage: python -m handbench.experiments.hot3d_fusion [--jobs 7] [--skip-detect] -> results/hot3d_fusion.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from handbench import paths
from handbench import build_refs as B
from handbench.datasets import hot3d as H
from handbench.experiments import hot3d_check as C
from handbench.experiments import hot3d_floor as F
from handbench.fit import umeyama
from handbench.stereo import StereoDepth

KP = paths.ROOT / "data" / "hot3d_fusion"
MIN_J, MAX_RES, HOLD, MIN_STEREO = 4, 0.03, 30, 0.3
BASE = ("raw", "tri-shift", "tri-sim", "tri-interp", "tri-interp+sgbm")
METHODS = BASE + ("raw+savgol", "tri-interp+savgol", "tri-interp+sgbm+savgol")


def runs_of(mask):
    idx = np.nonzero(mask)[0]
    return np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1) if len(idx) else []


def stereo_scale(V, faces, K, D):
    """(median stereo / rendered depth over the eroded silhouette, fraction of silhouette pixels with stereo)."""
    R = C.render_depth_half(V, faces, K)
    m = cv2.erode((R > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    g = m & (D > 0)
    if m.sum() < 30 or g.sum() < 30:
        return np.nan, g.sum() / max(m.sum(), 1)
    return float(np.median(D[g] / R[g])), g.sum() / m.sum()


def interp_offsets(off, obs):
    """off (n,3) NaN where no tri offset; outlier-reject, interpolate inside each observed run, hold HOLD frames."""
    out = np.full_like(off, np.nan)
    for run in runs_of(obs):
        o = off[run].copy()
        have = np.isfinite(o[:, 0])
        if have.sum() >= 3:
            med = np.stack([median_filter(np.interp(run, run[have], o[have, c]), 9, mode="nearest") for c in range(3)], 1)
            have &= np.linalg.norm(np.nan_to_num(o - med), axis=1) < 0.04
        if not have.any():
            continue
        t = run[have]
        filled = np.stack([np.interp(run, t, o[have, c]) for c in range(3)], 1)
        reach = (run >= t[0] - HOLD) & (run <= t[-1] + HOLD)
        out[run[reach]] = filled[reach]
    return out


def smooth(J, obs):
    J = J.copy()
    for run in runs_of(obs):
        if len(run) >= 7:
            J[run] = savgol_filter(J[run], 7, 2, axis=0, mode="interp")
    return J


def fuse_clip(cp, keep_joints=False):
    """-> (clip name, per-frame metric rows, info); with keep_joints also {side: (G, {method: joints})} for figures."""
    mano = H.Hot3dMano()
    K = H.pinhole_K()
    clip = H.Clip(cp)
    ex = np.load(C.OUT / cp.stem / "export.npz")
    d = np.load(KP / f"{cp.stem}.npz")
    cams, maps, (left, right), _ = C.rig(clip)
    cl = [cams[v] for v in B.VIEWS]
    sd = StereoDepth(cams, out_scale=0.5)
    n = min(clip.n, ex["observed"].shape[1])
    r = B.triangulate_kp(cl, d["kp"], F.gated(d, "gt-mask"))  # "gt-mask" gate on HaWoR silhouettes here (see detect)
    T, U = r["tri_abc"][:n].astype(float), r["tri_abc_views"][:n]
    usable = np.isfinite(T[..., 0]) & U[..., 1]
    depth = {}
    rows, info, joints = [], {}, {}
    for s, side, is_right in F.SIDES:
        faces = mano.mano.faces_for(is_right)
        Jr, Vr = F.hawor_hands(clip, ex, mano, s, is_right, n)
        obs = np.isfinite(Jr[:, 0, 0])
        # GT, hot3d_check's scoring rule
        G = np.full((n, 21, 3), np.nan)
        for i in range(n):
            g, _ = C.gt_in_P(clip, mano, i, side, is_right)
            if g is not None and (g[:, 2] > 0.1).all():
                w = g[0, :2] / g[0, 2] * K[0, 0] + K[:2, 2]
                if (0 <= w).all() and (w < H.SIZE_PIN).all():
                    G[i] = g
        off = np.full((n, 3), np.nan)
        J = {m: Jr.copy() for m in BASE}
        for i in np.nonzero(obs)[0]:
            u = usable[i, s]
            if u.sum() < MIN_J:
                continue
            o = np.median(T[i, s, u] - Jr[i, u], 0)
            if np.median(np.linalg.norm(T[i, s, u] - (Jr[i, u] + o), axis=1)) > MAX_RES:
                continue
            off[i] = o
            J["tri-shift"][i] = Jr[i] + o
            if u.sum() >= 6:
                sc_, R, t = umeyama(Jr[i, u], T[i, s, u])
                J["tri-sim"][i] = sc_ * Jr[i] @ R.T + t
            else:
                J["tri-sim"][i] = Jr[i] + o
        oi = interp_offsets(off, obs)
        cov = np.isfinite(oi[:, 0])
        J["tri-interp"][cov] = Jr[cov] + oi[cov, None]
        J["tri-interp+sgbm"] = J["tri-interp"].copy()
        sg = np.zeros(n, bool)
        for run in runs_of(obs & ~cov):
            ss = []
            for i in run:
                if i not in depth:
                    imL = cv2.remap(clip.image(i, left), *maps[left], cv2.INTER_LINEAR)
                    imR = cv2.remap(clip.image(i, right), *maps[right], cv2.INTER_LINEAR)
                    depth[i] = sd.compute(imL, imR)
                k, frac = stereo_scale(Vr[i], faces, K, depth[i])
                if frac >= MIN_STEREO and np.isfinite(k):
                    ss.append(k)
            if ss and 0.7 < np.median(ss) < 1.4:
                J["tri-interp+sgbm"][run] = Jr[run] * np.median(ss)
                sg[run] = True
        J["raw+savgol"] = smooth(J["raw"], obs)
        J["tri-interp+savgol"] = smooth(J["tri-interp"], obs)
        J["tri-interp+sgbm+savgol"] = smooth(J["tri-interp+sgbm"], obs)
        scored = obs & np.isfinite(G[:, 0, 0])
        has_tri = np.isfinite(off[:, 0])
        joints[side] = (G, J)
        info[side] = {"scored": int(scored.sum()), "tri_frames": float(has_tri[scored].mean()) if scored.any() else 0,
                      "tri_interp_frames": float(cov[scored].mean()) if scored.any() else 0,
                      "sgbm_frames": float(sg[scored].mean()) if scored.any() else 0}
        c3 = scored[1:-1] & scored[:-2] & scored[2:]
        for m in METHODS:
            met = C.metrics(J[m][scored], G[scored])
            for k_, i in enumerate(np.nonzero(scored)[0]):
                rows.append({"clip": cp.stem, "side": side, "method": m, "tri": bool(has_tri[i]), "covered": bool(cov[i]),
                             **{a: float(b[k_]) for a, b in met.items()}})
            if c3.any():
                kk = np.nonzero(c3)[0] + 1
                acc = np.linalg.norm(J[m][kk + 1] - 2 * J[m][kk] + J[m][kk - 1], axis=-1).mean(1) * 30 ** 2
                info[side][f"jitter_{m}"] = float(acc.mean())
    return (cp.stem, rows, info, joints) if keep_joints else (cp.stem, rows, info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--skip-detect", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    clips = [c for c in H.clips() if (C.OUT / c.stem / "export.npz").exists()]
    if not a.skip_detect:
        todo = [c for c in clips if not (KP / f"{c.stem}.npz").exists()]
        with ProcessPoolExecutor(a.jobs) as ex:
            for name, dt in ex.map(F.detect_clip, todo, ["hawor"] * len(todo), [KP] * len(todo)):
                print(f"{name}: keypoints in {dt:.0f}s", flush=True)
    import pandas as pd

    rows, per_clip = [], {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, r, info in ex.map(fuse_clip, clips):
            rows += r
            per_clip[name] = info
            print(name, json.dumps(info), flush=True)
    df = pd.DataFrame(rows)
    keys = ["mpjpe", "depth", "lateral", "bias", "pose"]
    jit = {m: float(np.nanmean([per_clip[c][s].get(f"jitter_{m}", np.nan) for c in per_clip for s in per_clip[c]]))
           for m in METHODS}
    summary = {m: {k: float(df[df.method == m][k].mean()) for k in keys}
               | {"mpjpe_median": float(df[df.method == m].mpjpe.median()), "jitter": jit[m],
                  "n_hand_frames": int((df.method == m).sum())} for m in METHODS}
    split = {m: {"tri_frames": float(df[(df.method == m) & df.tri].mpjpe.mean()),
                 "covered_frames": float(df[(df.method == m) & df.covered].mpjpe.mean()),
                 "uncovered_frames": float(df[(df.method == m) & ~df.covered].mpjpe.mean())} for m in METHODS}
    res = {"summary": summary, "mpjpe_by_frame_type": split,
           "per_clip_mpjpe": df.groupby(["clip", "method"]).mpjpe.mean().unstack().round(1).to_dict(orient="index"),
           "per_clip_info": per_clip, "params": {"min_joints": MIN_J, "max_residual_m": MAX_RES, "hold_frames": HOLD,
                                                 "min_stereo_frac": MIN_STEREO}, "wall_time_s": time.time() - t0}
    json.dump(res, open(paths.RESULTS / "hot3d_fusion.json", "w"), indent=1)
    print(pd.DataFrame(summary).T.round(1).to_string())
    print(pd.DataFrame(split).T.round(1).to_string())


if __name__ == "__main__":
    main()
