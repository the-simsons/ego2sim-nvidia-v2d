"""F2 / F3: EgoFound3R hands moved onto triangulated keypoints (translation only), then optionally smoothed.

F2 (per hand, on every frame EgoFound3R predicts, in time order):
  own      >= MIN_JOINTS triangulated joints on this frame (triangulation already requires reprojection < 3 px in
           every used view): translate the whole hand by the mean of (triangulated - EgoFound3R joint) over those
           joints, i.e. the least-squares translation. No rotation, no scale.
  carried  otherwise, if the last own shift is <= MAX_CARRY input frames old: reuse it
  none     otherwise: EgoFound3R unchanged
  rejected a shift (own or carried) that would put any joint < MIN_Z in front of the camera is not applied (EgoFound3R
           unchanged). Needed because a carried shift stays fixed while EgoFound3R's own depth can jump (Track 3 ep 22:
           a -24 cm shift carried 20 frames onto a hand EgoFound3R had moved from 0.55 to 0.18 m)
  The shift is applied to joints, the 778 vertices and the fitted MANO transl (camera frame); EgoFound3R's
  interpolated in-between frames get the shift of the previous predicted frame (never scored).
F3 = F2 + Savitzky-Golay (window SG_WINDOW, order SG_ORDER, mode interp) on joints and vertices over each run of
  consecutive predicted frames (handbench E1best / multiview_hands settings), runs shorter than the window untouched.
  Consecutive = at most EgoFound3R's own frame step apart (10 fps: 2 input frames at 20 or 15 fps, 3 at 30 fps; the
  largest step that makes up >= 10 % of its steps), so the window spans ~0.7 s.

Triangulated keypoints (input camera + one extra camera only):
  track3          handtracking/refs/<ep>/tri.npz "tri_ab" (cam_a + cam_b; cam_c held out)
  assemblyhands,  data/two_view_kp/<dataset>/<clip>.npz "tri" (detect_two_view_keypoints.py)
  hot3d

  ../.venv-bench/bin/python tools/triangulated_shift.py --dataset track3 [--src data/egofound3r]
Writes data/egofound3r_f2/<dataset>/<clip>/ and data/egofound3r_f3/<dataset>/<clip>/: export.npz (EgoFound3R export
layout, shifted / smoothed, + shift (T,2,3) m and status (T,2): 0 none, 1 own, 2 carried, 3 rejected, -1 not
predicted) and a
copy of run_meta.json; and data/egofound3r_f2/<dataset>/summary.json (status counts, shift sizes).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
MIN_JOINTS = 4
MAX_CARRY = 30
MIN_Z = 0.05
SG_WINDOW, SG_ORDER = 7, 2


def triangulated(dataset, clip):
    if dataset == "track3":
        return np.load(ROOT.parent / "refs" / clip / "tri.npz")["tri_ab"].astype(np.float64)
    return np.load(ROOT / "data" / "two_view_kp" / dataset / f"{clip}.npz")["tri"].astype(np.float64)


def f2(J, V, predicted, X):
    T = len(J)
    shift = np.zeros((T, 2, 3))
    status = np.full((T, 2), -1, np.int8)
    for s in range(2):
        last, last_t = None, -10 ** 9
        for t in range(T):
            safe = lambda d: (J[t, s, :, 2] + d[2]).min() >= MIN_Z  # noqa: E731
            if not predicted[t, s]:
                if last is not None and t - last_t <= MAX_CARRY and safe(last):
                    shift[t, s] = last  # in-between (interpolated) frames: follow the previous predicted frame
                continue
            m = np.isfinite(X[t, s, :, 0]) if t < len(X) else np.zeros(21, bool)
            if m.sum() >= MIN_JOINTS:
                d = (X[t, s, m] - J[t, s, m]).mean(0)
                if safe(d):
                    last, last_t = d, t
                    shift[t, s], status[t, s] = d, 1
                else:
                    status[t, s] = 3
            elif last is not None and t - last_t <= MAX_CARRY:
                if safe(last):
                    shift[t, s], status[t, s] = last, 2
                else:
                    status[t, s] = 3
            else:
                status[t, s] = 0
    return J + shift[:, :, None], V + shift[:, :, None], shift, status


def f3(J, V, predicted, step):
    J, V = J.copy(), V.copy()
    for s in range(2):
        idx = np.nonzero(predicted[:, s])[0]
        runs = np.split(idx, np.nonzero(np.diff(idx) > step)[0] + 1) if len(idx) else []
        for run in runs:
            if len(run) >= SG_WINDOW:
                J[run, s] = savgol_filter(J[run, s], SG_WINDOW, SG_ORDER, axis=0, mode="interp")
                V[run, s] = savgol_filter(V[run, s], SG_WINDOW, SG_ORDER, axis=0, mode="interp")
    return J, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["track3", "assemblyhands", "hot3d"])
    ap.add_argument("--src", default=str(ROOT / "data" / "egofound3r"), help="EgoFound3R exports root (F0 variant)")
    ap.add_argument("--out-suffix", default="", help="appended to egofound3r_f2 / egofound3r_f3 (e.g. _known_k)")
    a = ap.parse_args()
    src = Path(a.src) / a.dataset
    summary = {"min_joints": MIN_JOINTS, "max_carry_frames": MAX_CARRY, "savgol": [SG_WINDOW, SG_ORDER], "clips": {}}
    for run in sorted(p for p in src.iterdir() if (p / "export.npz").exists()):
        clip = run.name
        e = dict(np.load(run / "export.npz"))
        J, V, pred = e["joints_native"].astype(np.float64), e["verts778"].astype(np.float64), e["predicted"]
        X = triangulated(a.dataset, clip)
        J2, V2, shift, status = f2(J, V, pred, X)
        d = np.diff(e["high_frame_indices"])
        vals, cnt = np.unique(d, return_counts=True)
        step = int(vals[cnt >= 0.1 * len(d)].max()) if len(d) else 1
        J3, V3 = f3(J2, V2, pred, step)
        for name, (Jx, Vx) in (("egofound3r_f2", (J2, V2)), ("egofound3r_f3", (J3, V3))):
            od = ROOT / "data" / (name + a.out_suffix) / a.dataset / clip
            od.mkdir(parents=True, exist_ok=True)
            # MANO transl moves with the hand's mean vertex displacement (exact for F2's pure translation)
            dt = (Vx - V).mean(2)
            np.savez_compressed(od / "export.npz", **{**e, "joints_native": Jx.astype(np.float32),
                                                       "verts778": Vx.astype(np.float32), "transl": e["transl"] + dt,
                                                       "shift": shift, "status": status, "f0_source": str(run)})
            shutil.copy(run / "run_meta.json", od / "run_meta.json")
        own = status == 1
        summary["clips"][clip] = {
            "predicted": int(pred.sum()), "own": int(own.sum()), "carried": int((status == 2).sum()),
            "none": int((status == 0).sum()), "rejected": int((status == 3).sum()), "smoothing_step": step,
            "shift_mm_median": float(np.median(np.linalg.norm(shift[own], axis=-1)) * 1000) if own.any() else None,
            "shift_z_mm_median": float(np.median(shift[own][:, 2]) * 1000) if own.any() else None}
        print(clip, summary["clips"][clip])
    keys = ("own", "carried", "none", "rejected")
    tot = {k: sum(c[k] for c in summary["clips"].values()) for k in ("predicted",) + keys}
    summary["total"] = tot | {k + "_frac": tot[k] / max(tot["predicted"], 1) for k in keys}
    (ROOT / "data" / ("egofound3r_f2" + a.out_suffix) / a.dataset / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["total"]))


if __name__ == "__main__":
    main()
