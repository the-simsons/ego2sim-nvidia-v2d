"""Phase 1.3: how sensitive is the held-out cam_b view to hand depth errors?

Typical hand depth Z = median SGBM stereo depth inside the SAM2 hand masks (every --stride-th frame of every episode).
Reports, for dZ = 1/2/5 cm, the cam_b pixel shift (small-angle formula f_b * b * dZ / Z^2 and the exact reprojection of
the real hand pixels), and the triangulation depth uncertainty for +-1 px keypoint noise for the a-b, a-c, b-c pairs and
a+b+c. Writes refs/sensitivity.json.

Usage: python -m handbench.tools.sensitivity [--stride 10] [--jobs 12]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from handbench import paths
from handbench.adapters.v2d_upstream import UpstreamCamera, anycalib_dir
from handbench.calib import load_rig
from handbench.refs_io import hand_track_ids, read_frames
from handbench.stereo import StereoDepth


def episode_depths(args):
    ep, stride = args
    cams = load_rig(paths.rig_json())
    sd = StereoDepth(cams)
    U = UpstreamCamera(paths.CACHE / paths.ep_name(ep), cams["ego_cam_a"], anycalib_dir(ep))
    tracks = hand_track_ids(ep)
    out = []
    n = paths.num_frames(ep)
    for i in range(0, n, stride):
        fr = read_frames(ep, i, ("ego_cam_b", "ego_cam_c"))
        if fr is None:
            continue
        depth = sd.compute(fr["ego_cam_c"], fr["ego_cam_b"])
        for tid, side in tracks.items():
            mp = paths.CACHE / paths.ep_name(ep) / "masks" / str(tid) / f"{i:06d}.png"
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            mb = U.warp_to_bench(m) > 0
            if mb.sum() < 400:
                continue
            d = depth[mb]
            d = d[d > 0]
            if len(d) < 50:
                continue
            ys, xs = np.nonzero(mb)
            out.append({"ep": paths.ep_index(ep), "frame": i, "side": side, "z": float(np.median(d)),
                        "valid_frac": float(len(d) / mb.sum()), "u": float(xs.mean()), "v": float(ys.mean())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    eps = paths.episodes()
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = [r for rs in ex.map(episode_depths, [(e, a.stride) for e in eps]) for r in rs]
    cams = load_rig(paths.rig_json())
    A, B, C = cams["ego_cam_a"], cams["ego_cam_b"], cams["ego_cam_c"]
    fa, fb, fc = A.K[0, 0], B.K[0, 0], C.K[0, 0]
    b_ab = np.linalg.norm(B.center_in_a)
    b_ac = np.linalg.norm(C.center_in_a)
    b_bc = np.linalg.norm(B.center_in_a - C.center_in_a)
    z = np.array([r["z"] for r in rows])
    zq = {q: float(np.percentile(z, q)) for q in (10, 25, 50, 75, 90)}
    res = {"n_hand_frames": len(rows), "hand_depth_m": zq, "f_px": {"a": fa, "b": fb, "c": fc},
           "baseline_m": {"a-b": b_ab, "a-c": b_ac, "b-c": b_bc}, "cam_b_shift_px": {}, "tri_sigma_z_cm_per_px": {}}
    # exact: move real hand centres along the cam_a ray by dZ, reproject to cam_b
    for dz in (0.01, 0.02, 0.05):
        approx = {f"Z={k}": fb * b_ab * dz / v ** 2 for k, v in zq.items()}
        ex = []
        for r in rows:
            n = A.undistort_px(np.array([[r["u"], r["v"]]]))[0]  # bench pinhole: undistort is identity-ish on K
            ray = np.r_[(np.array([r["u"], r["v"]]) - A.K[:2, 2]) / np.diag(A.K)[:2], 1.0]
            p0, p1 = B.project(ray * r["z"]), B.project(ray * (r["z"] + dz))
            ex.append(np.linalg.norm(p1 - p0))
        ex = np.array(ex)
        res["cam_b_shift_px"][f"{int(dz * 100)}cm"] = {"formula_at_quantiles": approx,
                                                       "exact_median": float(np.median(ex)),
                                                       "exact_p10_p90": [float(np.percentile(ex, 10)), float(np.percentile(ex, 90))],
                                                       "frac_below_3px": float((ex < 3).mean())}
    for name, (f1, f2, b) in {"a-b": (fa, fb, b_ab), "a-c": (fa, fc, b_ac), "b-c": (fb, fc, b_bc)}.items():
        res["tri_sigma_z_cm_per_px"][name] = {f"Z={k}": 100 * v ** 2 / b * np.hypot(1 / f1, 1 / f2) for k, v in zq.items()}
    # three views: information adds (each pair's 1/sigma^2 approximately; a,b,c independent noise -> use b-c + a)
    res["tri_sigma_z_cm_per_px"]["a+b+c(approx)"] = {
        k: 1 / np.sqrt(sum(1 / res["tri_sigma_z_cm_per_px"][p][k] ** 2 for p in ("a-b", "a-c", "b-c")) * 1.5)
        for k in res["tri_sigma_z_cm_per_px"]["a-b"]}
    json.dump({"summary": res, "rows": rows}, open(paths.REFS / "sensitivity.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
