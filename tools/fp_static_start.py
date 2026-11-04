"""Static-start frame-0 smoothing for FoundationPose tracks (NOTES Phase 8d). The Track 3 scorer aligns the whole
scene on frame 0, so one noisy first pose becomes a rigid error over the clip. Objects usually rest before the hand
reaches them: replace frame 0 by the robust mean, in the WORLD frame (camera VO), of the first K frames that stay within
TOL of their median pose, then map it back into camera 0.

  python fp_static_start.py CAMERA_VO_NPZ FP_DIR [FP_DIR ...] [--k 5] [--tol-cm 1.0] [--tol-deg 3.0] [--min 3]
Idempotent: the original pose is kept as 000000.json.orig and always re-read from there. Frames missing or off the
still set are skipped; with fewer than --min still frames frame 0 is left as tracked.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def load(p):
    j = json.load(open(p))
    w, x, y, z = j["rotation"]
    T = np.eye(4)
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = j["translation"]
    return T, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vo")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tol-cm", type=float, default=1.0)
    ap.add_argument("--tol-deg", type=float, default=3.0)
    ap.add_argument("--min", type=int, default=3)
    a = ap.parse_args()
    c2w = np.load(a.vo)["camera_to_world"]
    for d in map(Path, a.dirs):
        f0, orig = d / "000000.json", d / "000000.json.orig"
        if not f0.exists():
            print(f"{d.name}: no frame 0, skipped")
            continue
        if not orig.exists():
            shutil.copyfile(f0, orig)
        W, idx = [], []
        for i in range(min(a.k, len(c2w))):
            p = orig if i == 0 else d / f"{i:06d}.json"
            if p.exists():
                W.append(c2w[i] @ load(p)[0])
                idx.append(i)
        W = np.array(W)
        t_med = np.median(W[:, :3, 3], axis=0)
        r_med = R.from_matrix(W[:, :3, :3]).mean()
        dt = np.linalg.norm(W[:, :3, 3] - t_med, axis=1) * 100
        dr = np.degrees((r_med.inv() * R.from_matrix(W[:, :3, :3])).magnitude())
        keep = (dt <= a.tol_cm) & (dr <= a.tol_deg)
        T0, j = load(orig)
        if keep.sum() < a.min:  # frame 0 itself may be the outlier: it need not be in the still set
            shutil.copyfile(orig, f0)
            print(f"{d.name}: not still ({keep.sum()}/{len(W)} frames within {a.tol_cm} cm / {a.tol_deg} deg), frame 0 kept")
            continue
        Wm = np.eye(4)
        Wm[:3, :3] = R.from_matrix(W[keep, :3, :3]).mean().as_matrix()
        Wm[:3, 3] = W[keep, :3, 3].mean(axis=0)
        Tn = np.linalg.inv(c2w[0]) @ Wm
        x, y, z, w = R.from_matrix(Tn[:3, :3]).as_quat()
        j["rotation"], j["translation"] = [w, x, y, z], Tn[:3, 3].tolist()
        json.dump(j, open(f0, "w"), indent=4)
        moved = np.linalg.norm(Tn[:3, 3] - T0[:3, 3]) * 100
        turned = np.degrees((R.from_matrix(T0[:3, :3]).inv() * R.from_matrix(Tn[:3, :3])).magnitude())
        print(f"{d.name}: frame 0 := mean of frames {[i for i, k in zip(idx, keep) if k]} "
              f"(moved {moved:.2f} cm, turned {turned:.2f} deg)")


if __name__ == "__main__":
    main()
