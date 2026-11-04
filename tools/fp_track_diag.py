"""Per-frame health of FoundationPose tracks: silhouette IoU of the tracked mesh against the SAM2 mask, and (optional)
deviation from a second track of the same object made with another mesh (e.g. the scan).

  python fp_track_diag.py INTRINSICS MASK_DIR LABEL=FP_DIR=MESH [LABEL=FP_DIR=MESH ...] [--every N]
The first track is the reference for the deviation column: other tracks are mapped into its body frame with the median
constant transform over the first 10 frames, then compared as mean distance of the reference mesh's surface points (cm).
Prints a frame table every N frames and a summary (median IoU, frames with IoU < 0.5, first frame IoU < 0.5).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


def load_track(d, n):
    T = np.full((n, 4, 4), np.nan)
    for i in range(n):
        p = Path(d) / f"{i:06d}.json"
        if p.exists():
            j = json.load(open(p))
            w, x, y, z = j["rotation"]
            T[i] = np.eye(4)
            T[i, :3, :3] = R.from_quat([x, y, z, w]).as_matrix()
            T[i, :3, 3] = j["translation"]
    return T


def silhouette(m, T, K, hw, s):
    v = m.vertices @ T[:3, :3].T + T[:3, 3]
    z = v[:, 2]
    uv = (v[:, :2] / np.maximum(z[:, None], 1e-6)) * [K["fx"], K["fy"]] + [K["cx"], K["cy"]]
    uv = (uv * s).astype(np.int32)
    img = np.zeros(hw, np.uint8)
    ok = (z[m.faces] > 0.01).all(1)
    cv2.fillPoly(img, list(uv[m.faces[ok]]), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("intr")
    ap.add_argument("masks")
    ap.add_argument("tracks", nargs="+")
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    K = json.load(open(a.intr))
    files = sorted(Path(a.masks).glob("*.png"))
    n = len(files)
    s = 0.25
    hw = (int(K["height"] * s), int(K["width"] * s))
    masks = [cv2.resize((cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8), hw[::-1],
                        interpolation=cv2.INTER_NEAREST) for f in files]
    tr = []
    for t in a.tracks:
        lab, d, mp = t.split("=", 2)
        m = trimesh.load(mp, force="mesh")
        if len(m.faces) > 6000:
            m = m.simplify_quadric_decimation(face_count=6000)
        tr.append((lab, load_track(d, n), m))
    pts = trimesh.sample.sample_surface(tr[0][2], 1000, seed=0)[0]
    iou = np.full((len(tr), n), np.nan)
    dev = np.full((len(tr), n), np.nan)
    for k, (lab, T, m) in enumerate(tr):
        for i in range(n):
            if np.isnan(T[i, 0, 0]) or masks[i].sum() == 0:
                continue
            sil = silhouette(m, T[i], K, hw, s)
            iou[k, i] = (sil & masks[i]).sum() / max((sil | masks[i]).sum(), 1)
        if k:
            T0 = tr[0][1]
            X = [np.linalg.inv(T[i]) @ T0[i] for i in range(min(10, n)) if not np.isnan(T[i, 0, 0] + T0[i, 0, 0])]
            Xm = np.eye(4)
            Xm[:3, :3] = R.from_matrix(np.array([x[:3, :3] for x in X])).mean().as_matrix()
            Xm[:3, 3] = np.median([x[:3, 3] for x in X], axis=0)
            for i in range(n):
                A, B = T0[i], T[i] @ Xm
                dev[k, i] = np.linalg.norm((pts @ A[:3, :3].T + A[:3, 3]) - (pts @ B[:3, :3].T + B[:3, 3]), axis=1).mean() * 100
    print("frame  " + "  ".join(f"{lab:>14s}" for lab, _, _ in tr) + "   (IoU / dev cm vs " + tr[0][0] + ")")
    for i in range(0, n, a.every):
        print(f"{i:5d}  " + "  ".join(f"{iou[k, i]:6.2f} /{dev[k, i]:6.1f}" for k in range(len(tr))))
    for k, (lab, _, _) in enumerate(tr):
        bad = np.where(iou[k] < 0.5)[0]
        print(f"{lab}: IoU median {np.nanmedian(iou[k]):.2f}, frames < 0.5: {len(bad)}/{n}, first {bad[0] if len(bad) else '-'}"
              + (f"; dev median {np.nanmedian(dev[k]):.1f} p90 {np.nanpercentile(dev[k], 90):.1f} cm" if k else ""))


if __name__ == "__main__":
    main()
