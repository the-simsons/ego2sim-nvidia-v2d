"""Compare stereo depth sources against the triangulated pseudo-GT at joint pixels (no method involved).

For every visible hand-frame with tri_abc joints: look up each source's depth (refs/<ep>/stereo*.npz, half-res crops in
the benchmark camera) at the projected joint pixel, inside the hand mask, and compare with the joint's triangulated
depth. Reports per source: coverage (joints with valid depth, mask pixels with valid depth), median signed and
absolute error, MAD, and the share of joints within 1 / 2 cm.

Usage: python -m handbench.tools.stereo_check [--sources sgbm,fs]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from handbench import paths
from handbench.calib import load_rig
from handbench.datasets.v2d import StereoCrops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="sgbm,fs")
    a = ap.parse_args()
    A = load_rig(paths.rig_json())["ego_cam_a"]
    src = a.sources.split(",")
    err = {s: [] for s in src}
    cov_joint = {s: [0, 0] for s in src}
    cov_mask = {s: [0, 0] for s in src}
    for ep in paths.episodes():
        T = np.load(paths.REFS / ep / "tri.npz")["tri_abc"]
        crops = {s: StereoCrops(paths.REFS / ep / ("stereo.npz" if s == "sgbm" else f"stereo_{s}.npz")) for s in src}
        n = T.shape[0]
        for i in range(n):
            for k, side in enumerate("LR"):
                g = np.isfinite(T[i, k, :, 0])
                got = {s: c.get(i, side) for s, c in crops.items()}
                if any(v is None for v in got.values()):
                    continue
                for s, (D, M, box) in got.items():
                    cov_mask[s][0] += int((M & (D > 0)).sum())
                    cov_mask[s][1] += int(M.sum())
                if g.sum() < 3:
                    continue
                X = T[i, k, g].astype(float)
                sc = crops[src[0]].scale
                uv = X[:, :2] / X[:, 2:3] * np.diag(A.K)[:2] + A.K[:2, 2]
                uvh = (uv + 0.5) * sc - 0.5
                for s, (D, M, box) in got.items():
                    q = np.round(uvh - box[:2]).astype(int)
                    ok = (q[:, 0] >= 0) & (q[:, 0] < D.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < D.shape[0])
                    d = np.zeros(len(X))
                    d[ok] = D[q[ok, 1], q[ok, 0]]
                    m = np.zeros(len(X), bool)
                    m[ok] = M[q[ok, 1], q[ok, 0]]
                    v = ok & m & (d > 0)
                    cov_joint[s][0] += int(v.sum())
                    cov_joint[s][1] += int((ok & m).sum())
                    err[s].extend((d[v] - X[v, 2]).tolist())
    out = {}
    for s in src:
        e = 1000 * np.array(err[s])
        out[s] = {"joints_compared": len(e), "joint_coverage": cov_joint[s][0] / max(cov_joint[s][1], 1),
                  "mask_pixel_coverage": cov_mask[s][0] / max(cov_mask[s][1], 1),
                  "median_signed_mm": float(np.median(e)), "median_abs_mm": float(np.median(np.abs(e))),
                  "mad_mm": float(np.median(np.abs(e - np.median(e)))),
                  "within_10mm": float((np.abs(e) < 10).mean()), "within_20mm": float((np.abs(e) < 20).mean())}
    print(json.dumps(out, indent=1))
    json.dump(out, open(paths.REFS / "stereo_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
