"""Score hand predictions against true 3D hands on AssemblyHands (val clips) and HOT3D (Aria clips).

Inputs per clip: data/<dataset>_input/<clip>.json (K, fps) and <clip>_gt.npz (prepare_assemblyhands.py /
export_hot3d_gt.py: joints_cam (T,2,21,3) in the input camera, m, OpenPose order; visible (T,2); valid (T,2,21)).
Predictions (--source):
  egofound3r   data/egofound3r/<dataset>/<clip>/export.npz (egofound3r_export.py): joints_native, predicted
               (EgoFound3R's own frames only; its interpolation to the input rate is not scored)
  hands        data/<dir>/<clip>/hands.npz in the npz layout below (--pred-dir), for other methods:
               joints_cam (T,2,21,3), predicted (T,2)
A hand-frame is scored when the hand is visible in the input camera, has >= 10 valid GT joints, and the method
predicted that hand on that frame. Coverage = scored / visible-with-labels over the frames the method predicts on.

Per scored hand-frame (mm unless noted): mpjpe; depth / lateral (error along / across the camera ray of each joint);
bias (signed along-ray error, + = too far); ca_mpjpe (both centred on their joint mean: pose + shape); pa_mpjpe
(after similarity Procrustes); kp_px (2D error in the input image) and pck10 (share of joints within 10 px).
AssemblyHands: also all of the above after removing a per-joint offset in the hand's own frame (wrist, index MCP,
pinky MCP), estimated on the OTHER sequence (held out by sequence/subject): "+offset" rows.
Temporal: jitter (mean joint acceleration, m/s², from 3 consecutive predicted frames at their true times, gaps
<= 0.15 s), bone-length std (mm).

  ../.venv-bench/bin/python tools/score_groundtruth.py --dataset assemblyhands --method F0
Writes results/<dataset>/<method>/{metrics.json,per_frame.csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
MIN_JOINTS = 10
MAX_GAP_S = 0.15


def load_pred(dataset, clip, source, pred_dir):
    if source == "egofound3r":
        e = np.load(Path(pred_dir or ROOT / "data" / "egofound3r") / dataset / clip / "export.npz")
        return e["joints_native"].astype(np.float64), e["predicted"]
    e = np.load(Path(pred_dir) / clip / "hands.npz")
    return e["joints_cam"].astype(np.float64), e["predicted"]


def hand_frame(J):
    """(n,21,3) -> (n,3,3) rotation, columns = hand axes (x to index MCP, z = palm normal)."""
    x = J[:, 5] - J[:, 0]
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x, J[:, 17] - J[:, 0])
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    return np.stack([x, np.cross(z, x), z], -1)


def procrustes(P, G):
    """Similarity-align P to G (both (21,3)); returns aligned P."""
    mp, mg = P.mean(0), G.mean(0)
    p, g = P - mp, G - mg
    U, S, Vt = np.linalg.svd(p.T @ g)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = (U @ D @ Vt).T
    s = (S * np.diag(D)).sum() / (p ** 2).sum()
    return s * p @ R.T + mg


def errors(P, G, m, K):
    e = P[m] - G[m]
    r = G[m] / np.linalg.norm(G[m], axis=-1, keepdims=True)
    along = (e * r).sum(-1)
    lat = np.linalg.norm(e - along[:, None] * r, axis=-1)
    ec = (P[m] - P[m].mean(0)) - (G[m] - G[m].mean(0))
    pa = procrustes(P[m], G[m]) - G[m]
    uv = lambda X: (X @ K.T)[:, :2] / (X @ K.T)[:, 2:]  # noqa: E731
    px = np.linalg.norm(uv(P[m]) - uv(G[m]), axis=-1)
    return {"mpjpe": 1000 * np.linalg.norm(e, axis=-1).mean(), "depth": 1000 * np.abs(along).mean(),
            "lateral": 1000 * lat.mean(), "bias": 1000 * along.mean(),
            "ca_mpjpe": 1000 * np.linalg.norm(ec, axis=-1).mean(), "pa_mpjpe": 1000 * np.linalg.norm(pa, axis=-1).mean(),
            "kp_px": px.mean(), "pck10": (px < 10).mean()}


def jitter_and_bones(J, scored, t):
    acc, bstd = [], []
    for s in range(2):
        idx = np.nonzero(scored[:, s])[0]
        for a, b, c in zip(idx, idx[1:], idx[2:]):
            d1, d2 = t[b] - t[a], t[c] - t[b]
            if max(d1, d2) <= MAX_GAP_S:
                v1, v2 = (J[b, s] - J[a, s]) / d1, (J[c, s] - J[b, s]) / d2
                acc.append(np.linalg.norm(2 * (v2 - v1) / (d1 + d2), axis=-1).mean())
        if len(idx) >= 3:
            bl = np.stack([np.linalg.norm(J[idx, s, c] - J[idx, s, p], axis=-1) for p, c in BONES], 1)
            bstd.append(1000 * bl.std(0).mean())
    return acc, bstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["assemblyhands", "hot3d"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--source", default="egofound3r", choices=["egofound3r", "hands"])
    ap.add_argument("--pred-dir", default=None, help="hands: <dir>/<clip>/hands.npz; egofound3r: exports root "
                    "(default data/egofound3r, e.g. data/egofound3r_f2)")
    a = ap.parse_args()
    inp = ROOT / "data" / f"{a.dataset}_input"
    rows, acc, bstd, n_vis = [], [], [], 0
    for meta_p in sorted(p for p in inp.glob("*.json") if not p.stem.endswith("_rootK")):
        meta, clip = json.loads(meta_p.read_text()), meta_p.stem
        g = np.load(inp / f"{clip}_gt.npz")
        G, vis, val = g["joints_cam"].astype(np.float64), g["visible"], g["valid"]
        P, pred = load_pred(a.dataset, clip, a.source, a.pred_dir)
        T = min(len(G), len(P))
        K, t = np.array(meta["K"]), np.arange(T) / meta["fps"]
        labelled = vis[:T] & (val[:T].sum(-1) >= MIN_JOINTS)
        on_pred_frames = pred[:T].any(1)  # frames the method predicts on (EgoFound3R: its own rate)
        n_vis += int((labelled & on_pred_frames[:, None]).sum())
        scored = labelled & pred[:T]
        a_, b_ = jitter_and_bones(P[:T], scored, t)
        acc += a_
        bstd += b_
        for i, s in zip(*np.nonzero(scored)):
            rows.append({"clip": clip, "sequence": meta.get("sequence", clip), "frame": int(i), "side": "LR"[s],
                         "P": P[i, s], "G": G[i, s], "m": val[i, s], "K": K})
    out = {}

    def summarise(key, P_of):
        es = [errors(P_of(r), r["G"], r["m"], r["K"]) for r in rows]
        for r, e in zip(rows, es):
            r[key] = e
        return {k: float(np.mean([e[k] for e in es])) for k in es[0]} | {
            "mpjpe_median": float(np.median([e["mpjpe"] for e in es]))}

    out["raw"] = summarise("raw", lambda r: r["P"])
    if a.dataset == "assemblyhands":  # per-joint offset in the hand frame, fitted on the other sequence
        seqs = sorted({r["sequence"] for r in rows})
        off = {}
        for sq in seqs:
            for side in "LR":
                fit = [r for r in rows if r["sequence"] != sq and r["side"] == side and r["m"].all()]
                R = hand_frame(np.stack([r["G"] for r in fit]))
                d = np.stack([r["P"] - r["G"] for r in fit])
                off[sq, side] = np.einsum("nij,nki->nkj", R, d).mean(0)  # (21,3) in hand axes
        out["offset_mm_mean_norm"] = {f"{k[0].split('_')[3]}_{k[1]}": float(1000 * np.linalg.norm(v, axis=-1).mean())
                                      for k, v in off.items()}

        def corrected(r):
            Rg = hand_frame(r["G"][None])[0]  # GT hand axes (offset defined against GT's frame)
            return r["P"] - off[r["sequence"], r["side"]] @ Rg.T

        out["+offset"] = summarise("off", corrected)
    out.update({"n_scored": len(rows), "n_visible_on_predicted_frames": n_vis,
                "coverage": len(rows) / n_vis if n_vis else float("nan"),
                "jitter_ms2": float(np.mean(acc)) if acc else float("nan"),
                "bone_std_mm": float(np.mean(bstd)) if bstd else float("nan"),
                "per_sequence_mpjpe": {sq: float(np.mean([r["raw"]["mpjpe"] for r in rows if r["sequence"] == sq]))
                                       for sq in sorted({r["sequence"] for r in rows})},
                "method": a.method, "source": a.source, "dataset": a.dataset})
    res = ROOT / "results" / a.dataset / a.method
    res.mkdir(parents=True, exist_ok=True)
    (res / "metrics.json").write_text(json.dumps(out, indent=1))
    keys = ["clip", "frame", "side"]
    with open(res / "per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + [f"raw_{k}" for k in rows[0]["raw"]])
        for r in rows:
            w.writerow([r[k] for k in keys] + [f"{v:.4f}" for v in r["raw"].values()])
    print(json.dumps({k: v for k, v in out.items() if k != "per_sequence_mpjpe"}, indent=1))


if __name__ == "__main__":
    main()
