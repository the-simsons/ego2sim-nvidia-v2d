"""E4: per-frame quality verifier ("second model").

Predicts the held-out error of a method's hand on a frame from cam_a-ONLY features:
  cam_a fit      iou_a, iou_a_full, kp_a, pck10_a (method vs cam_a SAM2 mask / cam_a detector keypoints)
  method         conf, wrist depth, finger-pose norm, MANO scale, betas norm
  temporal       local joint acceleration, wrist speed to the previous / next frame (same side, camera frame)
  image          hand bbox size in cam_a, distance to the image border, cam_a detector mean score / #confident joints
Target: triangulated MPJPE (tri_abc). Gradient-boosted trees, 5-fold GroupKFold by EPISODE (out-of-fold predictions).
Reports precision / recall / AUROC for "error > THR mm" and writes two derived methods:
  E4drop-<src>     flagged frames removed (source -> none: they leave the accuracy metrics, coverage drops)
  E4repl-<src>     flagged frames replaced by interpolation between the nearest unflagged frames (<= 10 frames away)

Usage: python -m handbench.experiments.e4 --src E1b-scale [--thr 20] [--p 0.5]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from handbench import format as hf
from handbench import paths
from handbench.calib import load_rig
from handbench.experiments.e1 import commit

FEATS = ["iou_a", "iou_a_full", "kp_a", "pck10_a", "conf", "z", "pose_norm", "scale", "betas_norm", "accel",
         "v_prev", "v_next", "bbox_px", "border_px", "det_score", "det_n"]


def episode_features(src, ep, pf: pd.DataFrame, cams):
    tr = hf.load(paths.RESULTS / src / ep / "hands.npz")
    kp = np.load(paths.REFS / ep / "kp2d.npz")
    A = cams["ego_cam_a"]
    fps = paths.fps()
    J = tr["joints_cam"].reshape(-1, 2, 21, 3).astype(float)
    ok = tr.scored_mask().reshape(-1, 2)
    n = J.shape[0]
    rows = []
    for _, r in pf[(pf.episode == ep) & pf.scored].iterrows():
        i, s = int(r.frame), "LR".index(r.side)
        rr = 2 * i + s
        Jt = J[i, s]
        c = Jt.mean(0)
        f = {k: r.get(k, np.nan) for k in ("iou_a", "iou_a_full", "kp_a", "pck10_a")}
        f.update(conf=float(tr["conf"][rr]), z=float(Jt[0, 2]), pose_norm=float(np.linalg.norm(tr["hand_pose"][rr])),
                 scale=float(tr["scale"][rr]), betas_norm=float(np.linalg.norm(tr["betas"][rr])))
        prev_ok = i > 0 and ok[i - 1, s]
        next_ok = i < n - 1 and ok[i + 1, s]
        f["v_prev"] = np.linalg.norm(Jt[0] - J[i - 1, s, 0]) * fps if prev_ok else np.nan
        f["v_next"] = np.linalg.norm(J[i + 1, s, 0] - Jt[0]) * fps if next_ok else np.nan
        f["accel"] = (np.linalg.norm(J[i + 1, s].mean(0) - 2 * c + J[i - 1, s].mean(0)) * fps ** 2
                      if prev_ok and next_ok else np.nan)
        uv = Jt[:, :2] / Jt[:, 2:3] * np.diag(A.K)[:2] + A.K[:2, 2]
        f["bbox_px"] = float(np.linalg.norm(uv.max(0) - uv.min(0)))
        W, H = A.size
        f["border_px"] = float(min(uv[:, 0].min(), uv[:, 1].min(), W - uv[:, 0].max(), H - uv[:, 1].max()))
        sc = kp["score"][i, s, 0]
        f["det_score"], f["det_n"] = float(sc.mean()), int((sc >= 0.5).sum())
        f.update(episode=ep, frame=i, side=r.side, target=r.get("mpjpe_tri_abc", np.nan))
        rows.append(f)
    return rows


def interpolate_rows(a, flagged_rows, max_gap=10):
    """Replace flagged rows by linear interpolation (joints, verts, transl, finger pose) between the nearest unflagged
    scored rows of the same side; rows without both neighbours within max_gap are dropped."""
    n = len(a["frame_idx"]) // 2
    good = (a["valid"] & (a["source"] == "observed")).copy()
    good[list(flagged_rows)] = False
    dropped = 0
    for r in flagged_rows:
        i, s = r // 2, r % 2
        lo = next((k for k in range(i - 1, max(-1, i - max_gap - 1), -1) if good[2 * k + s]), None)
        hi = next((k for k in range(i + 1, min(n, i + max_gap + 1)) if good[2 * k + s]), None)
        if lo is None or hi is None:
            a["valid"][r], a["source"][r], a["conf"][r] = False, "none", 0.0
            dropped += 1
            continue
        w = (i - lo) / (hi - lo)
        for k in ("joints_cam", "verts_cam", "transl", "hand_pose", "global_orient", "betas", "scale"):
            if k in a:
                a[k][r] = (1 - w) * a[k][2 * lo + s].astype(np.float64) + w * a[k][2 * hi + s].astype(np.float64)
        a["conf"][r] = 0.5
    return dropped


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--thr", type=float, default=20.0)
    ap.add_argument("--p", type=float, default=None, help="flag if predicted error > this (mm); default = thr")
    a = ap.parse_args()
    cams = load_rig(paths.rig_json())
    pf = pd.read_csv(paths.RESULTS / a.src / "per_frame_v2d.csv")
    rows = []
    for ep in paths.episodes():
        if (paths.RESULTS / a.src / ep / "hands.npz").exists():
            rows += episode_features(a.src, ep, pf, cams)
    df = pd.DataFrame(rows)
    tr = df[df.target.notna()].reset_index(drop=True)
    X, y, g = tr[FEATS].to_numpy(float), tr.target.to_numpy(), tr.episode.to_numpy()
    oof = np.full(len(tr), np.nan)
    for fit_idx, te in GroupKFold(5).split(X, y, g):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          random_state=0)
        m.fit(X[fit_idx], np.log(y[fit_idx]))
        oof[te] = np.exp(m.predict(X[te]))
    full = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                         random_state=0).fit(X, np.log(y))
    cut = a.p if a.p is not None else a.thr
    bad, flag = y > a.thr, oof > cut
    res = {"src": a.src, "n_train_rows": int(len(tr)), "thr_mm": a.thr, "flag_cut_mm": cut,
           "base_rate_bad": float(bad.mean()), "flag_rate": float(flag.mean()),
           "precision": float((bad & flag).sum() / max(flag.sum(), 1)), "recall": float((bad & flag).sum() / max(bad.sum(), 1)),
           "auroc": float(roc_auc_score(bad, oof)) if 0 < bad.mean() < 1 else None,
           "spearman_pred_vs_true": float(pd.Series(oof).corr(pd.Series(y), method="spearman"))}
    # feature importances by permutation on the OOF-free full model is costly; report simple correlations instead
    res["feature_spearman_with_error"] = {f: float(tr[f].corr(tr.target, method="spearman")) for f in FEATS}
    print(json.dumps(res, indent=1))
    out = paths.RESULTS / f"E4-{a.src}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "verifier.json", "w"), indent=1)
    # apply to ALL scored rows (also those without a tri reference); rows with a reference use their OOF prediction
    df["pred"] = np.exp(full.predict(df[FEATS].to_numpy(float)))
    key = tr.set_index(["episode", "frame", "side"]).index
    df.loc[df.set_index(["episode", "frame", "side"]).index.isin(key), "pred"] = \
        pd.Series(oof, index=key).reindex(df.set_index(["episode", "frame", "side"]).index).dropna().values
    df.to_csv(out / "predictions.csv", index=False)
    for mode in ("drop", "repl"):
        name = f"E4{mode}-{a.src}"
        for ep in paths.episodes():
            p = paths.RESULTS / a.src / ep / "hands.npz"
            if not p.exists():
                continue
            src = hf.load(p)
            arr = {k: v.copy() for k, v in src.arrays.items()}
            if "verts_cam" in arr:
                arr["verts_cam"] = arr["verts_cam"].astype(np.float32)
            d = df[(df.episode == ep) & (df.pred > cut)]
            fl = [2 * int(f) + "LR".index(s) for f, s in zip(d.frame, d.side)]
            info = {"flagged": len(fl)}
            if mode == "drop":
                for r in fl:
                    arr["valid"][r], arr["source"][r], arr["conf"][r] = False, "none", 0.0
            else:
                info["dropped_no_neighbours"] = interpolate_rows(arr, fl)
            if "verts_cam" in arr:
                arr["verts_cam"] = arr["verts_cam"].astype(np.float16)
            meta = dict(src.meta)
            meta.update(method=name, version=f"{src.meta.get('version')} + E4 verifier {mode}",
                        args={"src": a.src, "thr": a.thr, "cut": cut}, adapter_commit=commit(), info=info)
            hf.save(paths.RESULTS / name / ep / "hands.npz", arr, meta)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
