"""Evaluate one method on one set.

  python -m handbench.run_eval --method E0 --set v2d [--episodes 12,40] [--jobs 8] [--no-masks] [--no-stereo]

Reads results/<method>/<episode>/hands.npz, writes results/<method>/per_frame.csv (one row per visible hand-frame),
per_episode.csv and metrics.json (pooled over all scored hand-frames, plus the episode-level temporal / coverage
numbers averaged over episodes).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from handbench import format as hf
from handbench import paths
from handbench.datasets.v2d import get as get_dataset
from handbench.mano import Mano

FRAME_METRICS = ["mpjpe_tri_abc", "depth_tri_abc", "lateral_tri_abc", "bias_tri_abc", "ca_mpjpe_tri_abc",
                 "mpjpe_tri_ab", "depth_tri_ab", "lateral_tri_ab", "bias_tri_ab",
                 "kp_b", "pck10_b", "kp_c", "pck10_c", "iou_b", "iou_b_full", "kp_a", "pck10_a", "iou_a", "iou_a_full",
                 "stereo_abs", "stereo_signed"]
EPISODE_METRICS = ["jitter_accel_ms2", "wrist_fast_frac", "bone_len_std_mm", "betas_std", "swaps_per_min",
                   "coverage_observed", "coverage_any", "infilled_frac_of_visible", "infilled_frac_of_valid"]


def _one(args):
    import os

    import cv2

    from handbench.metrics import evaluate_episode

    if os.environ.get("OMP_NUM_THREADS"):  # one worker per core: cap OpenCV's own pool too (else ~14 threads/worker)
        cv2.setNumThreads(int(os.environ["OMP_NUM_THREADS"]))
    method, set_name, ep, do_masks, do_stereo = args
    p = paths.RESULTS / method / ep / "hands.npz"
    if not p.exists():
        return ep, None, None
    ds = get_dataset(set_name)
    rows, epm = evaluate_episode(hf.load(p), ds.refs(ep), Mano(), do_masks, do_stereo)
    for r in rows:
        r["episode"] = ep
    return ep, rows, epm


def aggregate(df: pd.DataFrame, ep_df: pd.DataFrame) -> dict:
    s = df[df.scored]
    out = {"n_visible_hand_frames": int(len(df)), "n_scored_hand_frames": int(len(s))}
    for m in FRAME_METRICS:
        if m in s and s[m].notna().any():
            out[m] = float(s[m].mean())
            out[m + "_median"] = float(s[m].median())
            out[m + "_n"] = int(s[m].notna().sum())
    for m in EPISODE_METRICS:
        if m in ep_df:
            out[m] = float(ep_df[m].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--set", default="v2d")
    ap.add_argument("--episodes", default="")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--no-stereo", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    ds = get_dataset(a.set)
    eps = [paths.ep_name(e) for e in a.episodes.split(",")] if a.episodes else ds.episodes()
    rows, per_ep = [], []
    with ProcessPoolExecutor(a.jobs) as ex:
        for ep, r, epm in ex.map(_one, [(a.method, a.set, e, not a.no_masks, not a.no_stereo) for e in eps]):
            if r is None:
                print(f"{ep}: no hands.npz, skipped")
                continue
            rows += r
            df_e = pd.DataFrame(r)
            per_ep.append({"episode": ep, **aggregate(df_e, pd.DataFrame([epm])), **epm})
            print(f"{ep}: {len(r)} visible hand-frames, MPJPE(abc) "
                  f"{per_ep[-1].get('mpjpe_tri_abc', float('nan')):.1f} mm", flush=True)
    out = paths.RESULTS / a.method
    df = pd.DataFrame(rows)
    ep_df = pd.DataFrame(per_ep)
    df.to_csv(out / f"per_frame_{a.set}.csv", index=False)
    ep_df.to_csv(out / f"per_episode_{a.set}.csv", index=False)
    metrics = aggregate(df, ep_df)
    metas = [hf.load(paths.RESULTS / a.method / e / "hands.npz", validate=False).meta for e in eps
             if (paths.RESULTS / a.method / e / "hands.npz").exists()]
    metrics["method_meta"] = metas[0] if metas else {}
    metrics["eval"] = {"set": a.set, "episodes": len(per_ep), "wall_time_s": time.time() - t0,
                       "commit": metas[0].get("adapter_commit") if metas else None}
    json.dump(metrics, open(out / f"metrics_{a.set}.json", "w"), indent=1)
    if a.set == "v2d":
        json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()
                      if not k.endswith(("_n", "_median")) and k not in ("method_meta",)}, indent=1))


if __name__ == "__main__":
    main()
