"""Grip library for the M grip reward (genv.make_grip_reward): known palm-frame grips per hand and object, and when they apply.

  grip_lib.py <episode> <out.npz> [--from_gdiag <gdiag npz>:<hand>:<obj>] [--from_teacher <labels dir>]
Grips (object pose in the palm frame), only from sources with the same mesh as the episode:
  --from_gdiag  an M + G diagnostic rollout (gdiag.py) where the grip held: median over frames touching the object with
                its error < 3 cm (e.g. ep30's sweeper grip for ep19)
  --from_teacher a teacher's labelled rollouts (label.py *_data.npz): per hand and object, median over acquire / hold frames
                in contact, kept worlds (target object identified from the shape-code block; spoon grasps are never labelled
                "hold", so hold-only would miss them)
Window (T, H, B): from the human contact labels of the episode's reference (>= 2 MANO links on the object, gaps < 6 frames
filled, runs < 4 dropped), active from 20 frames before each contact onset to 10 frames after (--span hold: to 10
frames after the end of that contact).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfeat import blocks  # noqa: E402

O = "/usr1/e2s/v2d_challenge_data/outputs"
SIDES = ["right", "left"]


def rot_from6(x):
    c1, c2 = x[..., 0:3], x[..., 3:6]
    return np.stack([c1, c2, np.cross(c1, c2)], -1)


def mean_rot(R):
    U, _, Vt = np.linalg.svd(R.mean(0))
    return U @ Vt


def runs(x):
    d = np.diff(np.concatenate([[0], x.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def windows(parquet_dir, B, T, before=20, after=10, span="onset"):
    import pyarrow.parquet as pq
    f = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))[0]
    row = pq.read_table(f, columns=[f"mano_{s}_object_contact_part_ids" for s in SIDES]).to_pylist()[0]
    win = np.zeros((T, len(SIDES), B), np.float32)
    for h, s in enumerate(SIDES):
        ids = np.asarray(row[f"mano_{s}_object_contact_part_ids"])[:T]
        for b in range(B):
            c = (ids == b + 1).sum(1) >= 2
            for a, e in runs(~c):
                if a > 0 and e < len(c) and e - a < 6:
                    c[a:e] = True
            for a, e in runs(c):
                if e - a >= 4:
                    win[max(0, a - before):min(T, (e if span == "hold" else a) + after), h, b] = 1.0
    return win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=int); ap.add_argument("out")
    ap.add_argument("--from_gdiag", action="append", default=[]); ap.add_argument("--from_teacher", default=None)
    ap.add_argument("--parquet", required=True); ap.add_argument("--features", required=True)
    ap.add_argument("--span", default="onset", choices=["onset", "hold"])   # hold: window runs to the end of each human hold
    args = ap.parse_args()
    ex = np.load(args.features); B = ex["object_surface_points"].shape[0]; bps = ex["bps"]
    meta = np.load(glob.glob(f"{O}/gmd/v2/ep18/meta.npz")[0], allow_pickle=True)["meta"].item(); BL = blocks(meta)
    valid = np.zeros((2, B), bool); gp = np.zeros((2, B, 3)); gR = np.tile(np.eye(3), (2, B, 1, 1))
    for spec in args.from_gdiag:
        path, h, b = spec.split(":"); h, b = int(h), int(b)
        d = np.load(path); X = d["X"].astype(np.float64); T = X.shape[0]
        err = np.linalg.norm(d["obj_pos"][:, :, b] - d["ref_obj_pos"][np.maximum(np.arange(T) - 1, 0)][:, None, b], axis=-1)
        m = (d["contact_obj"][:, :, h, b] == 1) & (err < 0.03) & (d["target"][:, :, h] == b)
        x = X[:, :, h][m]
        valid[h, b] = True; gp[h, b] = np.median(x[:, 0:3], 0); gR[h, b] = mean_rot(rot_from6(x[:, 3:9]))
        print(f"[grip_lib] ep{args.episode} {SIDES[h]} hand, object {b}: from {os.path.basename(path)} ({m.sum()} frames) p* {np.round(100 * gp[h, b], 1)} cm")
    if args.from_teacher:
        D = np.load(f"{args.from_teacher}/nominal_data.npz"); X = D["X"].astype(np.float64); ph = D["phase"]; keep = D["keep"]; con = D["contact"]
        b0, b1 = BL["bps"]
        for h in range(2):
            x = X[:, :, h][np.isin(ph[:, :, h], (2, 3)) & (con[:, :, h] > 0) & keep[None]]
            if not len(x):
                continue
            tgt = np.argmin(np.abs(x[:, None, b0:b1] - bps[None]).sum(-1), 1)      # target object from the shape-code block
            for b in range(B):
                xb = x[tgt == b]
                if len(xb) < 100:
                    continue
                Rs = rot_from6(xb[:, 3:9]); Rm = mean_rot(Rs)
                spread = np.degrees(np.arccos(np.clip((np.trace(np.einsum("ij,nkj->nik", Rm, Rs), axis1=1, axis2=2) - 1) / 2, -1, 1)))
                valid[h, b] = True; gp[h, b] = np.median(xb[:, 0:3], 0); gR[h, b] = Rm
                print(f"[grip_lib] ep{args.episode} {SIDES[h]} hand, object {b}: from teacher {args.from_teacher} ({len(xb)} acquire/hold frames in contact) "
                      f"p* {np.round(100 * gp[h, b], 1)} cm, rotation spread median {np.median(spread):.0f} deg")
    import pyarrow.parquet as pq
    f = sorted(glob.glob(os.path.join(args.parquet, "*.parquet")))[0]
    T = len(pq.read_table(f, columns=["object_body_position"]).to_pylist()[0]["object_body_position"])
    win = windows(args.parquet, B, T, span=args.span)
    for h in range(2):
        for b in range(B):
            if valid[h, b]:
                print(f"[grip_lib]   window {SIDES[h]}/{b}: {[(int(a), int(e)) for a, e in runs(win[:, h, b] > 0)]}")
    np.savez(args.out, valid=valid, grip_p=gp, grip_R=gR, window=win)


if __name__ == "__main__":
    main()
