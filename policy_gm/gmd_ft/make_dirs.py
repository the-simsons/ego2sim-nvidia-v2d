"""Finger directions for R (run once, CPU): per hand side, a squeeze direction and the top-2 principal components of G's
teachers' finger joint positions during holds.

  make_dirs.py <out.npz>
Data: the labelled teacher rollouts of G's 9 training episodes (outputs/gmd/v2/ep*/nominal_data.npz, kept worlds).
squeeze = median finger pose in hold frames - median in approach frames; pc1, pc2 = principal components of hold-frame
poses after removing the squeeze component. Each direction is scaled so its largest joint entry is 1 (offset = delta * r * d
then moves no joint by more than delta per direction). Also stores the explained variance of the hold-frame poses.
"""
import sys

import numpy as np

ROOT = "/usr1/e2s/v2d_challenge_data/outputs/gmd/v2"
EPS = [41, 40, 26, 18, 2, 35, 8, 3, 36]
FQ = slice(147, 169)      # gfeat.blocks(): pose 15, vec_t 33, nrm_t 33, bps 32, vec_o 33, present 1 -> fq (22)
APPROACH, HOLD = 1, 3

hold = {0: [], 1: []}; appr = {0: [], 1: []}
for ep in EPS:
    d = np.load(f"{ROOT}/ep{ep}/nominal_data.npz")
    X, P, keep = d["X"], d["phase"], d["keep"].astype(bool)
    for h in (0, 1):
        x, p = X[:, keep, h, FQ], P[:, keep, h]
        hold[h].append(x[p == HOLD]); appr[h].append(x[p == APPROACH])
out = {}
for h, side in ((0, "right"), (1, "left")):
    H = np.concatenate(hold[h]); A = np.concatenate(appr[h])
    sq = np.median(H, 0) - np.median(A, 0)
    u = sq / np.linalg.norm(sq)
    C = H - H.mean(0)
    C = C - np.outer(C @ u, u)
    _, s, vt = np.linalg.svd(C, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    tot = ((H - H.mean(0)) ** 2).sum()
    along = lambda v: (((H - H.mean(0)) @ (v / np.linalg.norm(v))) ** 2).sum() / tot  # noqa: E731
    dirs = np.stack([sq, vt[0], vt[1]])
    dirs = dirs / np.abs(dirs).max(1, keepdims=True)
    out[f"dirs_{side}"] = dirs.astype(np.float32)
    out[f"hold_var_explained_{side}"] = np.array([along(sq), along(vt[0]), along(vt[1])], np.float32)
    print(f"{side}: hold frames {len(H)}, approach frames {len(A)}, |squeeze| {np.linalg.norm(sq):.3f} rad, "
          f"hold variance along squeeze/pc1/pc2 {np.round(out[f'hold_var_explained_{side}'], 3)} "
          f"(sum {out[f'hold_var_explained_{side}'].sum():.2f}); residual-pc spectrum {np.round(var[:5], 3)}")
np.savez(sys.argv[1], **out)
