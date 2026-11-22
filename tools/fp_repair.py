"""Repair object-tracker failures that come as jumps (video only; plans/RECONSTRUCTION_PLAN.md P3.2 prelude).

FoundationPose can lose an object for a stretch: one or a few frames where it jumps by much more than the object is
turning, then a consistent but wrong pose until it re-registers with a jump back (clip BlueBox_Take1_001: frames
97-103 jump ~95 deg while the box turns ~13; wrong by 60-80 deg until frame 140). Per-frame checks against the mask
outline do not separate these frames on our clips (the uncalibrated focal length makes good poses fit poorly too), but
the jumps do.

  1. Object rotation in the world (camera path x tracker pose); per frame the relative rotation from the previous
     frame, predicted from the median of the neighbouring steps (+-3 frames, jumps excluded).
  2. A jump is a step that differs from its prediction by more than --jump-deg. The clip is split at jumps.
  3. Each segment s gets a rotation offset O_s in the object's own frame (about its centre): a tracker error stays
     fixed relative to the object while the object turns. Least squares: across every
     jump the corrected track must continue with the predicted step (sigma --cont-deg), and every offset is pulled to
     zero with a weight from how well the segment's projected outline fits the object mask (occlusion-aware: coverage
     of the visible mask and the share of the outline on the mask or a hand). A badly fitting segment between two jumps
     is carried by its neighbours; a well-fitting one keeps its pose. A re-registration jump cancels the failure jump.

    python tools/fp_repair.py --poses fp_box --camera camera_vo_pi3_joint.npz --mesh BlueBox.obj \
        --intrinsics '{"fx":713,"fy":713,"cx":960,"cy":540}' --object-masks box_masks --hand-masks hand_masks --out fp_rep
Writes per-frame camera <- mesh JSONs and OUT/fp_repair.json."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e2s.geometry import load_pose_dir, pose_to_dict, rotation_angle_deg  # noqa: E402
from e2s.masks import mask_dirs, union  # noqa: E402


def outline_score(T_cm, corners, K, obj_dir, hand_dirs, n):
    """Per frame min(coverage of the visible mask, share of the projected hull on mask or hand); NaN if unknown."""
    out = np.full(n, np.nan)
    hd = mask_dirs(hand_dirs)
    for i in range(n):
        if np.isnan(T_cm[i, 0, 0]):
            continue
        m = cv2.imread(str(Path(obj_dir) / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None or not m.any():
            continue
        m = m > 0
        X = (T_cm[i, :3, :3] @ corners.T).T + T_cm[i, :3, 3]
        if (X[:, 2] <= 0.02).any():
            out[i] = 0.0
            continue
        uv = np.c_[K["fx"] * X[:, 0] / X[:, 2] + K["cx"], K["fy"] * X[:, 1] / X[:, 2] + K["cy"]]
        p = np.zeros(m.shape, np.uint8)
        cv2.fillConvexPoly(p, cv2.convexHull(uv.astype(np.int32)), 1)
        p = p > 0
        h = cv2.dilate(union(hd, i, m.shape), np.ones((15, 15), np.uint8)) > 0
        out[i] = min((m & p).sum() / max(m.sum(), 1), (p & (m | h)).sum() / max(p.sum(), 1))
    return out


def repair(R, score, jump_deg=15.0, cont_deg=3.0, prior_deg=1.0, win=3):
    """R (N,3,3) world rotations (all valid). -> (offsets per frame (N,3,3), jump frames, segments, per-segment info)."""
    N = len(R)
    steps = np.einsum("nji,njk->nik", R[:-1], R[1:])                     # local step i-1 -> i, stored at i-1
    rv = Rot.from_matrix(steps).as_rotvec()
    ang = np.degrees(np.linalg.norm(rv, axis=1))
    # first pass: obvious outliers (much larger than the local median step) excluded from the predictions
    med = np.array([np.median(ang[max(0, k - win):k + win + 1]) for k in range(N - 1)])
    rough = ang > med + jump_deg
    pred = np.zeros_like(rv)
    for k in range(N - 1):
        nb = [j for j in range(max(0, k - win), min(N - 1, k + win + 1)) if j != k and not rough[j]]
        pred[k] = np.median(rv[nb], axis=0) if nb else 0.0
    dev = rotation_angle_deg(steps, Rot.from_rotvec(pred).as_matrix())
    jumps = np.nonzero(dev > jump_deg)[0] + 1                             # frame index where the new segment starts
    bounds = [0] + jumps.tolist() + [N]
    segs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    S = len(segs)
    if S == 1:
        return np.repeat(np.eye(3)[None], N, 0), jumps, segs, []
    seg_score = np.array([np.nanmedian(score[a:b]) if np.isfinite(score[a:b]).any() else 0.0 for a, b in segs])
    # prior weight: 1 for a good fit (>= the clip's median score), down to 0.05 for a poor one
    ref = np.nanmedian(score)
    w = np.clip(1.0 - (ref - seg_score) / max(0.25 * ref, 1e-3), 0.05, 1.0)
    w = w * np.sqrt([b - a for a, b in segs]) / np.sqrt(np.median([b - a for a, b in segs]))
    P = Rot.from_rotvec(pred).as_matrix()

    def res(x):
        O = Rot.from_rotvec(x.reshape(S, 3)).as_matrix()
        out = []
        for s in range(1, S):
            a = segs[s][0]
            left = R[a - 1] @ O[s - 1] @ P[a - 1]          # offsets live in the object's frame: R' = R O
            right = R[a] @ O[s]
            out.append(Rot.from_matrix(left.T @ right).as_rotvec() / np.radians(cont_deg))
        out.append((x.reshape(S, 3) * w[:, None] / np.radians(prior_deg)).ravel())
        return np.concatenate([np.ravel(o) for o in out])

    sol = least_squares(res, np.zeros(3 * S), loss="soft_l1", f_scale=3.0)
    O = Rot.from_rotvec(sol.x.reshape(S, 3)).as_matrix()
    off = np.empty((N, 3, 3))
    info = []
    for s, (a, b) in enumerate(segs):
        off[a:b] = O[s]
        info.append({"frames": [int(a), int(b)], "outline_score": float(seg_score[s]), "prior_weight": float(w[s]),
                     "offset_deg": float(np.degrees(np.linalg.norm(sol.x[3 * s:3 * s + 3])))})
    return off, jumps, segs, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--poses", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--object-masks", required=True)
    ap.add_argument("--hand-masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jump-deg", type=float, default=15.0)
    ap.add_argument("--prior-deg", type=float, default=1.0, help="offset prior sigma for a well-fitting segment")
    a = ap.parse_args(argv)
    K = json.loads(Path(a.intrinsics).read_text()) if Path(a.intrinsics).is_file() else json.loads(a.intrinsics)
    c2w = np.load(a.camera)["camera_to_world"].astype(float)
    Tcm = load_pose_dir(a.poses, len(c2w))
    N = len(c2w)
    v = np.array([[float(x) for x in l.split()[1:4]] for l in open(a.mesh) if l.startswith("v ")])
    lo, hi = v.min(0), v.max(0)
    centre = (lo + hi) / 2
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    ok = ~np.isnan(Tcm[:, 0, 0])
    idx = np.nonzero(ok)[0]
    W = c2w[idx] @ Tcm[idx]
    score = outline_score(Tcm, corners, K, a.object_masks, a.hand_masks, N)
    off, jumps, segs, info = repair(W[:, :3, :3], score[idx], jump_deg=a.jump_deg, prior_deg=a.prior_deg)
    cw = np.einsum("nij,j->ni", W[:, :3, :3], centre) + W[:, :3, 3]
    Wn = W.copy()
    Wn[:, :3, :3] = W[:, :3, :3] @ off
    Wn[:, :3, 3] = cw - np.einsum("nij,j->ni", Wn[:, :3, :3], centre)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(idx):
        json.dump(pose_to_dict(np.linalg.inv(c2w[i]) @ Wn[k]), open(out / f"{i:06d}.json", "w"), indent=1)
    rep = {"jumps": [int(idx[j]) for j in jumps], "segments": info, "camera": str(a.camera), "source": str(a.poses),
           "trajectory_source": "video", "outline_score_median": float(np.nanmedian(score))}
    json.dump(rep, open(out / "fp_repair.json", "w"), indent=1)
    big = [s for s in info if s["offset_deg"] > 5]
    print(f"{out}: {len(idx)} frames, jumps at {rep['jumps']}, segments with offset > 5 deg: "
          f"{[(s['frames'], round(s['offset_deg']), round(s['outline_score'], 2)) for s in big]}")


if __name__ == "__main__":
    main()
