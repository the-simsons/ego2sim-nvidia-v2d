"""Refinement Passes 1-2 on a FoundationPose object track (camera frame), from FoundationStereo depth + SAM2 masks.

  python track_refine.py EP_DIR NAME MASK_ID MESH IN_POSE_DIR OUT_PREFIX [--depth stereo_fs] [--lam 1.0]
  -> OUT_PREFIX_r1/ (Pass 1)  OUT_PREFIX_r2/ (Passes 1+2), FoundationPose-style <frame>.json, + OUT_PREFIX_refine.json

Pass 1, per frame: starting from the input pose, robust least squares over the 6-DoF pose (rotation about the object
  centre) of
  - depth: point-to-plane distance from the stereo points inside the eroded, unoccluded mask to the mesh surface;
  - silhouette: distance (px -> m at the point's depth) from each projected mesh surface sample to the observed mask,
    zero where hands or other objects cover the image (they may hide the object);
  - a weak prior toward the input pose.
  Frames with fewer than --min-points depth points keep the input pose.
Pass 2: per frame, hypotheses = the Pass 1 pose and its 180-degree flips about the mesh's three principal axes (through
  the centroid), each refined like Pass 1 and scored by its mean robust cost. A Viterbi path picks one hypothesis per
  frame: unary log(cost / best cost in the frame), pairwise --lam * rotation change (rad) + translation change / radius.
  Needs no mocap, so it runs the same on eval episodes.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def load_pose(p):
    j = json.load(open(p))
    w, x, y, z = j["rotation"]
    T = np.eye(4)
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = j["translation"]
    return T


def save_pose(p, T):
    x, y, z, w = R.from_matrix(T[:3, :3]).as_quat()
    json.dump({"rotation": [float(w), float(x), float(y), float(z)], "translation": [float(v) for v in T[:3, 3]],
               "scale": [1.0, 1.0, 1.0]}, open(p, "w"), indent=4)


def read_depth(d, i):
    raw = cv2.imread(str(Path(d) / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    raw = raw.astype(np.float32)
    z = 65535.0 / np.maximum(raw, 1.0) - 1.0
    z[(raw >= 65535) | (raw <= 0)] = 0
    return z


def read_mask(p, shape):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape, bool)
    return cv2.resize(m, shape[::-1], interpolation=cv2.INTER_NEAREST) > 127


class Model:
    def __init__(self, mesh_path, n_surf=20000, n_sil=2000):
        m = trimesh.load(mesh_path, force="mesh")
        pts, fi = trimesh.sample.sample_surface(m, n_surf, seed=0)
        self.pts, self.nrm = np.asarray(pts), np.asarray(m.face_normals[fi])
        self.tree = cKDTree(self.pts)
        self.sil = self.pts[np.random.default_rng(0).choice(n_surf, n_sil, replace=False)]
        self.c = self.pts.mean(0)
        self.radius = float(np.linalg.norm(self.pts - self.c, axis=1).max())
        _, _, vt = np.linalg.svd(self.pts - self.c, full_matrices=False)
        self.flips = [np.eye(4)]
        for a in vt:
            F = np.eye(4)
            F[:3, :3] = R.from_rotvec(np.pi * a).as_matrix()
            F[:3, 3] = self.c - F[:3, :3] @ self.c
            self.flips.append(F)


class Frame:
    def __init__(self, z, mask, dontcare, K, max_pts, min_pts, rng):
        self.K, self.shape = K, z.shape
        self.dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        self.dontcare = cv2.dilate(dontcare.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        m = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        med = cv2.medianBlur(z, 5)
        m &= (z > 0) & ~self.dontcare & (np.abs(z - med) < 0.015)
        v, u = np.nonzero(m)
        if len(u) > max_pts:
            k = rng.choice(len(u), max_pts, replace=False)
            u, v = u[k], v[k]
        zz = z[v, u]
        self.obs = np.stack([(u - K[0, 2]) * zz / K[0, 0], (v - K[1, 2]) * zz / K[1, 1], zz], 1)
        self.ok = len(u) >= min_pts


def compose(T0, x, c_obj):
    """Delta about the object centre: X_cam = dR (T0 X - c_cam) + c_cam + dt."""
    dR = R.from_rotvec(x[:3]).as_matrix()
    c_cam = T0[:3, :3] @ c_obj + T0[:3, 3]
    T = np.eye(4)
    T[:3, :3] = dR @ T0[:3, :3]
    T[:3, 3] = dR @ (T0[:3, 3] - c_cam) + c_cam + x[3:]
    return T


def residuals(x, T0, fr, md, w_sil, w_prior, cap_px=40.0):
    T = compose(T0, x, md.c)
    Rm, t = T[:3, :3], T[:3, 3]
    q = (fr.obs - t) @ Rm  # observed points in the object frame
    _, idx = md.tree.query(q)
    r_depth = np.einsum("ij,ij->i", md.nrm[idx], q - md.pts[idx])
    P = md.sil @ Rm.T + t
    z = np.maximum(P[:, 2], 0.05)
    u = np.round(fr.K[0, 0] * P[:, 0] / z + fr.K[0, 2]).astype(int)
    v = np.round(fr.K[1, 1] * P[:, 1] / z + fr.K[1, 2]).astype(int)
    inside = (u >= 0) & (u < fr.shape[1]) & (v >= 0) & (v < fr.shape[0])
    d = np.full(len(P), cap_px, np.float64)
    d[inside] = np.minimum(fr.dt[v[inside], u[inside]], cap_px)
    d[inside & fr.dontcare[np.clip(v, 0, fr.shape[0] - 1), np.clip(u, 0, fr.shape[1] - 1)]] = 0.0
    r_sil = w_sil * d * z / fr.K[0, 0]
    r_prior = w_prior * np.concatenate([x[:3] * md.radius, x[3:]])
    return np.concatenate([r_depth, r_sil, r_prior])


def refine(T0, fr, md, a):
    res = least_squares(residuals, np.zeros(6), args=(T0, fr, md, a.w_sil, a.w_prior), loss="huber",
                        f_scale=a.f_scale, max_nfev=a.max_nfev, x_scale=[0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
    return compose(T0, res.x, md.c), float(res.cost) / len(res.fun)


def angle(Ra, Rb):
    return float(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ep_dir"); ap.add_argument("name"); ap.add_argument("mask_id"); ap.add_argument("mesh")
    ap.add_argument("in_pose_dir"); ap.add_argument("out_prefix")
    ap.add_argument("--depth", default="stereo_fs")
    ap.add_argument("--lam", type=float, default=1.0, help="Pass 2 pairwise weight on rotation change (per rad)")
    ap.add_argument("--w-sil", type=float, default=0.5)
    ap.add_argument("--w-prior", type=float, default=0.1)
    ap.add_argument("--f-scale", type=float, default=0.005, help="Huber scale (m)")
    ap.add_argument("--max-nfev", type=int, default=30)
    ap.add_argument("--max-points", type=int, default=1500)
    ap.add_argument("--min-points", type=int, default=150)
    a = ap.parse_args()
    E = Path(a.ep_dir)
    Kj = json.load(open(E / "stereo" / "intrinsics_undist.json"))
    D = E / a.depth / "stereo_depth"
    md = Model(a.mesh)
    poses_in = sorted(Path(a.in_pose_dir).glob("[0-9]*.json"))
    hand_dirs = sorted((E / "hand_masks").iterdir())
    other = [p for p in sorted((E / "masks").iterdir()) if p.name != a.mask_id]
    rng = np.random.default_rng(0)
    frames, T_in, hyp, cost, refined = [], [], [], [], []
    K = None
    for p in poses_in:
        i = int(p.stem)
        T0 = load_pose(p)
        z = read_depth(D, i)
        if K is None:
            sx, sy = z.shape[1] / Kj["width"], z.shape[0] / Kj["height"]
            K = np.array([[Kj["fx"] * sx, 0, Kj["cx"] * sx], [0, Kj["fy"] * sy, Kj["cy"] * sy], [0, 0, 1]])
        mask = read_mask(E / "masks" / a.mask_id / f"{i:06d}.png", z.shape)
        dc = np.zeros(z.shape, bool)
        for d in hand_dirs + other:
            dc |= read_mask(d / f"{i:06d}.png", z.shape)
        fr = Frame(z, mask, dc, K, a.max_points, a.min_points, rng)
        frames.append(i); T_in.append(T0)
        if not fr.ok:
            hyp.append([T0 @ F for F in md.flips]); cost.append(np.ones(len(md.flips))); refined.append(False)
            continue
        hs, cs = [], []
        for F in md.flips:
            T, c = refine(T0 @ F, fr, md, a)
            hs.append(T); cs.append(c)
        hyp.append(hs); cost.append(np.array(cs)); refined.append(True)
    n, k = len(frames), len(md.flips)
    # Pass 2: Viterbi over hypotheses
    U = np.array([np.log((c + 1e-12) / (c.min() + 1e-12)) for c in cost])
    acc, back = U[0].copy(), np.zeros((n, k), int)
    for t in range(1, n):
        pw = np.array([[a.lam * (angle(hyp[t - 1][p][:3, :3], hyp[t][q][:3, :3]) +
                                 np.linalg.norm(hyp[t - 1][p][:3, 3] - hyp[t][q][:3, 3]) / md.radius)
                        for p in range(k)] for q in range(k)])  # [q, p]
        tot = acc[None, :] + pw
        back[t] = tot.argmin(1)
        acc = tot.min(1) + U[t]
    path = [int(acc.argmin())]
    for t in range(n - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    path = path[::-1]
    o1, o2 = Path(f"{a.out_prefix}_r1"), Path(f"{a.out_prefix}_r2")
    o1.mkdir(parents=True, exist_ok=True); o2.mkdir(parents=True, exist_ok=True)
    for t, i in enumerate(frames):
        save_pose(o1 / f"{i:06d}.json", hyp[t][0])
        save_pose(o2 / f"{i:06d}.json", hyp[t][path[t]])
    moved = [np.linalg.norm(hyp[t][0][:3, 3] - T_in[t][:3, 3]) * 100 for t in range(n) if refined[t]]
    summary = {"frames": n, "refined": int(sum(refined)), "flipped_frames": int(sum(p != 0 for p in path)),
               "hypothesis_counts": np.bincount(path, minlength=k).tolist(),
               "pass1_shift_cm_median": float(np.median(moved)) if moved else None,
               "pass1_cost_median": float(np.median([c[0] for c, r in zip(cost, refined) if r])) if any(refined) else None,
               "args": vars(a)}
    json.dump(summary, open(f"{a.out_prefix}_refine.json", "w"), indent=1)
    print(f"{a.name}: {summary['refined']}/{n} frames refined, Pass 1 shift median {summary['pass1_shift_cm_median']:.2f} cm, "
          f"Pass 2 hypotheses per frame {summary['hypothesis_counts']} ({summary['flipped_frames']} flipped)")


if __name__ == "__main__":
    main()
