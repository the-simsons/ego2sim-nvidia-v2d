"""E1: cheap fixes on top of an existing method's hands.npz (default E0). Each variant writes results/<name>/.

  a  betas      one shared betas per hand per episode (median over confident observed frames); global pose re-fit
                with betas fixed so the mesh keeps its image placement (rotation + translation, scale fixed)
  b  align      depth alignment against SGBM stereo (refs/<ep>/stereo.npz, uses cam_b+cam_c => consumed_stereo):
                  dz     upstream rule with stereo: shift along the centroid ray by median(stereo - rendered) over
                         mask & rendered pixels (mesh size unchanged)
                  scale  scale about the camera centre by median(stereo / rendered): image placement exactly kept
                  huber  robust (Huber, IRLS) point-to-surface fit of the visible MANO vertices to the stereo points
                         inside mask & silhouette, free 3D translation + scale about the camera centre, with a soft
                         term keeping the projected hand centroid within ~2 px
  c  smooth     temporal smoothing of joints/verts per side over runs of observed frames:
                  oneeuro (min_cutoff 1.0 Hz, beta 0.3), savgol (window 7, order 2),
                  traj   small trajectory optimisation: data + acceleration prior + cam_a 2D reprojection term to the
                         cam_a detector keypoints (makes cam_a kp error non-independent; flagged)
  d  swaps      handedness-swap repair (labels crossed vs the previous frame by > 5 cm => swap the rows)

Usage: python -m handbench.experiments.e1 {betas|align|smooth|swaps} --src E0 --name E1b-huber [--mode huber]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from handbench import format as hf
from handbench import paths
from handbench.calib import load_rig
from handbench.datasets.v2d import StereoCrops
from handbench.fit import refit_global
from handbench.mano import Mano
from handbench.render import depth_map

SIDES = ("L", "R")


def commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT, text=True).strip()
    except Exception:
        return "unknown"


def regen(mano, a, r):
    V, J = mano.forward(a["global_orient"][r], a["hand_pose"][r], a["betas"][r], a["transl"][r], a["side"][r] == "R",
                        a["scale"][r])
    return V[0], J[0]


def set_row(a, r, V, J, mano, refit=True):
    """Store new geometry; re-express it as MANO params (global orient/transl/scale) when refit."""
    a["joints_cam"][r] = J
    if "verts_cam" in a:
        a["verts_cam"][r] = V
    if refit:
        go, tr, s = refit_global(mano, a["hand_pose"][r], a["betas"][r], a["side"][r] == "R", V)
        a["global_orient"][r], a["transl"][r], a["scale"][r] = go, tr, s


def similarity_about_camera(V, s, t=np.zeros(3)):
    return V * s + t


# ---------------------------------------------------------------- a: shared betas
def do_betas(a, ep, mano, cams, args):
    for side in SIDES:
        rows = np.nonzero((a["side"] == side) & a["valid"])[0]
        conf = rows[(a["source"][rows] == "observed") & (a["conf"][rows] >= 1.0)]
        if len(conf) < 3:
            continue
        b = np.median(a["betas"][conf], 0)
        s_med = float(np.median(a["scale"][conf]))
        for r in rows:
            V_old, _ = regen(mano, a, r)
            a["betas"][r] = b
            if args.mode == "size":  # also one hand size per side: keep the centroid, let a later align fix depth
                a["scale"][r] = s_med
            go, tr, s = refit_global(mano, a["hand_pose"][r], b, side == "R", V_old, with_scale=False)
            a["global_orient"][r], a["transl"][r] = go, tr
            if args.mode == "size":
                V_new, _ = regen(mano, a, r)
                a["transl"][r] += V_old.mean(0) - V_new.mean(0)
            V, J = regen(mano, a, r)
            a["joints_cam"][r] = J
            if "verts_cam" in a:
                a["verts_cam"][r] = V
    return {"betas": "median of conf>=1 observed frames per side; global refit", "mode": args.mode or "betas"}


# ---------------------------------------------------------------- b: stereo alignment
def _render_crop(V, faces, A, sc, box, shape):
    K = A.K.copy()
    K[:2] *= sc
    K[:2, 2] = (A.K[:2, 2] + 0.5) * sc - 0.5 - box[:2]
    uv = V[:, :2] / V[:, 2:3] * np.diag(K)[:2] + K[:2, 2]
    return depth_map(uv, V[:, 2], faces, (shape[1], shape[0])), K, uv


def _huber_fit(V, faces, A, D, M, sc, box, delta=0.01, lam_px=2.0, iters=6):
    """Fit scale s (about the camera centre) and translation t so the visible mesh surface matches the stereo points."""
    from scipy.spatial import cKDTree

    R, K, uv = _render_crop(V, faces, A, sc, box, D.shape)
    g = M & (R > 0) & (D > 0)
    if g.sum() < 30:
        return None
    s0 = float(np.median(D[g] / R[g]))
    ys, xs = np.nonzero(g)
    rays = np.c_[(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones(len(xs))]
    P = rays * D[g][:, None]  # stereo points (benchmark frame)
    # visible vertices: projected inside the crop and not hidden behind the rendered surface
    ui = np.round(uv).astype(int)
    ins = (ui[:, 0] >= 0) & (ui[:, 0] < D.shape[1]) & (ui[:, 1] >= 0) & (ui[:, 1] < D.shape[0])
    vis = np.zeros(len(V), bool)
    vis[ins] = V[ins, 2] <= R[ui[ins, 1], ui[ins, 0]] + 0.004
    if vis.sum() < 20:
        return None
    Q0 = V[vis]
    c0 = V.mean(0)
    pc0 = c0[:2] / c0[2]
    s, t = s0, np.zeros(3)
    f = A.K[0, 0]
    for _ in range(iters):
        Q = Q0 * s + t
        tree = cKDTree(Q)
        d, j = tree.query(P)
        r = P - Q[j]
        w = np.where(d < delta, 1.0, delta / np.maximum(d, 1e-9))
        # linearised least squares in (ds, dt): P ~ (s+ds) Q0 + t + dt  =>  r ~ ds Q0[j] + dt
        Jm = np.c_[Q0[j].reshape(-1, 3, 1), np.broadcast_to(np.eye(3), (len(j), 3, 3))].reshape(-1, 4)
        rv = r.reshape(-1)
        W = np.repeat(w, 3)
        H = Jm.T @ (Jm * W[:, None])
        b = Jm.T @ (W * rv)
        # soft 2D anchor: projected centroid (s c0 + t) stays at pc0  (weight lam_px px)
        c = c0 * s + t
        z = c[2]
        Jp = np.array([[c0[0] / z - c[0] * c0[2] / z ** 2, 1 / z, 0, -c[0] / z ** 2],
                       [c0[1] / z - c[1] * c0[2] / z ** 2, 0, 1 / z, -c[1] / z ** 2]]) * f
        rp = (pc0 - c[:2] / z) * f
        wp = len(P) * 0.01 / lam_px ** 2 * delta ** 2
        H += wp * Jp.T @ Jp
        b += wp * Jp.T @ rp
        dx = np.linalg.solve(H + 1e-9 * np.eye(4), b)
        s, t = s + dx[0], t + dx[1:]
        if np.abs(dx).max() < 1e-5:
            break
    return s, t


def do_align(a, ep, mano, cams, args):
    st = StereoCrops(paths.REFS / paths.ep_name(ep) / ("stereo.npz" if args.stereo == "sgbm" else f"stereo_{args.stereo}.npz"))
    A = cams["ego_cam_a"]
    stats = {"aligned": 0, "kept": 0}
    for r in np.nonzero(a["valid"])[0]:
        i, side = int(a["frame_idx"][r]), a["side"][r]
        V, J = regen(mano, a, r)
        got = st.get(i, side)
        new = None
        if got is not None:
            D, M, box = got
            faces = mano.faces_for(side == "R")
            if args.mode in ("dz", "scale"):
                R, _, _ = _render_crop(V, faces, A, st.scale, box, D.shape)
                g = M & (R > 0) & (D > 0)
                if g.sum() >= 30:
                    if args.mode == "dz":
                        dz = float(np.median(D[g] - R[g]))
                        c = V.mean(0)
                        off = c / c[2] * dz
                        new = (V + off, J + off)
                    else:
                        s = float(np.median(D[g] / R[g]))
                        new = (V * s, J * s)
            else:
                fit = _huber_fit(V, faces, A, D, M, st.scale, box)
                if fit is not None:
                    s, t = fit
                    new = (V * s + t, J * s + t)
        if new is None:
            # no stereo for this frame (hand outside the stereo overlap / too few pixels): keep, mark low confidence
            a["conf"][r] = min(a["conf"][r], 0.3)
            stats["kept"] += 1
            continue
        set_row(a, r, new[0], new[1], mano)
        stats["aligned"] += 1
    return {"align": args.mode, **stats}


# ---------------------------------------------------------------- c: smoothing
def _runs(mask):
    idx = np.nonzero(mask)[0]
    if not len(idx):
        return []
    br = np.nonzero(np.diff(idx) > 1)[0]
    return np.split(idx, br + 1)


def one_euro(x, fps, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
    alpha = lambda c: 1.0 / (1.0 + fps / (2 * np.pi * c))
    y = x.copy()
    dx = np.zeros_like(x[0])
    for k in range(1, len(x)):
        d = (x[k] - y[k - 1]) * fps
        ad = alpha(d_cutoff)
        dx = ad * d + (1 - ad) * dx
        c = min_cutoff + beta * np.linalg.norm(dx.reshape(-1, 3), axis=1).mean()
        a_ = alpha(c)
        y[k] = a_ * x[k] + (1 - a_) * y[k - 1]
    return y


def do_smooth(a, ep, mano, cams, args):
    from scipy.signal import savgol_filter

    fps = paths.fps()
    A = cams["ego_cam_a"]
    kp = None
    if args.mode == "traj":
        d = np.load(paths.REFS / paths.ep_name(ep) / "kp2d.npz")
        kp, ks = d["kp"][:, :, 0], d["score"][:, :, 0]
    for s_i, side in enumerate(SIDES):
        rows = np.nonzero(a["side"] == side)[0]
        obs = a["valid"][rows] & (a["source"][rows] == "observed")
        for run in _runs(obs):
            if len(run) < 3:
                continue
            rr = rows[run]
            VJ = np.stack([np.concatenate(regen(mano, a, r)) for r in rr])  # (T, 799, 3)
            if args.mode == "oneeuro":
                out = one_euro(VJ, fps)
            elif args.mode == "savgol":
                w = min(7, len(run) - (1 - len(run) % 2))
                if w < 3:
                    continue
                out = savgol_filter(VJ, w, 2, axis=0, mode="interp")
            else:
                out = _traj(VJ, rr, a, kp, ks, s_i, A, fps)
            for k, r in enumerate(rr):
                set_row(a, r, out[k, :778], out[k, 778:], mano)
    return {"smooth": args.mode}


def _traj(VJ, rr, a, kp, ks, s_i, A, fps, lam_acc=2e-4, lam_2d=1e-6):
    """Per-frame rigid offsets o_t (3) minimising  sum ||o_t||^2 (stay near data) + lam_acc sum ||accel of joints||^2
    + lam_2d sum_joints score * ||proj(J_t + o_t) - kp_t||^2  (linearised; solved jointly as a banded system)."""
    T = len(rr)
    Jc = VJ[:, 778:].mean(1)  # joint centroids drive the smoothness term
    n = 3 * T
    H = np.eye(n)
    b = np.zeros(n)
    # acceleration of (centroid + offset)
    for t in range(1, T - 1):
        acc = Jc[t + 1] - 2 * Jc[t] + Jc[t - 1]
        idx = [(t - 1, 1.0), (t, -2.0), (t + 1, 1.0)]
        for (p, cp) in idx:
            for (q, cq) in idx:
                H[3 * p:3 * p + 3, 3 * q:3 * q + 3] += lam_acc * fps ** 4 / 1e2 * cp * cq * np.eye(3)
            b[3 * p:3 * p + 3] -= lam_acc * fps ** 4 / 1e2 * cp * acc
    # 2D term on cam_a keypoints (pinhole benchmark camera; keypoints undistorted)
    f = A.K[0, 0]
    for k, r in enumerate(rr):
        i = int(a["frame_idx"][r])
        m = ks[i, s_i] >= 0.5
        if m.sum() < 5:
            continue
        n_ = A.undistort_px(np.nan_to_num(kp[i, s_i][m]))
        J = VJ[k, 778:][m]
        for jj in range(len(J)):
            X, Y, Z = J[jj]
            Jp = f * np.array([[1 / Z, 0, -X / Z ** 2], [0, 1 / Z, -Y / Z ** 2]])
            rp = f * (n_[jj] - J[jj, :2] / Z)
            e = np.linalg.norm(rp)
            if e > 60:  # wrong keypoint (other hand / swap): ignore
                continue
            w = 1.0 if e < 10 else 10 / e  # Huber weight, 10 px
            H[3 * k:3 * k + 3, 3 * k:3 * k + 3] += w * lam_2d * Jp.T @ Jp * 1e4
            b[3 * k:3 * k + 3] += w * lam_2d * Jp.T @ rp * 1e4
    o = np.linalg.solve(H, b).reshape(T, 3)
    nrm = np.linalg.norm(o, axis=1, keepdims=True)
    o *= np.minimum(1.0, 0.05 / np.maximum(nrm, 1e-9))  # at most 5 cm correction per frame
    return VJ + o[:, None]


# ---------------------------------------------------------------- d: handedness swaps
def do_swaps(a, ep, mano, cams, args):
    n = len(a["frame_idx"]) // 2
    ok = (a["valid"] & (a["source"] == "observed")).reshape(n, 2)
    w = a["joints_cam"].reshape(n, 2, 21, 3)[:, :, 0].astype(float)
    swapped = []
    last = {0: None, 1: None}
    for i in range(n):
        if ok[i].all() and last[0] is not None and last[1] is not None:
            d = np.linalg.norm(w[i, 0] - last[0]) + np.linalg.norm(w[i, 1] - last[1])
            c = np.linalg.norm(w[i, 0] - last[1]) + np.linalg.norm(w[i, 1] - last[0])
            if c < d - 0.05:
                swapped.append(i)
                for k, v in a.items():
                    if k in ("frame_idx", "t", "side"):
                        continue
                    v[2 * i], v[2 * i + 1] = v[2 * i + 1].copy(), v[2 * i].copy()
                # geometry was produced for the other hand model: mirror is not applied, MANO params stay as is
                w[i] = w[i, ::-1]
        for s in range(2):
            if ok[i, s]:
                last[s] = w[i, s]
    return {"swaps_repaired": swapped}


OPS = {"betas": do_betas, "align": do_align, "smooth": do_smooth, "swaps": do_swaps}


def _run(args_ep):
    args, ep = args_ep
    t0 = time.time()
    mano, cams = Mano(), load_rig(paths.rig_json())
    src = hf.load(paths.RESULTS / args.src / ep / "hands.npz")
    a = {k: v.copy() for k, v in src.arrays.items()}
    if "verts_cam" in a:
        a["verts_cam"] = a["verts_cam"].astype(np.float32)
    info = OPS[args.op](a, ep, mano, cams, args)
    if "verts_cam" in a:
        a["verts_cam"] = a["verts_cam"].astype(np.float16)
    meta = dict(src.meta)
    meta.update(method=args.name, version=f"{src.meta.get('version')} + E1 {args.op} {args.mode or ''}".strip(),
                args={"op": args.op, "mode": args.mode, "src": args.src, "stereo": args.stereo}, adapter_commit=commit(),
                derived_wall_time_s=time.time() - t0, info=info,
                consumed_stereo_depth=bool(src.meta.get("consumed_stereo_depth")) or args.op == "align",
                uses_cam_a_keypoints=bool(src.meta.get("uses_cam_a_keypoints")) or args.mode == "traj")
    hf.save(paths.RESULTS / args.name / ep / "hands.npz", a, meta)
    return ep, info, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=list(OPS))
    ap.add_argument("--src", default="E0")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="")
    ap.add_argument("--stereo", default="sgbm", choices=["sgbm", "fs"], help="depth source for align (fs = FoundationStereo)")
    ap.add_argument("--episodes", default="")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    eps = [paths.ep_name(e) for e in a.episodes.split(",")] if a.episodes else paths.episodes()
    with ProcessPoolExecutor(a.jobs) as ex:
        for ep, info, dt in ex.map(_run, [(a, e) for e in eps]):
            short = {k: (v if not isinstance(v, list) else len(v)) for k, v in info.items()}
            print(f"{ep}: {short} ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
