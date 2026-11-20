"""Mocap-assisted camera path (ported from the 2026-11-08 AIMS scripts cam_from_fp.py + refine_camera.py).

    world <- camera = (world <- mesh, mocap) @ inv(camera <- mesh, object tracker)

Stage 1 (from_tracker): the bare cuboid looks the same under 4 turns, so a Viterbi pass picks one per frame (continuous
camera, above the table, looking down); the sync is refined to the lag giving the smoothest camera path; the path is
median + Savitzky-Golay smoothed. Writes camera_vo_fp.npz.
Stage 2 (refine): each camera pose is refined against the object mask outline (both directions, Cauchy loss, a prior on
the stage-1 pose), the sync re-estimated by outline cost, then smoothed. Writes camera_vo.npz and fp_mocap/*.json
(the mocap object expressed in the refined camera).

    python -m e2s.mocap.assist.camera --clip BlueBox_Take1_000 [--root DATA] [--out DIR] [--check]
--check writes to a temporary directory and compares with the outputs already in the clip's outputs_dir (P0.1
regression: within 1 mm). The initial lag comes from <clip dir>/sync.json (e2s.mocap.assist.sync)."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.transform import Rotation as Rot

from bench import manifest as mf
from e2s.geometry import box_symmetries, load_pose_dir, mat
from e2s.mocap.assist.common import Clip, write_pose_dir

SYM = box_symmetries()
# stage 1
MIN_CAMERA_HEIGHT = 1.0       # m in the mocap world; the table top is at about 0.77 m
LAG_SEARCH_1 = (-0.25, 0.2501, 0.002)
# stage 2
SC, NF, NB, FS = 0.5, 80, 80, 3.0
LAG_SEARCH_2 = (-0.16, 0.1601, 0.01)


def _unwrap(q):
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def from_tracker(c, tracker_dir, lag0):
    """Stage 1. -> (camera_to_world (N,4,4), lag, flips (N,))."""
    Tcb = load_pose_dir(tracker_dir)
    N = len(Tcb)
    pts = c.pts[:N]
    inv0 = np.linalg.inv(Tcb)
    hi = c.up

    def pick_flips(l):
        Twb = c.world_mesh(pts + l)
        cand = np.stack([Twb @ S @ inv0 for S in SYM], axis=1)
        p = cand[:, :, :3, 3]
        unary = 8.0 * np.maximum(MIN_CAMERA_HEIGHT - p[:, :, hi], 0) + 2.0 * np.maximum(cand[:, :, hi, 2], 0)
        cost = unary[0].copy()
        back = np.zeros((N, 4), int)
        for i in range(1, N):
            dp = np.linalg.norm(p[i][:, None] - p[i - 1][None], axis=2)
            tr = np.einsum("aij,bij->ab", cand[i, :, :3, :3], cand[i - 1, :, :3, :3])
            da = np.arccos(np.clip((tr - 1) / 2, -1, 1))
            t = cost[None] + 10 * dp + 1.5 * da
            back[i] = np.argmin(t, axis=1)
            cost = t.min(axis=1) + unary[i]
        s = np.zeros(N, int)
        s[-1] = int(np.argmin(cost))
        for i in range(N - 1, 0, -1):
            s[i - 1] = back[i, s[i]]
        return s

    flips = pick_flips(lag0)
    inv = np.stack([SYM[flips[i]] @ inv0[i] for i in range(N)])
    Twb0 = c.world_mesh(pts + lag0)
    speed = np.linalg.norm(np.gradient(Twb0[:, :3, 3], axis=0), axis=1)
    moving = speed > max(np.percentile(speed, 50), 0.003)

    def rough(l):
        p = (c.world_mesh(pts + l) @ inv)[:, :3, 3]
        a = np.linalg.norm(np.diff(p, 2, axis=0), axis=1)
        return float(np.median(a[moving[1:-1]])) if moving[1:-1].any() else float(np.median(a))

    lags = lag0 + np.arange(*LAG_SEARCH_1)
    lag = float(lags[int(np.argmin([rough(l) for l in lags]))])
    flips = pick_flips(lag)
    inv = np.stack([SYM[flips[i]] @ inv0[i] for i in range(N)])
    Twc = c.world_mesh(pts + lag) @ inv
    c_s = savgol_filter(np.c_[[medfilt(Twc[:, a, 3], 7) for a in range(3)]].T, 11, 2, axis=0)
    q = _unwrap(Rot.from_matrix(Twc[:, :3, :3]).as_quat())
    q_s = savgol_filter(np.c_[[medfilt(q[:, a], 7) for a in range(4)]].T, 11, 2, axis=0)
    q_s /= np.linalg.norm(q_s, axis=1, keepdims=True)
    return mat(Rot.from_quat(q_s).as_matrix(), c_s), lag, flips


def refine(c, c2w, lag):
    """Stage 2. -> (camera_to_world, lag, overlap before, overlap after)."""
    N = len(c2w)
    pts = c.pts[:N]
    dts, conts, masks = [], [], []
    for i in range(N):
        m = cv2.resize(c.mask(i), None, fx=SC, fy=SC, interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ct = max(cs, key=len)[:, 0, :].astype(float)
        k = (ct[:, 0] > 2) & (ct[:, 0] < m.shape[1] - 3) & (ct[:, 1] > 2) & (ct[:, 1] < m.shape[0] - 3)
        ct = ct[k] if k.sum() > 20 else ct
        e = np.ones_like(m)
        e[ct[:, 1].astype(int), ct[:, 0].astype(int)] = 0
        dts.append(cv2.distanceTransform(e, cv2.DIST_L2, 5))
        conts.append(ct[np.linspace(0, len(ct) - 1, NB).astype(int)])
        masks.append(m)

    def cam_mesh(x, Twb):
        T = np.eye(4)
        T[:3, :3] = Rot.from_rotvec(x[:3]).as_matrix()
        T[:3, 3] = x[3:]
        return T @ Twb

    def res(x, i, Twb):
        uv, _ = c.project(cam_mesh(x, Twb), SC)
        hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(float)
        a, b = hull, np.roll(hull, -1, axis=0)
        seg = np.linalg.norm(b - a, axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        s = np.linspace(0, cum[-1], NF, endpoint=False)
        k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(a) - 1)
        pf = a[k] + (b[k] - a[k]) * ((s - cum[k]) / np.maximum(seg[k], 1e-6))[:, None]
        fw = map_coordinates(dts[i], [pf[:, 1], pf[:, 0]], order=1, mode="nearest")
        ins = (0 <= pf[:, 0]) & (pf[:, 0] < dts[i].shape[1]) & (0 <= pf[:, 1]) & (pf[:, 1] < dts[i].shape[0])
        q, ab = conts[i], b - a
        t = np.clip(((q[:, None] - a[None]) * ab[None]).sum(2) / np.maximum((ab * ab).sum(1)[None], 1e-6), 0, 1)
        bw = np.linalg.norm(q[:, None] - (a[None] + t[..., None] * ab[None]), axis=2).min(1)
        return np.r_[np.where(ins, fw, 0.0), bw]

    def to_x(T):
        Rc = T[:3, :3].T
        return np.r_[Rot.from_matrix(Rc).as_rotvec(), -Rc @ T[:3, 3]]

    def from_x(xs):
        T = np.tile(np.eye(4), (len(xs), 1, 1))
        for i, x in enumerate(xs):
            Rc = Rot.from_rotvec(x[:3]).as_matrix()
            T[i, :3, :3], T[i, :3, 3] = Rc.T, -Rc.T @ x[3:]
        return T

    def smooth(T, win=9):
        ctr = np.c_[[medfilt(T[:, a, 3], 5) for a in range(3)]].T
        q = _unwrap(Rot.from_matrix(T[:, :3, :3]).as_quat())
        qs = savgol_filter(np.c_[[medfilt(q[:, a], 5) for a in range(4)]].T, win, 2, axis=0)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        return mat(Rot.from_quat(qs).as_matrix(), savgol_filter(ctr, win, 2, axis=0))

    def solve(i, x0, Twb, lam=1.5):
        f = lambda x: np.r_[res(x, i, Twb), lam * (x - x0) / 0.01]
        return least_squares(f, x0, loss="cauchy", f_scale=FS, x_scale=0.02, max_nfev=20).x

    def overlap(T, Twb):
        return np.array([c.iou(np.linalg.inv(T[i]) @ Twb[i], masks[i], SC) for i in range(N)])

    Twb = c.world_mesh(pts + lag)
    before = overlap(c2w, Twb)
    cur = c2w.copy()
    for _ in range(3):
        cur = smooth(from_x(np.array([solve(i, to_x(cur[i]), Twb[i]) for i in range(N)])))
    sp = np.linalg.norm(np.gradient(Twb[:, :3, 3], axis=0), axis=1)
    mv = np.nonzero(sp > np.percentile(sp, 60))[0][::2]
    lags = lag + np.arange(*LAG_SEARCH_2)
    costs = []
    for l in lags:
        T2 = c.world_mesh(pts + l)
        costs.append(sum(float(np.sum(np.log1p((res(solve(i, to_x(cur[i]), T2[i], lam=0.5), i, T2[i]) / FS) ** 2)))
                         for i in mv))
    lag2 = float(lags[int(np.argmin(costs))])
    Twb = c.world_mesh(pts + lag2)
    for _ in range(2):
        cur = smooth(from_x(np.array([solve(i, to_x(cur[i]), Twb[i]) for i in range(N)])))
    return cur, lag2, before, overlap(cur, Twb)


def run(c, out, tracker="fp_box"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lag0 = float(json.load(open(c.clip_dir / "sync.json"))["lag_s"])
    c2w1, lag1, flips = from_tracker(c, c.out / tracker, lag0)
    N = len(c2w1)
    np.savez(out / "camera_vo_fp.npz", camera_to_world=c2w1, lag=lag1, pts=c.pts[:N])
    c2w, lag2, before, after = refine(c, c2w1, lag1)
    np.savez(out / "camera_vo.npz", camera_to_world=c2w, lag=lag2, pts=c.pts[:N], iou=after)
    Tcb = np.linalg.inv(c2w) @ c.world_mesh(c.pts[:N] + lag2)
    write_pose_dir(out / "fp_mocap", Tcb)
    print(f"{c.name}: {N} frames; flips per symmetry {np.bincount(flips, minlength=4).tolist()}; lag {lag0:.3f} -> "
          f"{lag1:.3f} -> {lag2:.3f} s; outline overlap median {np.median(before):.3f} -> {np.median(after):.3f}, "
          f"min {after.min():.3f}; frames < 0.7: {(after < 0.7).sum()}/{N}")
    return c2w, lag2


def compare(new_dir, old_dir):
    """Max camera-position difference (mm) and rotation difference (deg) between two runs, per file."""
    rep = {}
    for f in ("camera_vo_fp.npz", "camera_vo.npz"):
        a, b = np.load(Path(new_dir) / f)["camera_to_world"], np.load(Path(old_dir) / f)["camera_to_world"]
        n = min(len(a), len(b))
        dp = np.linalg.norm(a[:n, :3, 3] - b[:n, :3, 3], axis=1).max() * 1000
        tr = np.einsum("nij,nij->n", a[:n, :3, :3], b[:n, :3, :3])
        dr = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).max()
        rep[f] = (float(dp), float(dr), float(np.load(Path(new_dir) / f)["lag"]), float(np.load(Path(old_dir) / f)["lag"]))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--root")
    ap.add_argument("--out", help="default: the clip's outputs_dir")
    ap.add_argument("--tracker", default="fp_box", help="object-tracker pose dir under outputs_dir")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    root = mf.data_root(a.root)
    c = Clip(mf.load(mf.CLIPS / f"{a.clip}.json"), root)
    if a.check:
        with tempfile.TemporaryDirectory() as t:
            run(c, t, a.tracker)
            rep = compare(t, c.out)
        ok = True
        for f, (dp, dr, l_new, l_old) in rep.items():
            good = dp <= 1.0
            ok &= good
            print(f"  {f}: max position diff {dp:.3f} mm, rotation {dr:.4f} deg, lag {l_new:.3f} vs {l_old:.3f} "
                  f"-> {'OK' if good else 'DIFFERS'}")
        raise SystemExit(0 if ok else 1)
    run(c, a.out or c.out, a.tracker)


if __name__ == "__main__":
    main()
