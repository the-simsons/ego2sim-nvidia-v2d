"""Video-only camera path, refined by a pose graph over the whole clip (plans/RECONSTRUCTION_PLAN.md P4, after the
2026-11-19 error split: the camera path, not the object tracker, causes most of the object-trajectory error).

Variables: one camera pose per frame (a correction to the odometry, tools/rgbd_vo.py) and one world pose per stretch in
which the object rests. Residuals, each divided by its sigma, robust loss:
  odometry   relative motion between consecutive frames as the odometry measured it (weak where tracking was lost)
  table      the table plane fitted to each frame's depth near the object: its normal, seen from the camera, must map to
             the world table normal, and the camera's height above it must match. This holds roll, pitch and height
             against drift (3 of 6 degrees of freedom per frame).
  resting    while no hand touches the object (hand masks vs object mask), the object does not move: the camera pose
             times the tracker's camera <- object pose must equal that stretch's object pose. Ties the camera to the
             object over the still start, the still end, and any set-down in between.
  frame 0    fixed (the world is camera 0, as in the odometry).

    python tools/camera_pose_graph.py --vo camera_vo_video.npz --poses fp_box --mesh BlueBox.obj --depth moge_depth \
        --intrinsics '{"fx":713,"fy":713,"cx":960,"cy":540}' --object-masks box_masks --hand-masks hand_masks \
        --out camera_vo_pg.npz
Writes camera_to_world (same world as --vo) and OUT.json (planes, stretches, residual summaries).

Recommended (2026-11-19, 3 val clips): on the Pi3X camera (tools/pi3_camera.py) with --no-table --scale-every 10:
8.15 -> 5.62 cm, ADD-S AUC 0.549 -> 0.648. On rgbd_vo it did not help, and MoGe table planes hurt rotation in both."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e2s.geometry import load_pose_dir, mean_pose, runs  # noqa: E402
from e2s.masks import hand_contact, mask_dirs, union  # noqa: E402

SIGMA = {"scale_prior_log": 0.3, "scale_smooth_log": 0.05, "odo_rot_deg": 0.15, "odo_trans_m": 0.002, "lost_factor": 20.0, "table_normal_deg": 1.5, "table_height_m": 0.01,
         "rest_rot_deg": 2.0, "rest_pos_m": 0.015}


def decode_depth(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = 65535.0 / np.maximum(raw, 1.0) - 1.0
    d[(raw >= 65535) | (raw <= 0)] = 0
    return d


def fit_plane(P, n_pred=None, max_angle_deg=20.0, thresh=0.01, iters=300, rng=None):
    """RANSAC plane (n, d) with n . x + d = 0 and n pointing at the camera; candidates far from n_pred are skipped."""
    rng = rng or np.random.default_rng(0)
    if len(P) < 50:
        return None
    best, cos_max = (0, None), np.cos(np.radians(max_angle_deg))
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ a > 0:
            n = -n
        if n_pred is not None and n @ n_pred < cos_max:
            continue
        inl = np.abs((P - a) @ n) < thresh
        if inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    if best[1] is None or best[0] < 50:
        return None
    Q = P[best[1]]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c, full_matrices=False)[2][-1]
    if n @ c > 0:
        n = -n
    return n, float(-n @ c), best[0] / len(P)


def frame_planes(depth_dir, K, n, centres, excl_dirs, n_pred0, R_vo, step=4, radius=0.4):
    """Per-frame table plane in the camera frame near the object; normal predicted from frame 0 through the odometry."""
    planes = [None] * n
    rng = np.random.default_rng(0)
    for i in range(n):
        p = Path(depth_dir) / f"{i:06d}.png"
        if not p.is_file():
            continue
        d = decode_depth(p)[::step, ::step]
        ex = union(excl_dirs, i, d.shape)
        ex = cv2.dilate(ex, np.ones((7, 7), np.uint8)) > 0
        v, u = np.nonzero((d > 0.1) & (d < 3.0) & ~ex)
        z = d[v, u]
        P = np.c_[(u * step - K["cx"]) * z / K["fx"], (v * step - K["cy"]) * z / K["fy"], z]
        if centres is not None and not np.isnan(centres[i]).any():
            P = P[np.linalg.norm(P - centres[i], axis=1) < radius]
        if len(P) > 4000:
            P = P[rng.choice(len(P), 4000, replace=False)]
        pred = R_vo[i].T @ n_pred0 if n_pred0 is not None else None
        planes[i] = fit_plane(P, pred, rng=rng)
    return planes


def optimise(c2w0, Tcm, centre, planes, stretches, lost, sigma=SIGMA, scale_every=0):
    """scale_every > 0 adds one odometry scale per that many frames (log scale, prior and smoothness), for inputs whose
    translation scale wanders along the clip (Pi3X chunks are chained by Sim(3))."""
    N = len(c2w0)
    nS = int(np.ceil((N - 1) / scale_every)) if scale_every else 0
    seg = (np.arange(1, N) - 1) // scale_every if scale_every else np.zeros(N - 1, int)
    R0, t0 = c2w0[:, :3, :3], c2w0[:, :3, 3]
    rel_R = np.einsum("nji,njk->nik", R0[:-1], R0[1:])
    rel_t = np.einsum("nji,nj->ni", R0[:-1], t0[1:] - t0[:-1])
    so = np.where(lost[1:], sigma["lost_factor"], 1.0)
    s_or, s_ot = np.radians(sigma["odo_rot_deg"]) * so, sigma["odo_trans_m"] * so
    # world table plane: robust mean over frames with a plane, through the initial odometry (first second)
    have = [i for i in range(N) if planes[i] is not None]
    first = [i for i in have if i < max(30, have[0] + 1)] if have else []
    if first:
        nw = np.median([R0[i] @ planes[i][0] for i in first], axis=0)
        nw /= np.linalg.norm(nw)
        dw = float(np.median([planes[i][1] - nw @ t0[i] for i in first]))
    tab = np.array([i for i in have if i > 0])
    E0 = c2w0 @ np.nan_to_num(Tcm)
    W0 = [mean_pose(E0[a:b]) for a, b in stretches]
    rest_idx = [(k, i) for k, (a, b) in enumerate(stretches) for i in range(a, b) if not np.isnan(Tcm[i, 0, 0])]
    nW = 6 * len(stretches)
    nP = 6 * N + nW + nS

    def unpack(x):
        d = x[:6 * N].reshape(N, 6)
        R = Rot.from_rotvec(d[:, :3]).as_matrix() @ R0
        t = t0 + d[:, 3:]
        w = x[6 * N:6 * N + nW].reshape(-1, 6)
        WR = [Rot.from_rotvec(w[k, :3]).as_matrix() @ W0[k][:3, :3] for k in range(len(W0))]
        Wt = [W0[k][:3, 3] + w[k, 3:] for k in range(len(W0))]
        ls = x[6 * N + nW:]
        return R, t, WR, Wt, ls

    def residuals(x):
        R, t, WR, Wt, ls = unpack(x)
        sc = np.exp(ls[seg])[:, None] if nS else 1.0
        out = [x[:6] * 1e3]
        Rr = np.einsum("nji,njk->nik", R[:-1], R[1:])
        er = Rot.from_matrix(np.einsum("nji,njk->nik", rel_R, Rr)).as_rotvec() / s_or[:, None]
        et = (np.einsum("nji,nj->ni", R[:-1], t[1:] - t[:-1]) - sc * rel_t) / s_ot[:, None]
        out += [er.ravel(), et.ravel()]
        if len(tab):
            nc = np.array([planes[i][0] for i in tab])
            dc = np.array([planes[i][1] for i in tab])
            en = (np.einsum("nij,nj->ni", R[tab], nc) - nw) / np.radians(sigma["table_normal_deg"])
            eh = (t[tab] @ nw + dw - dc) / sigma["table_height_m"]
            out += [en.ravel(), eh]
        for k, i in rest_idx:
            ER = R[i] @ Tcm[i, :3, :3]
            Ec = ER @ centre + R[i] @ Tcm[i, :3, 3] + t[i]
            Wc = WR[k] @ centre + Wt[k]
            out += [Rot.from_matrix(WR[k].T @ ER).as_rotvec() / np.radians(sigma["rest_rot_deg"]),
                    (Ec - Wc) / sigma["rest_pos_m"]]
        if nS:
            out += [ls / sigma["scale_prior_log"], np.diff(ls) / sigma["scale_smooth_log"]]
        return np.concatenate(out)

    # Jacobian sparsity, in the residual order above
    rows = 6 + 6 * (N - 1) + (4 * len(tab)) + 6 * len(rest_idx) + (2 * nS - 1 if nS else 0)
    J = lil_matrix((rows, nP), dtype=int)
    J[0:6, 0:6] = 1
    r = 6
    for blk in range(2):
        for i in range(1, N):
            J[r:r + 3, 6 * (i - 1):6 * (i + 1)] = 1
            if blk == 1 and nS:
                J[r:r + 3, 6 * N + nW + seg[i - 1]] = 1
            r += 3
    for i in tab:
        J[r:r + 3, 6 * i:6 * i + 6] = 1
        r += 3
    for i in tab:
        J[r, 6 * i:6 * i + 6] = 1
        r += 1
    for k, i in rest_idx:
        J[r:r + 6, 6 * i:6 * i + 6] = 1
        J[r:r + 6, 6 * N + 6 * k:6 * N + 6 * k + 6] = 1
        r += 6
    for k in range(nS):
        J[r, 6 * N + nW + k] = 1
        r += 1
    for k in range(1, nS):
        J[r, 6 * N + nW + k - 1:6 * N + nW + k + 1] = 1
        r += 1
    assert r == rows, (r, rows)
    x0 = np.zeros(nP)
    r0 = residuals(x0)
    sol = least_squares(residuals, x0, jac_sparsity=J, loss="soft_l1", f_scale=2.0, x_scale=1e-2, max_nfev=50, verbose=0)
    R, t, _, _, ls = unpack(sol.x)
    T = np.tile(np.eye(4), (N, 1, 1))
    T[:, :3, :3], T[:, :3, 3] = R, t

    def part(res):
        o, k = {}, 6
        for name, m in (("odometry", 6 * (N - 1)), ("table", 4 * len(tab)), ("resting", 6 * len(rest_idx))):
            o[name] = float(np.sqrt(np.mean(res[k:k + m] ** 2))) if m else None
            k += m
        return o
    return T, {"rms_before": part(r0), "rms_after": part(sol.fun), "cost": float(sol.cost), "nfev": int(sol.nfev),
               "table_frames": int(len(tab)), "rest_frames": len(rest_idx),
               "odometry_scales": np.exp(ls).round(3).tolist() if nS else None,
               "world_table": {"normal": nw.tolist(), "d": dw} if first else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vo", required=True)
    ap.add_argument("--poses", required=True, help="object tracker, camera <- mesh JSONs")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--intrinsics", required=True, help="JSON file or inline JSON with fx, fy, cx, cy")
    ap.add_argument("--object-masks", required=True)
    ap.add_argument("--hand-masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-rest-frames", type=int, default=8)
    ap.add_argument("--no-table", action="store_true")
    ap.add_argument("--no-rest", action="store_true")
    ap.add_argument("--scale-every", type=int, default=0, help="frames per odometry scale variable (0 = fixed scale)")
    ap.add_argument("--odo-rot-deg", type=float, default=SIGMA["odo_rot_deg"])
    ap.add_argument("--odo-trans-m", type=float, default=SIGMA["odo_trans_m"])
    a = ap.parse_args(argv)
    K = json.loads(Path(a.intrinsics).read_text()) if Path(a.intrinsics).is_file() else json.loads(a.intrinsics)
    vo = np.load(a.vo)
    c2w0 = vo["camera_to_world"].astype(float)
    stats_p = Path(a.vo).with_suffix(".json")
    lost = np.zeros(len(c2w0), bool)
    if stats_p.is_file():
        for s in json.load(open(stats_p)).get("per_frame", []):
            lost[s["frame"]] = s["lost"]
    Tcm = load_pose_dir(a.poses, len(c2w0))
    N = len(c2w0)
    v = np.array([[float(x) for x in l.split()[1:4]] for l in open(a.mesh) if l.startswith("v ")])
    centre = (v.min(0) + v.max(0)) / 2
    centres = np.einsum("nij,j->ni", Tcm[:, :3, :3], centre) + Tcm[:, :3, 3]

    contact = hand_contact(a.object_masks, a.hand_masks, N)
    stretches = [] if a.no_rest else runs(~contact & ~np.isnan(Tcm[:, 0, 0]), a.min_rest_frames)
    planes = [None] * N
    if not a.no_table:
        excl = mask_dirs([a.object_masks]) + mask_dirs(a.hand_masks)
        first = frame_planes(a.depth, K, 1, centres, excl, None, c2w0[:, :3, :3])[0]
        n_w0 = c2w0[0, :3, :3] @ first[0] if first else None
        planes = frame_planes(a.depth, K, N, centres, excl, n_w0, c2w0[:, :3, :3])
    sig = dict(SIGMA, odo_rot_deg=a.odo_rot_deg, odo_trans_m=a.odo_trans_m)
    T, info = optimise(c2w0, Tcm, centre, planes, stretches, lost, sigma=sig, scale_every=a.scale_every)
    np.savez(a.out, camera_to_world=T, method="pose_graph", source=str(a.vo))
    info.update({"rest_stretches": [list(map(int, s)) for s in stretches], "contact_frames": int(contact.sum()),
                 "planes_found": int(sum(p is not None for p in planes)), "trajectory_source": "video"})
    json.dump(info, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    d = np.linalg.norm(T[:, :3, 3] - c2w0[:, :3, 3], axis=1)
    print(f"{a.out}: {N} frames, planes {info['planes_found']}, rest stretches {info['rest_stretches']}; rms before "
          f"{info['rms_before']} after {info['rms_after']}; camera moved median {np.median(d) * 100:.1f} max {d.max() * 100:.1f} cm")


if __name__ == "__main__":
    main()
