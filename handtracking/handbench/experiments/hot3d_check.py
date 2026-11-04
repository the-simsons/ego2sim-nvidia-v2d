"""Ground-truth check of the E1best methodology on 8 HOT3D Aria clips (true MANO hand GT from marker mocap).

E1best's own meshes come from a cached HaMeR run that cannot be re-run here, so the SAME recipe is applied to HaWoR's
camera-frame hands (handbench.runners.hawor_hot3d):
  raw            HaWoR monocular hands (its own depth)
  stereo-scale   + rescale about the camera so the rendered hand depth matches stereo depth (E1b-scale)
  e1best         + one hand size per side, depth re-aligned along the centroid ray, Savitzky-Golay (w7, o2) (E1best)
  stereo-global  ONE stereo scale per hand per clip (median of the per-frame ratios), no per-frame depth (+ savgol)
Stereo: the two Aria mono SLAM cameras, each re-projected into a virtual pinhole with the benchmark camera's
orientation, then the same SGBM code as Track 3 (handbench.stereo). No SAM2 masks on HOT3D: depth is compared over the
rendered silhouette eroded by 2 px (Track 3 used mask & silhouette).
Frame = upright pinhole view of the RGB camera (handbench.datasets.hot3d). Scored: frames where the GT hand is in
front of the camera with its wrist inside the image AND HaWoR has an observed hand of that side.

Usage: python -m handbench.experiments.hot3d_check      -> results/hot3d_check.json
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
from scipy.signal import savgol_filter

from handbench import paths
from handbench.adapters.hawor import to_ours
from handbench.calib import Cam
from handbench.datasets import hot3d as H
from handbench.fit import refit_global
from handbench.render import depth_map
from handbench.stereo import StereoDepth

OUT = paths.ROOT / "data" / "hot3d_hawor"
METHODS = ("raw", "stereo-scale", "e1best", "stereo-global", "stereo-global+savgol")
SIDES = {0: ("left", False), 1: ("right", True)}
VSIZE, VF = 640, 300.0  # virtual pinhole for the mono cameras


def rig(clip):
    """Benchmark camera P + virtual left/right pinholes (P's orientation, mono-camera centres) as handbench Cams, and
    the remap maps that render the virtual views from the fisheye mono images."""
    cams = clip.cameras(0)
    T_w_rgb = cams[H.RGB]["T_world_from_camera"]
    T_rgb_P = np.eye(4)
    T_rgb_P[:3, :3] = H.R_CAM_FROM_PIN
    out, maps = {}, {}
    centres = {}
    for s in (H.SLAM_L, H.SLAM_R):
        T_s_P = np.linalg.inv(cams[s]["T_world_from_camera"]) @ T_w_rgb @ T_rgb_P
        centres[s] = -T_s_P[:3, :3].T @ T_s_P[:3, 3]
        Kv = np.array([[VF, 0, VSIZE / 2], [0, VF, VSIZE / 2], [0, 0, 1.0]])
        maps[s] = H.pinhole_from_fisheye_maps(cams[s]["params"], Kv, (VSIZE, VSIZE), T_s_P[:3, :3])
    left, right = sorted(centres, key=lambda s: centres[s][0])  # smaller x in P = left
    Kv = np.array([[VF, 0, VSIZE / 2], [0, VF, VSIZE / 2], [0, 0, 1.0]])
    for name, s in (("ego_cam_c", left), ("ego_cam_b", right)):
        T = np.eye(4)
        T[:3, 3] = -centres[s]
        out[name] = Cam(name, Kv, np.zeros(5), (VSIZE, VSIZE), T)
    out["ego_cam_a"] = Cam("ego_cam_a", H.pinhole_K(), np.zeros(5), (H.SIZE_PIN, H.SIZE_PIN))
    baseline = float(np.linalg.norm(centres[left] - centres[right]))
    return out, maps, (left, right), baseline


def gt_in_P(clip, mano, i, side, is_right):
    hands = clip.json(i, "hands")
    if side not in hands:
        return None, None
    J, V = mano.world_joints_verts(hands[side], clip.shape, is_right)
    T_cam_w = np.linalg.inv(clip.cameras(i)[H.RGB]["T_world_from_camera"])
    R = H.R_CAM_FROM_PIN.T  # cam -> P
    f = lambda X: (X @ T_cam_w[:3, :3].T + T_cam_w[:3, 3]) @ R.T
    return f(J), f(V)


def render_depth_half(V, faces, K):
    s = 0.5
    Kh = K.copy()
    Kh[:2] *= s
    Kh[:2, 2] = (K[:2, 2] + 0.5) * s - 0.5
    uv = V[:, :2] / V[:, 2:3] * np.diag(Kh)[:2] + Kh[:2, 2]
    n = int(H.SIZE_PIN * s)
    return depth_map(uv, V[:, 2], faces, (n, n))


def stereo_ratio(V, faces, K, D, mode="ratio"):
    R = render_depth_half(V, faces, K)
    m = cv2.erode((R > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    g = m & (D > 0)
    if g.sum() < 30:
        return None
    return float(np.median(D[g] / R[g])) if mode == "ratio" else float(np.median(D[g] - R[g]))


def metrics(J, G):
    e = J - G
    r = G / np.linalg.norm(G, axis=-1, keepdims=True)
    along = (e * r).sum(-1)
    lat = np.linalg.norm(e - along[..., None] * r, axis=-1)
    ec = e - e.mean(1, keepdims=True)
    return {"mpjpe": 1000 * np.linalg.norm(e, axis=-1).mean(1), "depth": 1000 * np.abs(along).mean(1),
            "lateral": 1000 * lat.mean(1), "bias": 1000 * along.mean(1), "pose": 1000 * np.linalg.norm(ec, axis=-1).mean(1)}


def main():
    t0 = time.time()
    mano = H.Hot3dMano()
    K = H.pinhole_K()
    rows = []
    per_clip = {}
    for cp in H.clips():
        ex_p = OUT / cp.stem / "export.npz"
        if not ex_p.exists():
            print(f"{cp.stem}: no HaWoR export, skipped")
            continue
        clip = H.Clip(cp)
        ex = np.load(ex_p)
        cams, maps, (left, right), baseline = rig(clip)
        sd = StereoDepth(cams, out_scale=0.5)
        n = min(clip.n, ex["observed"].shape[1])
        depth = {}
        info = {"baseline_m": baseline, "n": n}
        for h, (side, is_right) in SIDES.items():
            faces = mano.mano.faces_for(is_right)
            obs = ex["observed"][h, :n] & np.isfinite(ex["cam_verts"][h, :n, 0, 0])
            G = np.full((n, 21, 3), np.nan)
            Jraw = np.full((n, 21, 3), np.nan)
            Vraw = np.full((n, 778, 3), np.nan)
            for i in range(n):
                g, _ = gt_in_P(clip, mano, i, side, is_right)
                if g is not None and (g[:, 2] > 0.1).all():
                    w = g[0, :2] / g[0, 2] * K[0, 0] + K[:2, 2]
                    if (0 <= w).all() and (w < H.SIZE_PIN).all():
                        G[i] = g
                if obs[i]:
                    V = ex["cam_verts"][h, i].astype(float)
                    hp = to_ours(ex["cam_pose"][h, i], is_right)
                    go, tr, s = refit_global(mano.mano, hp, ex["cam_betas"][h, i], is_right, V)
                    Vf, Jf = mano.mano.forward(go, hp, ex["cam_betas"][h, i], tr, is_right, s)
                    Vraw[i], Jraw[i] = Vf[0], Jf[0]
            # stereo depth per frame (lazily, only frames with a hand)
            s_t = np.full(n, np.nan)
            for i in np.nonzero(obs)[0]:
                if i not in depth:
                    imL = cv2.remap(clip.image(i, left), *maps[left], cv2.INTER_LINEAR)
                    imR = cv2.remap(clip.image(i, right), *maps[right], cv2.INTER_LINEAR)
                    depth[i] = sd.compute(imL, imR)
                r = stereo_ratio(Vraw[i], faces, K, depth[i])
                if r is not None and 0.3 < r < 3.0:
                    s_t[i] = r
            Jsc, Vsc = Jraw.copy(), Vraw.copy()
            ok = np.isfinite(s_t)
            Jsc[ok] *= s_t[ok, None, None]
            Vsc[ok] *= s_t[ok, None, None]
            # E1best: one size per side (median per-frame scale), centroid where the per-frame scale put it, then a dz
            # along the centroid ray to the stereo depth, then Savitzky-Golay over runs of consecutive frames
            S = float(np.nanmedian(s_t)) if ok.any() else 1.0
            Jb, Vb = Jsc.copy(), Vsc.copy()
            for i in np.nonzero(obs)[0]:
                c = Vraw[i].mean(0)
                c_new = c * (s_t[i] if ok[i] else S)
                Vb[i] = S * (Vraw[i] - c) + c_new
                Jb[i] = S * (Jraw[i] - c) + c_new
                dz = stereo_ratio(Vb[i], faces, K, depth[i], mode="diff")
                if dz is not None and abs(dz) < 0.3:
                    off = c_new / c_new[2] * dz
                    Vb[i] += off
                    Jb[i] += off
            idx = np.nonzero(obs)[0]
            runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1) if len(idx) else []
            for run in runs:
                if len(run) >= 7:
                    Jb[run] = savgol_filter(Jb[run], 7, 2, axis=0, mode="interp")
            scored = obs & np.isfinite(G[:, 0, 0])
            visible = np.isfinite(G[:, 0, 0])
            info[side] = {"visible_gt_frames": int(visible.sum()), "scored": int(scored.sum()),
                          "coverage": float(scored.sum() / max(visible.sum(), 1)), "stereo_ok_frac": float(ok[obs].mean()) if obs.any() else 0,
                          "shared_scale": S}
            Jg = Jraw * S  # one stereo scale per hand per clip (median of the per-frame ratios), no per-frame depth
            Jgs = Jg.copy()
            for run in runs:
                if len(run) >= 7:
                    Jgs[run] = savgol_filter(Jg[run], 7, 2, axis=0, mode="interp")
            for name, J in (("raw", Jraw), ("stereo-scale", Jsc), ("e1best", Jb), ("stereo-global", Jg),
                            ("stereo-global+savgol", Jgs)):
                m = metrics(J[scored], G[scored])
                for k in range(scored.sum()):
                    rows.append({"clip": cp.stem, "side": side, "method": name, **{a: float(b[k]) for a, b in m.items()}})
            # jitter (m/s^2) on consecutive scored frames, 30 fps
            for name, J in (("raw", Jraw), ("stereo-scale", Jsc), ("e1best", Jb), ("stereo-global", Jg),
                            ("stereo-global+savgol", Jgs)):
                c3 = scored[1:-1] & scored[:-2] & scored[2:]
                if c3.any():
                    kk = np.nonzero(c3)[0] + 1
                    acc = np.linalg.norm(J[kk + 1] - 2 * J[kk] + J[kk - 1], axis=-1).mean(1) * 30 ** 2
                    info[side][f"jitter_{name}"] = float(acc.mean())
        per_clip[cp.stem] = info
        print(cp.stem, json.dumps({s: {k: round(v, 3) if isinstance(v, float) else v for k, v in info[s].items()}
                                   for s in ("left", "right") if s in info}), f"baseline {baseline:.3f} m", flush=True)
    import pandas as pd

    df = pd.DataFrame(rows)
    summ = df.groupby("method")[["mpjpe", "depth", "lateral", "bias", "pose"]].agg(["mean", "median"]).round(1)
    jit = {m: float(np.nanmean([per_clip[c][s].get(f"jitter_{m}", np.nan) for c in per_clip for s in ("left", "right")
                                if s in per_clip[c]])) for m in METHODS}
    print(summ.to_string())
    print("jitter m/s^2:", {k: round(v, 2) for k, v in jit.items()})
    res = {"summary": {m: {k: float(df[df.method == m][k].mean()) for k in ("mpjpe", "depth", "lateral", "bias", "pose")}
                       | {f"{k}_median": float(df[df.method == m][k].median()) for k in ("mpjpe", "bias")}
                       | {"jitter": jit[m], "n_hand_frames": int((df.method == m).sum())}
                       for m in METHODS},
           "per_clip_mpjpe": df.groupby(["clip", "method"]).mpjpe.mean().unstack().round(1).to_dict(orient="index"),
           "per_clip_info": per_clip, "wall_time_s": time.time() - t0}
    json.dump(res, open(paths.RESULTS / "hot3d_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
