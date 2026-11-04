"""Measure the floor: how far is the triangulated pseudo-GT (Track 3 refs, build_refs) from the truth?

The Track 3 reference protocol is replayed on the 8 HOT3D Aria clips, where marker-mocap MANO GT exists:
  views      a = upright pinhole of the RGB camera (benchmark frame), b / c = right / left mono SLAM cameras re-projected
             into virtual pinholes (hot3d_check.rig); b / c are detected in grey, as on Track 3
  crops      build_refs' two-pass RTMPose crops; pass 1 is placed on the hand silhouette in a, transferred to b / c over
             the same 0.25-0.9 m depth band. HOT3D has no SAM2 masks: the GT hand silhouette stands in for them
  gate       "none" (no mask-consistency gate) and "gt-mask" (build_refs' gate, GT silhouettes in a and b instead of SAM2
             masks). The two bracket Track 3, whose SAM2 masks are imperfect
  triangulate build_refs.triangulate_kp, unchanged (score >= 0.5, reprojection < 3 px in every used view)
Then:
  floor      kept pseudo-GT joints vs GT joints (mm), split by the views used. On Aria the RGB camera is 1.5 cm from the
             left mono camera, so a+c joints have almost no baseline; a+b (12.6 cm) and a+b+c are well conditioned
  offsets    the fixed part of the error per (side, joint) in the hand's own frame (MANO joints vs the detector's
             annotated keypoints), median over well-conditioned joints
  transfer   depth-error model: |along-ray error| = sigma * g, g = z^2 / B * sqrt(1/fa^2 + 1/fv^2) per camera pair
             (crop focals f, so sigma is in crop px), pairs combined as 1/g^2 = sum 1/g_pair^2; lateral = sigma_lat * z / fa.
             sigma is fitted per view combination on HOT3D (the spread over an 8x baseline range is the model check) and
             applied to Track 3's own reference joints, views and pass-2 crop sizes
  methods    HaWoR (raw, as hot3d_check) scored against the pseudo-GT and against the GT on the same joints

Usage: python -m handbench.experiments.hot3d_floor [--jobs 8] [--skip-detect]
       -> data/hot3d_floor/<clip>.npz (keypoints, cached), results/hot3d_floor.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from handbench import paths
from handbench import build_refs as B
from handbench.calib import load_rig
from handbench.crops import PerspectiveCrop, look_at
from handbench.datasets import hot3d as H
from handbench.experiments import hot3d_check as C
from handbench.render import depth_map

OUT = paths.ROOT / "data" / "hot3d_floor"
SIDES = ((0, "left", False), (1, "right", True))  # index matches build_refs.SIDES (L, R)
GATES = ("none", "gt-mask")
VIEW_KIND = ("rgb", "mono", "mono")
COMBOS = ("ab", "ac", "abc")
WELL = ("ab", "abc")  # well-conditioned on HOT3D


def silhouette(cam, V, faces):
    X = cam.to_cam(V)
    uv = X[:, :2] / X[:, 2:3] * np.diag(cam.K)[:2] + cam.K[:2, 2]
    return depth_map(uv, X[:, 2], faces, cam.size) > 0


def detect_clip(cp, source="gt", out=OUT):
    """build_refs.keypoints_episode + gate_episode on one HOT3D clip. The hand silhouettes that place the crops and
    feed the gate stand in for SAM2 masks: source "gt" = GT hand meshes (reference study), "hawor" = HaWoR's own
    hands (no GT input, for methods built on these keypoints). G (GT joints) is stored either way, for scoring."""
    t0 = time.time()
    rtm, _, _ = B._models()
    clip, mano = H.Clip(cp), H.Hot3dMano()
    cams, maps, (left, right), baseline = C.rig(clip)
    K = H.pinhole_K()
    maps["a"] = H.pinhole_from_fisheye_maps(clip.cameras(0)[H.RGB]["params"], K, (H.SIZE_PIN, H.SIZE_PIN), H.R_CAM_FROM_PIN)
    A = cams["ego_cam_a"]
    n = clip.n
    if source == "hawor":
        ex = np.load(C.OUT / cp.stem / "export.npz")
        n_h = min(n, ex["observed"].shape[1])
        Vh = [hawor_hands(clip, ex, mano, s, ir, n_h)[1] for s, _, ir in SIDES]
    kp = np.full((n, 2, 3, 21, 2), np.nan, np.float32)
    sc = np.zeros((n, 2, 3, 21), np.float32)
    crop_f = np.full((n, 2, 3), np.nan, np.float32)
    G = np.full((n, 2, 21, 3), np.nan)
    area = np.zeros((n, 2), np.int32)
    inside = np.full((n, 2, 2), np.nan)
    for i in range(n):
        fr = None
        for s, side, is_right in SIDES:
            faces = mano.mano.faces_for(is_right)
            g, V = C.gt_in_P(clip, mano, i, side, is_right)
            if g is not None and (V[:, 2] > 0.1).all() and silhouette(A, V, faces).sum() >= B.MIN_MASK_PX:
                G[i, s] = g
            if source == "hawor":
                V = Vh[s][i] if i < n_h and np.isfinite(Vh[s][i, 0, 0]) else None
            elif not np.isfinite(G[i, s, 0, 0]):
                V = None
            if V is None or not (V[:, 2] > 0.1).all():
                continue
            ma = silhouette(A, V, faces)
            area[i, s] = int(ma.sum())
            if area[i, s] < B.MIN_MASK_PX:
                continue
            if fr is None:
                fr = {"ego_cam_a": cv2.remap(clip.image(i, H.RGB), *maps["a"], cv2.INTER_LINEAR),
                      "ego_cam_b": cv2.remap(clip.image(i, right), *maps[right], cv2.INTER_LINEAR),
                      "ego_cam_c": cv2.remap(clip.image(i, left), *maps[left], cv2.INTER_LINEAR)}
            ys, xs = np.nonzero(ma)
            pts = np.c_[xs, ys][:: max(1, len(xs) // 300)].astype(float)
            rays_a = np.c_[(pts - A.K[:2, 2]) / np.diag(A.K)[:2], np.ones(len(pts))]
            for v, cn in enumerate(B.VIEWS):
                c = cams[cn]
                rays = rays_a if cn == "ego_cam_a" else c.to_cam(np.concatenate([rays_a * z for z in B.DEPTH_BAND]))
                if (rays[:, 2] <= 0.05).any():
                    continue
                crop = PerspectiveCrop.around(c, rays, 256, 1.3)
                got = B._detect(rtm, crop, fr[cn], cn != "ego_cam_a")
                if got is None:
                    continue
                k, s_, img = got
                good = s_ >= 0.3
                if good.sum() >= 5:
                    n2 = crop.crop_px_to_norm(k[good])
                    crop2 = PerspectiveCrop.around(c, np.c_[n2, np.ones(len(n2))], 256, 1.6)
                    got2 = B._detect(rtm, crop2, fr[cn], cn != "ego_cam_a")
                    if got2 is not None and np.median(got2[1]) >= np.median(s_) - 0.05:
                        crop, (k, s_, img) = crop2, got2
                kp[i, s, v] = crop.crop_px_to_img(k)
                sc[i, s, v] = s_
                crop_f[i, s, v] = crop.f
            # mask-consistency gate inputs (build_refs.gate_episode): a and b only, c has no mask on Track 3 either
            mb = silhouette(cams["ego_cam_b"], V, faces)
            inside[i, s] = (B._inside_frac(ma, kp[i, s, 0].astype(float), sc[i, s, 0]),
                            B._inside_frac(mb, kp[i, s, 1].astype(float), sc[i, s, 1]))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{clip.name}.npz", kp=kp, score=sc, crop_f=crop_f, G=G, mask_area_a=area, inside=inside,
                        baseline=baseline, meta=np.array(json.dumps({"wall_time_s": time.time() - t0})))
    return clip.name, time.time() - t0


def gated(d, gate):
    sc = d["score"].copy()
    if gate == "gt-mask":
        keep = np.ones(sc.shape[:3], bool)
        keep[:, :, :2] = ~(d["inside"] < B.GATE_MIN_INSIDE)
        sc *= keep[..., None]
    return sc


def along_lateral(e, G):
    r = G / np.linalg.norm(G, axis=-1, keepdims=True)
    along = (e * r).sum(-1)
    return along, np.linalg.norm(e - along[..., None] * r, axis=-1)


def hand_frames(G):
    """(…,21,3) joints -> (…,3,3) rotation rows: x wrist->middle knuckle, y index->pinky knuckle (orthogonalised)."""
    x = G[..., 9, :] - G[..., 0, :]
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    y = G[..., 17, :] - G[..., 5, :]
    y -= (y * x).sum(-1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], -2)


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90))}


def crop_f_from_joints(cam, X):
    """Focal of build_refs' pass-2 crop (PerspectiveCrop.around, margin 1.6) around joints X (N,3, a frame) in cam."""
    c = cam.to_cam(X)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    q = c @ look_at(c.mean(0)).T
    half = np.abs(q[:, :2] / q[:, 2:3]).max() * 1.6
    return max(50.0, 128 / max(half, 1e-3))


def sensitivity(cl, X, U, cf):
    """Per joint: g = along-ray metres per crop px of its view combination, lat = lateral metres per crop px in a.
    X (N,3) a frame, U (N,3) views used, cf (N,3) crop focals."""
    z = cl[0].depth(X)
    inv = np.zeros(len(X))
    for v in (1, 2):
        b = np.linalg.norm(cl[v].center_in_a - cl[0].center_in_a)
        g = z ** 2 / b * np.sqrt(1 / cf[:, 0] ** 2 + 1 / cf[:, v] ** 2)
        m = U[:, v] & np.isfinite(g)
        inv[m] += 1 / g[m] ** 2
    with np.errstate(divide="ignore"):
        return 1 / np.sqrt(inv), z / cf[:, 0]


def combo(U):
    return np.where(U.sum(1) == 3, "abc", np.where(U[:, 1], "ab", "ac"))


def hawor_hands(clip, ex, mano, h, is_right, n):
    """HaWoR raw camera-frame joints (n,21,3) and verts (n,778,3) in P, exactly as hot3d_check builds them."""
    from handbench.adapters.hawor import to_ours
    from handbench.fit import refit_global

    obs = ex["observed"][h, :n] & np.isfinite(ex["cam_verts"][h, :n, 0, 0])
    J = np.full((n, 21, 3), np.nan)
    V = np.full((n, 778, 3), np.nan)
    for i in np.nonzero(obs)[0]:
        hp = to_ours(ex["cam_pose"][h, i], is_right)
        go, tr, s = refit_global(mano.mano, hp, ex["cam_betas"][h, i], is_right, ex["cam_verts"][h, i].astype(float))
        Vf, Jf = mano.mano.forward(go, hp, ex["cam_betas"][h, i], tr, is_right, s)
        V[i], J[i] = Vf[0], Jf[0]
    return J, V


def track3_sensitivity(cl):
    """g, lat (as sensitivity) and the view combination of every Track 3 tri_abc reference joint."""
    g_all, l_all, c_all = [], [], []
    for ep in paths.episodes("v2d_dev"):
        tp = paths.REFS / ep / "tri.npz"
        if not tp.exists():
            continue
        t = np.load(tp)
        X, U = t["tri_abc"].astype(float), t["tri_abc_views"]
        k = np.isfinite(X[..., 0])
        cf = np.full(X.shape[:2] + (3,), np.nan)
        for fi, si in zip(*np.nonzero(k.any(-1))):
            for v in range(3):
                m = k[fi, si] & U[fi, si, :, v]
                if m.sum() >= 3:
                    cf[fi, si, v] = crop_f_from_joints(cl[v], X[fi, si, m])
        f_s, s_s, _ = np.nonzero(k)
        g, lat = sensitivity(cl, X[k], U[k], cf[f_s, s_s])
        ok = np.isfinite(g) & np.isfinite(lat)
        g_all.append(g[ok])
        l_all.append(lat[ok])
        c_all.append(combo(U[k])[ok])
    return np.concatenate(g_all), np.concatenate(l_all), np.concatenate(c_all)


def score():
    mano = H.Hot3dMano()
    res = {"floor": {}, "detector_2d": {}, "offsets": {}, "transfer": {}, "methods": {}, "per_clip": {}}
    acc = {g: {k: [] for k in ("err", "combo", "visible")} for g in GATES}
    j3 = {k: [] for k in ("along", "lat", "g", "latg", "combo", "joint", "side", "local", "err")}  # gt-mask, kept joints
    px2d = {"rgb": [], "mono": []}
    meth = {k: [] for k in ("vs_tri", "vs_gt", "combo")}
    meth_all = []
    for cp in H.clips():
        p = OUT / f"{cp.stem}.npz"
        if not p.exists():
            print(f"{cp.stem}: no keypoints, skipped")
            continue
        d = np.load(p)
        clip = H.Clip(cp)
        cams, _, _, baseline = C.rig(clip)
        cl = [cams[v] for v in B.VIEWS]
        G = d["G"]
        vis = np.isfinite(G[..., 0, 0])
        info = {"baseline_bc_m": float(baseline), "visible_hand_frames": int(vis.sum())}
        for gate in GATES:
            sc = gated(d, gate)
            r = B.triangulate_kp(cl, d["kp"], sc)
            X = r["tri_abc"]
            k = np.isfinite(X[..., 0]) & vis[..., None]
            f_s, s_s, j_s = np.nonzero(k)
            e = X[k] - G[k]
            cmb = combo(r["tri_abc_views"][k])
            acc[gate]["err"].append(1000 * np.linalg.norm(e, axis=-1))
            acc[gate]["combo"].append(cmb)
            acc[gate]["visible"].append(int(vis.sum() * 21))
            info[f"{gate}_mpjpe"] = float(1000 * np.linalg.norm(e, axis=-1).mean()) if k.any() else None
            info[f"{gate}_coverage"] = float(k.sum() / max(vis.sum() * 21, 1))
            if gate != "gt-mask":
                continue
            al, la = along_lateral(e, G[k])
            g, latg = sensitivity(cl, G[k], r["tri_abc_views"][k], d["crop_f"][f_s, s_s].astype(float))
            for name, val in (("along", 1000 * al), ("lat", 1000 * la), ("g", g), ("latg", latg), ("combo", cmb),
                              ("joint", j_s), ("side", s_s), ("err", 1000 * np.linalg.norm(e, axis=-1)),
                              ("local", 1000 * np.einsum("nab,nb->na", hand_frames(G)[f_s, s_s], e))):
                j3[name].append(val)
            # 2D detector errors of the views that enter triangulation (score >= 0.5 after the gate)
            for v in range(3):
                ok = (sc[:, :, v] >= 0.5) & np.isfinite(d["kp"][:, :, v, :, 0]) & vis[..., None]
                fv, sv, jv = np.nonzero(ok)
                r2 = d["kp"][fv, sv, v, jv].astype(float) - cl[v].project(G[fv, sv, jv], distort=False)
                px = np.linalg.norm(r2, axis=1)
                px2d[VIEW_KIND[v]].append(np.c_[px, px * d["crop_f"][fv, sv, v] / cl[v].K[0, 0]])
            # HaWoR against the pseudo-GT and against the GT
            ex_p = C.OUT / cp.stem / "export.npz"
            if ex_p.exists():
                ex = np.load(ex_p)
                n = min(clip.n, ex["observed"].shape[1])
                Jh = np.stack([hawor_hands(clip, ex, mano, s, ir, n)[0] for s, _, ir in SIDES], 1)  # (n,2,21,3)
                ok_h = np.isfinite(Jh[..., :1, 0])
                kk = k[:n] & ok_h
                meth["vs_tri"].append(1000 * np.linalg.norm(Jh[kk] - X[:n][kk], axis=-1))
                meth["vs_gt"].append(1000 * np.linalg.norm(Jh[kk] - G[:n][kk], axis=-1))
                meth["combo"].append(combo(r["tri_abc_views"][:n][kk]))
                ka = np.broadcast_to(vis[:n, :, None] & ok_h, Jh.shape[:3])
                meth_all.append(1000 * np.linalg.norm(Jh - G[:n], axis=-1)[ka])
        res["per_clip"][cp.stem] = info
        print(cp.stem, {k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()}, flush=True)

    for gate in GATES:
        err, cmb = np.concatenate(acc[gate]["err"]), np.concatenate(acc[gate]["combo"])
        res["floor"][gate] = {"tri_abc_mm": stats(err), "coverage": float(len(err) / sum(acc[gate]["visible"])),
                              "well_conditioned_mm": stats(err[np.isin(cmb, WELL)]),
                              "by_combo": {c: stats(err[cmb == c]) | {"share": float((cmb == c).mean())} for c in COMBOS}}
    a = {k: np.concatenate(v) for k, v in j3.items()}
    well = np.isin(a["combo"], WELL)
    res["floor"]["gt-mask"]["along_ray_bias_mm"] = {c: float(np.median(a["along"][a["combo"] == c])) for c in COMBOS}
    res["floor"]["gt-mask"]["lateral_mm"] = stats(a["lat"])
    for kind in px2d:
        x = np.concatenate(px2d[kind])
        res["detector_2d"][kind] = {"image_px": stats(x[:, 0]), "crop_px": stats(x[:, 1])}

    # fixed offset per (side, joint), median vector in the hand frame over well-conditioned joints
    key = a["side"] * 21 + a["joint"]
    off = np.zeros((42, 3))
    for q in np.unique(key[well]):
        off[q] = np.median(a["local"][well & (key == q)], 0)
    res["offsets"] = {"fixed_offset_mm": stats(np.linalg.norm(off[key[well]], axis=1)),
                      "remaining_after_offset_mm": stats(np.linalg.norm(a["local"][well] - off[key[well]], axis=1)),
                      "per_joint_mm": np.linalg.norm(off, axis=1).reshape(2, 21).round(1).tolist()}

    # depth-error model: sigma per view combination, checked by how well one sigma predicts every combination
    sig = {c: float(np.median(np.abs(a["along"][a["combo"] == c]) / (1000 * a["g"][a["combo"] == c]))) for c in COMBOS}
    sig_lat = float(np.median(a["lat"] / (1000 * a["latg"])))
    sig_mid = float(np.median(np.abs(a["along"][well]) / (1000 * a["g"][well])))
    tail = float(np.mean(a["err"][well]) / np.median(a["err"][well]))  # mean / median of the measured error
    pred = lambda s, g, latg: 1000 * np.sqrt((s * g) ** 2 + (sig_lat * latg) ** 2)
    res["transfer"]["model"] = {"sigma_along_crop_px": sig, "sigma_well_conditioned": sig_mid,
                                "sigma_lateral_crop_px": sig_lat, "mean_over_median": tail,
                                "hot3d_check": {c: {"measured_median_mm": float(np.median(a["err"][a["combo"] == c])),
                                                    "predicted_median_mm": float(np.median(pred(sig_mid, a["g"][a["combo"] == c],
                                                                                                a["latg"][a["combo"] == c])))}
                                                for c in COMBOS}}
    cams3 = load_rig(paths.rig_json())
    g3, l3, c3 = track3_sensitivity([cams3[v] for v in B.VIEWS])
    lo, hi = min(sig.values()), max(sig.values())
    res["transfer"]["track3"] = {
        "n_joints": int(len(g3)), "share": {c: float((c3 == c).mean()) for c in COMBOS},
        "g_median_mm_per_px": float(1000 * np.median(g3)),
        "hot3d_g_median_mm_per_px": {c: float(1000 * np.median(a["g"][a["combo"] == c])) for c in COMBOS},
        "predicted_median_mm": {"low": float(np.median(pred(lo, g3, l3))), "mid": float(np.median(pred(sig_mid, g3, l3))),
                                "high": float(np.median(pred(hi, g3, l3)))},
        "predicted_mean_mm_with_hot3d_tail": {"low": float(np.median(pred(lo, g3, l3)) * tail),
                                              "mid": float(np.median(pred(sig_mid, g3, l3)) * tail),
                                              "high": float(np.median(pred(hi, g3, l3)) * tail)}}

    vt, vg, cm = (np.concatenate(meth[k]) for k in ("vs_tri", "vs_gt", "combo"))
    w = np.isin(cm, WELL)
    res["methods"]["hawor_raw"] = {"all_scored": {"vs_tri": stats(vt), "vs_gt": stats(vg)},
                                   "well_conditioned": {"vs_tri": stats(vt[w]), "vs_gt": stats(vg[w])},
                                   "vs_gt_every_visible_joint": stats(np.concatenate(meth_all))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--skip-detect", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if not a.skip_detect:
        todo = [c for c in H.clips() if not (OUT / f"{c.stem}.npz").exists()]
        with ProcessPoolExecutor(a.jobs) as ex:
            for name, dt in ex.map(detect_clip, todo):
                print(f"{name}: keypoints in {dt:.0f}s", flush=True)
    res = score()
    res["wall_time_s"] = time.time() - t0
    json.dump(res, open(paths.RESULTS / "hot3d_floor.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("floor", "detector_2d", "offsets", "transfer", "methods")}, indent=1))


if __name__ == "__main__":
    main()
