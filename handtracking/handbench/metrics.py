"""Benchmark metrics (CPU / NumPy + OpenCV only; no CUDA imports).

Scored rows: the hand is visible in the input view (EpisodeRefs.visible) AND the method gives an observed (not
infilled) valid hand there. Everything else only enters the coverage line.

Per (frame, side) row:
  mpjpe_<ref>        mean joint error vs triangulated pseudo-GT (mm, no alignment), joints with a reference only
  depth_<ref>        mean |error along the cam_a ray of the reference joint| (mm)
  lateral_<ref>      mean |error perpendicular to that ray| (mm)
  bias_<ref>         mean signed depth error (mm, + = method too far)
  ca_mpjpe_<ref>     MPJPE after removing the mean translation (pose/shape error, mm)
  kp_<view>, pck10_<view>   2D error of the method's joints projected into that view vs that view's own detector
                     keypoints (score >= KP_MIN_SCORE), px of that view's published image
  iou_a, iou_b, iou_c  rendered MANO silhouette vs SAM2 hand mask in cam_a (benchmark px) / cam_b, cam_c (published px),
                     both clipped to the hand region from that view's detector keypoints (masks include the
                     forearm); iou_*_full = unclipped
  stereo_abs, stereo_signed   median |rendered depth - SGBM| / signed median over mask pixels (mm). Uses cam_b!
Episode-level temporal metrics per side over runs of consecutive scored frames.
"""
from __future__ import annotations

import numpy as np

from handbench.format import HandTracks
from handbench.mano import BONES, Mano
from handbench.render import depth_map, iou, silhouette

SIDES = ("L", "R")
KP_MIN_SCORE = 0.5
MIN_GT_JOINTS = 5
PCK_PX = 10.0


def joint_errors(J: np.ndarray, G: np.ndarray) -> dict:
    """J, G (21,3) metres (G NaN where missing)."""
    m = np.isfinite(G[:, 0])
    if m.sum() < MIN_GT_JOINTS:
        return {}
    e = J[m] - G[m]
    r = G[m] / np.linalg.norm(G[m], axis=1, keepdims=True)
    along = (e * r).sum(1)
    lat = np.linalg.norm(e - along[:, None] * r, axis=1)
    ec = e - e.mean(0)
    return {"mpjpe": 1000 * np.linalg.norm(e, axis=1).mean(), "depth": 1000 * np.abs(along).mean(),
            "lateral": 1000 * lat.mean(), "bias": 1000 * along.mean(),
            "ca_mpjpe": 1000 * np.linalg.norm(ec, axis=1).mean(), "n_joints": int(m.sum())}


def hand_region(px: np.ndarray, score: np.ndarray, shape: tuple, min_score: float = 0.3, grow: float = 0.25):
    """Hand-only region of a view, from THAT view's own detector keypoints (method-independent): convex hull of the
    confident keypoints dilated by `grow` x hull size. SAM2 'hand' masks include the forearm; IoU is computed on
    mask & region and silhouette & region."""
    import cv2

    m = (score >= min_score) & np.isfinite(px[:, 0])
    if m.sum() < 5:
        return None
    p = px[m].astype(np.float32)
    hull = cv2.convexHull(p).reshape(-1, 2)
    size = float(np.linalg.norm(p.max(0) - p.min(0)))
    reg = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(reg, np.round(hull).astype(np.int32), 1)
    r = max(3, int(grow * size))
    reg = cv2.dilate(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return reg > 0


def kp_errors(cam, J, kp, score):
    m = (score >= KP_MIN_SCORE) & np.isfinite(kp[:, 0])
    if m.sum() < MIN_GT_JOINTS:
        return None, None
    uv = cam.project(J[m])
    e = np.linalg.norm(uv - kp[m], axis=1)
    return float(e.mean()), float((e < PCK_PX).mean())


def evaluate_episode(tr: HandTracks, ref, mano: Mano, do_masks: bool = True, do_stereo: bool = True):
    cams = ref.cams
    A = cams["ego_cam_a"]
    n = min(ref.n_frames, tr.n_frames)
    scored = tr.scored_mask().reshape(-1, 2)[:n]
    valid = tr["valid"].reshape(-1, 2)[:n]
    infilled = (tr["source"] == "infilled").reshape(-1, 2)[:n]
    vis = ref.visible[:n]
    J_all = tr["joints_cam"].reshape(-1, 2, 21, 3)[:n].astype(np.float64)
    V_all = tr.get("verts_cam")
    V_all = None if V_all is None else V_all.reshape(-1, 2, 778, 3)
    rows = []
    for i in range(n):
        for s, side in enumerate(SIDES):
            if not vis[i, s]:
                continue
            row = {"frame": i, "side": side, "scored": bool(scored[i, s])}
            rows.append(row)
            if not scored[i, s]:
                continue
            J = J_all[i, s]
            for name, G in ref.joints_gt.items():
                for k, v in joint_errors(J, G[i, s]).items():
                    row[f"{k}_{name}"] = v
            if ref.kp2d is not None:
                for v, view in enumerate(ref.kp_views):
                    e, p = kp_errors(cams[view], J, ref.kp2d[i, s, v], ref.kp2d_score[i, s, v])
                    tag = view[-1]
                    row[f"kp_{tag}"], row[f"pck10_{tag}"] = e, p
            if not (do_masks or do_stereo):
                continue
            if V_all is not None and np.isfinite(V_all[i, s].astype(np.float32)).all():
                V = V_all[i, s].astype(np.float64)
            else:
                rr = 2 * i + s
                V, _ = mano.forward(tr["global_orient"][rr], tr["hand_pose"][rr], tr["betas"][rr], tr["transl"][rr],
                                    side == "R", tr["scale"][rr] if "scale" in tr.arrays else 1.0)
                V = V[0]
            faces = mano.faces_for(side == "R")
            if do_masks:
                ma = ref.mask("ego_cam_a", i, side)
                if ma is not None:
                    uv = V[:, :2] / V[:, 2:3] * np.diag(A.K)[:2] + A.K[:2, 2]
                    sil = silhouette(uv, faces, A.size)
                    row["iou_a_full"] = iou(sil, ma)
                    if ref.kp2d is not None:
                        v = ref.kp_views.index("ego_cam_a")
                        n_ = A.undistort_px(np.nan_to_num(ref.kp2d[i, s, v]))
                        reg = hand_region(n_ * np.diag(A.K)[:2] + A.K[:2, 2], ref.kp2d_score[i, s, v], ma.shape)
                        if reg is not None:
                            row["iou_a"] = iou(sil & reg, ma & reg)
                for tag in ("b", "c"):  # held-out cameras (published px); cam_c masks: build_masks_b --camera c
                    mb = ref.mask(f"ego_cam_{tag}", i, side)
                    if mb is None or mb.sum() < 400:
                        continue
                    B = cams[f"ego_cam_{tag}"]
                    sil = silhouette(B.project(V), faces, B.size)
                    row[f"iou_{tag}_full"] = iou(sil, mb)
                    if ref.kp2d is not None:
                        v = ref.kp_views.index(f"ego_cam_{tag}")
                        reg = hand_region(ref.kp2d[i, s, v], ref.kp2d_score[i, s, v], mb.shape)
                        if reg is not None:
                            row[f"iou_{tag}"] = iou(sil & reg, mb & reg)
            if do_stereo and ref.stereo is not None:
                got = ref.stereo.get(i, side)
                if got is not None:
                    D, M, b = got
                    sc = ref.stereo.scale
                    K = A.K.copy()
                    K[:2] *= sc
                    K[:2, 2] = (A.K[:2, 2] + 0.5) * sc - 0.5 - b[:2]
                    uv = V[:, :2] / V[:, 2:3] * np.diag(K)[:2] + K[:2, 2]
                    R = depth_map(uv, V[:, 2], faces, (D.shape[1], D.shape[0]))
                    g = M & (R > 0) & (D > 0)
                    if g.sum() >= 30:
                        dd = R[g] - D[g]
                        row["stereo_abs"] = 1000 * float(np.median(np.abs(dd)))
                        row["stereo_signed"] = 1000 * float(np.median(dd))
    temporal = temporal_metrics(tr, scored, ref.fps, n)
    cov = {"visible_hand_frames": int(vis.sum()),
           "coverage_observed": float((scored & vis).sum() / max(vis.sum(), 1)),
           "coverage_any": float((valid & vis).sum() / max(vis.sum(), 1)),
           "infilled_frac_of_visible": float((infilled & vis).sum() / max(vis.sum(), 1)),
           "infilled_frac_of_valid": float(infilled[:n].sum() / max(valid.sum(), 1))}
    return rows, {**temporal, **cov}


def temporal_metrics(tr: HandTracks, scored: np.ndarray, fps: float, n: int) -> dict:
    J = tr["joints_cam"].reshape(-1, 2, 21, 3)[:n].astype(np.float64)
    W = tr.get("joints_world")
    if W is not None:  # prefer the world frame when the method gives one (removes head motion)
        J = W.reshape(-1, 2, 21, 3)[:n].astype(np.float64)
    betas = tr["betas"].reshape(-1, 2, 10)[:n]
    acc, fast, steps, bone_std, beta_std = [], 0, 0, [], []
    for s in range(2):
        sc = scored[:, s]
        idx = np.nonzero(sc)[0]
        if len(idx) < 3:
            continue
        c3 = sc[1:-1] & sc[:-2] & sc[2:]
        if c3.any():
            k = np.nonzero(c3)[0] + 1
            a = (J[k + 1, s] - 2 * J[k, s] + J[k - 1, s]) * fps ** 2
            acc.append(np.linalg.norm(a, axis=-1).mean(1))
        c2 = sc[1:] & sc[:-1]
        if c2.any():
            k = np.nonzero(c2)[0]
            v = np.linalg.norm(J[k + 1, s, 0] - J[k, s, 0], axis=-1) * fps
            fast += int((v > 2.0).sum())
            steps += len(k)
        bl = np.stack([np.linalg.norm(J[idx, s, c] - J[idx, s, p], axis=-1) for p, c in BONES], 1)
        bone_std.append(1000 * bl.std(0).mean())
        beta_std.append(betas[idx, s].std(0).mean())
    # handedness swaps: both hands scored at t-1 and t and crossing the labels is 5 cm cheaper than keeping them
    both = scored[1:].all(1) & scored[:-1].all(1)
    k = np.nonzero(both)[0]
    w = J[:, :, 0]
    direct = np.linalg.norm(w[k + 1, 0] - w[k, 0], axis=-1) + np.linalg.norm(w[k + 1, 1] - w[k, 1], axis=-1)
    cross = np.linalg.norm(w[k + 1, 0] - w[k, 1], axis=-1) + np.linalg.norm(w[k + 1, 1] - w[k, 0], axis=-1)
    swaps = int((cross < direct - 0.05).sum())
    minutes = n / fps / 60
    return {"jitter_accel_ms2": float(np.concatenate(acc).mean()) if acc else np.nan,
            "wrist_fast_frac": fast / steps if steps else np.nan,
            "bone_len_std_mm": float(np.mean(bone_std)) if bone_std else np.nan,
            "betas_std": float(np.mean(beta_std)) if beta_std else np.nan,
            "swaps_per_min": swaps / minutes if minutes else np.nan}
