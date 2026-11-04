"""Set A references (built once, cached in refs/<episode>/):

  kp2d.npz   2D hand keypoints per (frame, side, view a/b/c), detected INDEPENDENTLY in each view:
             RTMPose-m hand (top-down, per-joint scores) on a perspective crop of that view's own image, plus
             MediaPipe HandLandmarker on the same crop as a cross-check. Crop placement: cam_a SAM2 hand mask
             (cam_a) / the mask's rays over a 0.25-0.9 m depth band transferred through the rig (cam_b, cam_c).
             Keypoints are stored in published (distorted) image pixels of their view.
  masks_a    cam_a SAM2 hand-mask area per (frame, side) in the benchmark camera (visible-frame rule, 1.4).
  tri.npz    triangulated pseudo-GT joints in the cam_a frame:
               tri_ab   cam_a + cam_b only (spec A.3)
               tri_abc  every confident view among a, b, c (cam_a always included)
             a joint is kept if its score >= --min-score in every used view and its reprojection error is < 3 px
             in every used view.

Usage: python -m handbench.build_refs [--episodes 12] [--jobs 8]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from handbench import paths
from handbench.adapters.v2d_upstream import UpstreamCamera, anycalib_dir
from handbench.calib import load_rig
from handbench.crops import PerspectiveCrop
from handbench.refs_io import VideoReader, hand_track_ids

VIEWS = ("ego_cam_a", "ego_cam_b", "ego_cam_c")
SIDES = ("L", "R")
DEPTH_BAND = (0.25, 0.4, 0.6, 0.9)
MIN_MASK_PX = 400


def _single_thread_ort():
    """rtmlib builds its own onnxruntime session sized to every core (130+ spinning threads per worker on this
    shared machine). Force one intra-op thread per session."""
    import onnxruntime as ort

    if getattr(ort.InferenceSession, "_handbench_patched", False):
        return
    base = ort.InferenceSession

    class OneThread(base):
        _handbench_patched = True

        def __init__(self, *args, **kw):
            so = kw.pop("sess_options", None) or (args[1] if len(args) > 1 else None) or ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            path = kw.pop("path_or_bytes", None) or args[0]
            super().__init__(path, so, *args[2:], **kw)

    ort.InferenceSession = OneThread


def _models():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _single_thread_ort()
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from rtmlib import RTMPose

    onnx = glob.glob(str(paths.ROOT / "assets/models/rtmpose/**/end2end.onnx"), recursive=True)[0]
    rtm = RTMPose(onnx, model_input_size=(256, 256), backend="onnxruntime", device="cpu")
    opt = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(paths.ROOT / "assets/models/hand_landmarker.task")),
        num_hands=1, min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2,
        running_mode=vision.RunningMode.IMAGE)
    mpd = vision.HandLandmarker.create_from_options(opt)
    return rtm, mpd, mp


def _detect(rtm, crop, frame, grey: bool):
    img = crop.sample(frame)
    if grey:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if (img.sum(2) == 0).mean() > 0.6:  # crop mostly outside the image
        return None
    k, s = rtm(img, bboxes=[[0, 0, 255, 255]])
    return k[0], s[0], img


def side_masks(ep, U, tracks, i):
    """Union of the cached SAM2 masks per side, warped into the benchmark camera."""
    out = {}
    for tid, side in tracks.items():
        m = cv2.imread(str(paths.CACHE / paths.ep_name(ep) / "masks" / str(tid) / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        mb = U.warp_to_bench(m) > 0
        out[side] = mb if side not in out else (out[side] | mb)
    return out


def keypoints_episode(ep):
    t0 = time.time()
    rtm, mpd, mp = _models()
    cams = load_rig(paths.rig_json())
    A = cams["ego_cam_a"]
    U = UpstreamCamera(paths.CACHE / paths.ep_name(ep), A, anycalib_dir(ep))
    tracks = hand_track_ids(ep)
    n = paths.num_frames(ep)
    kp = np.full((n, 2, 3, 21, 2), np.nan, np.float32)
    sc = np.zeros((n, 2, 3, 21), np.float32)
    kp_mp = np.full((n, 2, 3, 21, 2), np.nan, np.float32)
    area = np.zeros((n, 2), np.int32)
    bbox_a = np.zeros((n, 2, 4), np.float32)
    for i, fr in VideoReader(ep, VIEWS):
        if i >= n:
            break
        masks = side_masks(ep, U, tracks, i)
        for s, side in enumerate(SIDES):
            mb = masks.get(side)
            if mb is None:
                continue
            area[i, s] = int(mb.sum())
            if area[i, s] < MIN_MASK_PX:
                continue
            ys, xs = np.nonzero(mb)
            bbox_a[i, s] = (xs.min(), ys.min(), xs.max(), ys.max())
            pts = np.c_[xs, ys][:: max(1, len(xs) // 300)].astype(float)
            rays_a = np.c_[(pts - A.K[:2, 2]) / np.diag(A.K)[:2], np.ones(len(pts))]
            for v, cn in enumerate(VIEWS):
                c = cams[cn]
                rays = rays_a if cn == "ego_cam_a" else c.to_cam(np.concatenate([rays_a * z for z in DEPTH_BAND]))
                if (rays[:, 2] <= 0.05).any():
                    continue
                # pass 1: crop on the SAM2 mask (which includes the forearm); pass 2: re-crop tightly on the
                # confident pass-1 keypoints so the hand fills the 256 px crop
                crop = PerspectiveCrop.around(c, rays, 256, 1.3)
                got = _detect(rtm, crop, fr[cn], cn != "ego_cam_a")
                if got is None:
                    continue
                k, s_, img = got
                good = s_ >= 0.3
                if good.sum() >= 5:
                    n2 = crop.crop_px_to_norm(k[good])
                    crop2 = PerspectiveCrop.around(c, np.c_[n2, np.ones(len(n2))], 256, 1.6)
                    got2 = _detect(rtm, crop2, fr[cn], cn != "ego_cam_a")
                    if got2 is not None and np.median(got2[1]) >= np.median(s_) - 0.05:
                        crop, (k, s_, img) = crop2, got2
                kp[i, s, v] = crop.crop_px_to_img(k)
                sc[i, s, v] = s_
                r = mpd.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                if r.hand_landmarks:
                    q = np.array([[l.x * 256, l.y * 256] for l in r.hand_landmarks[0]])
                    kp_mp[i, s, v] = crop.crop_px_to_img(q)
    out = paths.REFS / paths.ep_name(ep)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "kp2d.npz", kp=kp, score=sc, kp_mp=kp_mp, mask_area_a=area, bbox_a=bbox_a,
                        views=np.array(VIEWS), sides=np.array(SIDES),
                        meta=np.array(json.dumps({"detector": "rtmpose-m_simcc-hand5 256x256 (rtmlib, onnxruntime CPU)",
                                                  "crosscheck": "mediapipe HandLandmarker float16 v1",
                                                  "crop": "perspective crop, margin 1.3, 256 px",
                                                  "wall_time_s": time.time() - t0})))
    return paths.ep_name(ep), time.time() - t0


GATE_MIN_INSIDE = 0.6
GATE_DILATE = 11


def _inside_frac(mask, px, score):
    g = (score >= 0.5) & np.isfinite(px[:, 0])
    if g.sum() < 5 or mask is None:
        return np.nan
    m = cv2.dilate(mask.astype(np.uint8), np.ones((GATE_DILATE, GATE_DILATE), np.uint8)) > 0
    H, W = m.shape
    q = np.round(px[g]).astype(int)
    q[:, 0], q[:, 1] = q[:, 0].clip(0, W - 1), q[:, 1].clip(0, H - 1)
    return float(m[q[:, 1], q[:, 0]].mean())


def gate_episode(ep):
    """Mask-consistency gate (keypoint <-> hand identity). A view's keypoints for a hand are kept only if >= 60 % of
    its confident keypoints fall inside THAT hand's mask in THAT view (dilated 5 px); otherwise the view's keypoints
    for that hand-frame are dropped, and for cam_b the mask too (one of the two is on the wrong hand; we cannot tell
    which). cam_a uses the cam_a SAM2 masks (benchmark px), cam_b the cam_b SAM2 masks (build_masks_b). cam_c has no
    mask: its keypoints must agree with cam_a in the triangulation (reprojection gate).
    refs/<ep>/kp_gate.npz: keep (n,2,3) bool, inside (n,2,2) fraction for a,b, mask_b_ok (n,2)."""
    cams = load_rig(paths.rig_json())
    A = cams["ego_cam_a"]
    U = UpstreamCamera(paths.CACHE / paths.ep_name(ep), A, anycalib_dir(ep))
    tracks = hand_track_ids(ep)
    d = np.load(paths.REFS / paths.ep_name(ep) / "kp2d.npz")
    kp, sc = d["kp"], d["score"]
    n = kp.shape[0]
    keep = np.ones((n, 2, 3), bool)
    inside = np.full((n, 2, 2), np.nan)
    mask_b_ok = np.zeros((n, 2), bool)
    mbd = paths.REFS / paths.ep_name(ep) / "masks_b"
    for i in range(n):
        if not (d["mask_area_a"][i] >= MIN_MASK_PX).any():
            continue
        ma = side_masks(ep, U, tracks, i)
        for s, side in enumerate(SIDES):
            if d["mask_area_a"][i, s] < MIN_MASK_PX:
                continue
            n_ = A.undistort_px(np.nan_to_num(kp[i, s, 0]))
            fa = _inside_frac(ma.get(side), n_ * np.diag(A.K)[:2] + A.K[:2, 2], sc[i, s, 0])
            mb = cv2.imread(str(mbd / side / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
            fb = _inside_frac(None if mb is None else mb > 0, kp[i, s, 1], sc[i, s, 1])
            inside[i, s] = fa, fb
            keep[i, s, 0] = not (fa < GATE_MIN_INSIDE)  # NaN (too few confident kps) -> kept, scores decide
            keep[i, s, 1] = not (fb < GATE_MIN_INSIDE)
            mask_b_ok[i, s] = mb is not None and not (fb < GATE_MIN_INSIDE)
    np.savez_compressed(paths.REFS / paths.ep_name(ep) / "kp_gate.npz", keep=keep, inside=inside, mask_b_ok=mask_b_ok,
                        meta=np.array(json.dumps({"min_inside": GATE_MIN_INSIDE, "dilate_px": GATE_DILATE // 2})))
    vis = d["mask_area_a"] >= MIN_MASK_PX
    return {"dropped_a": int((~keep[..., 0] & vis).sum()), "dropped_b": int((~keep[..., 1] & vis).sum()),
            "visible": int(vis.sum())}


def gated_scores(ep):
    """kp2d scores with gated-out views set to 0 (used by triangulation and by the dataset)."""
    d = np.load(paths.REFS / paths.ep_name(ep) / "kp2d.npz")
    sc = d["score"].copy()
    gp = paths.REFS / paths.ep_name(ep) / "kp_gate.npz"
    if gp.exists():
        sc *= np.load(gp)["keep"][..., None]
    return sc


def reproj_px(cam, X, uv):
    return np.linalg.norm(cam.project(X) - uv, axis=-1)


def triangulate_kp(cl, kp, sc, min_score=0.5, max_reproj=3.0):
    """Joint-wise triangulation of kp (n,2,3,21,2) published px with scores sc (n,2,3,21) for cameras cl (a, b, c);
    vectorised over (frame, side, joint). Returns {tri_ab, tri_abc} + their _views and _err_px."""
    kp = kp.astype(np.float64)
    n = kp.shape[0]
    nrm = np.full(kp.shape, np.nan)
    for v, c in enumerate(cl):
        m = np.isfinite(kp[:, :, v, :, 0])
        nrm[:, :, v][m] = c.undistort_px(kp[:, :, v][m])
    res = {}
    for name, allowed in (("tri_ab", (0, 1)), ("tri_abc", (0, 1, 2))):
        X = np.full((n, 2, 21, 3), np.nan)
        used = np.zeros((n, 2, 21, 3), bool)
        err = np.full((n, 2, 21), np.nan)
        ok_v = np.stack([(sc[:, :, v] >= min_score) & np.isfinite(kp[:, :, v, :, 0]) if v in allowed
                         else np.zeros(sc.shape[:2] + (21,), bool) for v in range(3)], -1)  # (n,2,21,3)
        # cam_a must be confident and at least one other view
        cand = ok_v[..., 0] & (ok_v[..., 1:].sum(-1) >= 1)
        for _ in range(2):  # second pass drops the worst view if a 3-view joint fails the gate
            idx = np.argwhere(cand)
            if not len(idx):
                break
            rows = []
            for v in range(3):
                P = cl[v].T_from_a[:3]
                nv = nrm[idx[:, 0], idx[:, 1], v, idx[:, 2]]
                w = ok_v[idx[:, 0], idx[:, 1], idx[:, 2], v].astype(float)[:, None]
                nv = np.nan_to_num(nv)
                rows.append(w * (nv[:, :1] * P[2] - P[0]))
                rows.append(w * (nv[:, 1:] * P[2] - P[1]))
            Amat = np.stack(rows, 1)
            _, _, Vt = np.linalg.svd(Amat)
            Xh = Vt[:, -1]
            Xi = Xh[:, :3] / Xh[:, 3:4]
            e = np.zeros((len(idx), 3))
            for v in range(3):
                uv = kp[idx[:, 0], idx[:, 1], v, idx[:, 2]]
                e[:, v] = np.where(ok_v[idx[:, 0], idx[:, 1], idx[:, 2], v], reproj_px(cl[v], Xi, np.nan_to_num(uv)), 0)
            good = (e.max(1) < max_reproj) & (Xi[:, 2] > 0.1) & (Xi[:, 2] < 1.5)
            g = idx[good]
            X[g[:, 0], g[:, 1], g[:, 2]] = Xi[good]
            used[g[:, 0], g[:, 1], g[:, 2]] = ok_v[g[:, 0], g[:, 1], g[:, 2]]
            err[g[:, 0], g[:, 1], g[:, 2]] = e[good].max(1)
            # retry failures with their worst non-a view removed (only helps 3-view joints)
            bad = idx[~good]
            if not len(bad):
                break
            eb = e[~good]
            worst = 1 + np.argmax(eb[:, 1:], 1)
            ok_v[bad[:, 0], bad[:, 1], bad[:, 2], worst] = False
            cand = np.zeros_like(cand)
            retry = ok_v[bad[:, 0], bad[:, 1], bad[:, 2], 0] & (ok_v[bad[:, 0], bad[:, 1], bad[:, 2], 1:].sum(-1) >= 1)
            cand[bad[retry, 0], bad[retry, 1], bad[retry, 2]] = True
        res[name] = X.astype(np.float32)
        res[name + "_views"] = used
        res[name + "_err_px"] = err.astype(np.float32)
    return res


def triangulate_episode(ep, min_score=0.5, max_reproj=3.0):
    """Joint-wise triangulation from kp2d.npz (gated scores) -> refs/<ep>/tri.npz."""
    cams = load_rig(paths.rig_json())
    d = np.load(paths.REFS / paths.ep_name(ep) / "kp2d.npz")
    res = triangulate_kp([cams[v] for v in VIEWS], d["kp"], gated_scores(ep), min_score, max_reproj)
    np.savez_compressed(paths.REFS / paths.ep_name(ep) / "tri.npz", **res,
                        meta=np.array(json.dumps({"min_score": min_score, "max_reproj_px": max_reproj})))
    return {k: int(np.isfinite(v[..., 0]).sum()) for k, v in res.items() if k in ("tri_ab", "tri_abc")}


STEREO_SCALE = 0.5
STEREO_PAD = 0.15


def stereo_episode(ep, backend: str = "sgbm"):
    """SGBM depth (benchmark camera, half resolution) cropped to each visible hand's cam_a mask box + 15 % padding,
    with the half-res mask. refs/<ep>/stereo.npz: per (frame, side) a crop + its box [x0,y0,x1,y1] at half-res."""
    from handbench.stereo import FoundationStereoDepth, StereoDepth

    t0 = time.time()
    cams = load_rig(paths.rig_json())
    sd = (FoundationStereoDepth if backend == "fs" else StereoDepth)(cams, out_scale=STEREO_SCALE)
    U = UpstreamCamera(paths.CACHE / paths.ep_name(ep), cams["ego_cam_a"], anycalib_dir(ep))
    tracks = hand_track_ids(ep)
    n = paths.num_frames(ep)
    crops, masks, boxes = {}, {}, np.full((n, 2, 4), -1, np.int32)
    for i, fr in VideoReader(ep, ("ego_cam_b", "ego_cam_c")):
        if i >= n:
            break
        ms = {}
        for tid, side in tracks.items():
            m = cv2.imread(str(paths.CACHE / paths.ep_name(ep) / "masks" / str(tid) / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            mb = U.warp_to_bench(m, scale=STEREO_SCALE) > 0
            ms[side] = mb if side not in ms else ms[side] | mb
        ms = {s: m for s, m in ms.items() if m.sum() * (1 / STEREO_SCALE) ** 2 >= MIN_MASK_PX}
        if not ms:
            continue
        D = sd.compute(fr["ego_cam_c"], fr["ego_cam_b"])
        H, W = D.shape
        for side, mb in ms.items():
            s = SIDES.index(side)
            ys, xs = np.nonzero(mb)
            px, py = STEREO_PAD * (xs.max() - xs.min() + 1), STEREO_PAD * (ys.max() - ys.min() + 1)
            b = (int(max(0, xs.min() - px)), int(max(0, ys.min() - py)), int(min(W - 1, xs.max() + px)),
                 int(min(H - 1, ys.max() + py)))
            boxes[i, s] = b
            crops[f"d_{i}_{side}"] = D[b[1]:b[3] + 1, b[0]:b[2] + 1].astype(np.float16)
            masks[f"m_{i}_{side}"] = np.packbits(mb[b[1]:b[3] + 1, b[0]:b[2] + 1])
    name = "stereo.npz" if backend == "sgbm" else f"stereo_{backend}.npz"
    np.savez_compressed(paths.REFS / paths.ep_name(ep) / name, boxes=boxes, **crops, **masks,
                        meta=np.array(json.dumps({"backend": backend, "scale": STEREO_SCALE, "pad": STEREO_PAD, "f_rect": sd.f,
                                                  "baseline_m": sd.baseline, "uses_views": ["ego_cam_b", "ego_cam_c"]})))
    return paths.ep_name(ep), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="v2d_dev")
    ap.add_argument("--episodes", default="")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--skip-kp", action="store_true")
    ap.add_argument("--stereo", action="store_true", help="only build refs/<ep>/stereo.npz")
    ap.add_argument("--backend", default="sgbm", choices=["sgbm", "fs"],
                    help="stereo backend; fs (FoundationStereo, GPU) writes refs/<ep>/stereo_fs.npz")
    ap.add_argument("--gate", action="store_true", help="mask-consistency gate (needs masks_b), then re-triangulate")
    ap.add_argument("--min-score", type=float, default=0.5)
    a = ap.parse_args()
    eps = [paths.ep_name(e) for e in a.episodes.split(",")] if a.episodes else paths.episodes(a.set)
    if a.gate:
        with ProcessPoolExecutor(a.jobs) as ex:
            for e, r in zip(eps, ex.map(gate_episode, eps)):
                print(e, r, flush=True)
        a.skip_kp = True
    elif a.stereo:
        (paths.REFS / eps[0]).mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(a.jobs) as ex:
            for name, dt in ex.map(stereo_episode, eps, [a.backend] * len(eps)):
                print(f"{name}: stereo crops in {dt:.0f}s", flush=True)
        return
    if not a.skip_kp:
        todo = [e for e in eps if not (paths.REFS / e / "kp2d.npz").exists()]
        with ProcessPoolExecutor(a.jobs) as ex:
            for name, dt in ex.map(keypoints_episode, todo):
                print(f"{name}: keypoints in {dt:.0f}s", flush=True)
    for e in eps:
        print(e, triangulate_episode(e, a.min_score), flush=True)


if __name__ == "__main__":
    main()
