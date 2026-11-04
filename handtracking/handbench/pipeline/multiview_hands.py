"""Multi-view hand tracking for the Ego2Sim pipeline: a drop-in replacement for tools/ego_hands.py.

Single-camera hands (HaWoR) corrected with the other cameras: RTMPose keypoints triangulated from cam_a + cam_c, and
SGBM stereo depth where no keypoint correction reaches.

Writes the SAME per-frame JSON layout as ego_hands.py (OUT/{left,right}/<frame>.json: mano {betas, global_orient,
hand_pose}, camera {cam_t, intrinsics}, hand_scale, interpolated), in the undistorted cam_a camera of
tools/ego_stereo_depth.py (factory K), so tools/clean_hand_tracks.py, tools/fill_hand_gaps_object.py and
tools/build_ego_bundle.py consume it unchanged.

Steps (benchmarked as experiment "E7tri-hawor", see handtracking/results/REPORT_TABLES_team9.md):
  1 HaWoR on raw cam_a (GPU, methods/.venv-hawor)                         -> MANO hands per frame
  2 RTMPose hand keypoints in cam_a and cam_c, on crops placed by the SAM2 hand masks (cam_b is not used)
  3 triangulate cam_a + cam_c keypoints; translate each hand onto them (frames with >= 4 joints), carry the shift
    across frames without joints (up to 1 s), SGBM stereo scale ($S/stereo_depth) where nothing else applies
  4 per-frame HaMeR fallback (handbench.hand_fallback, benchmarked as "E8fallback-e1best-smooth"): upstream HaMeR on
    the SAM2 hand masks (pipeline/hamer_pass.py), its depth corrected like HaWoR's (keypoint shift, else stereo scale
    per frame; uncorrected HaMeR hands are never used). Frames where the HaWoR hand is implausible in cam_a (joints
    outside its SAM2 mask or far from the cam_a keypoints, e.g. HaWoR on another person's hand) or missing take the
    HaMeR hand when that one is plausible; the jump at each switch is spread over a few frames. Off with
    --no-hamer-fallback, or automatically when no HaMeR environment is found.
  5 Savitzky-Golay smoothing over the spliced track; short gaps (<= --max-gap frames) linearly filled and marked
    interpolated
Left/right of each SAM2 hand track is decided by overlap with HaWoR's left / right hands. Each written frame records
how it was corrected ("correction": keypoints / carried / stereo / none / fallback / filled).

Usage (bench venv, handtracking/.venv-bench):
  python -m handbench.pipeline.multiview_hands --video-dir VIDEO_DIR --episode EP --stereo-dir S --hand-masks E/hand_masks \
      --out E/hands/hands_by_side [--gpu 0]
VIDEO_DIR holds observation.images.ego_cam_{a,c}/episode_<EP>.mp4 (Track 3 layout).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

HB = Path(__file__).resolve().parents[2]  # handtracking/
sys.path.insert(0, str(HB))

from handbench.adapters.hawor import to_ours  # noqa: E402
from handbench.build_refs import DEPTH_BAND, GATE_MIN_INSIDE, _detect, _inside_frac, _models, triangulate_kp  # noqa: E402
from handbench.calib import load_rig  # noqa: E402
from handbench.crops import PerspectiveCrop  # noqa: E402
from handbench.hand_correction import (HOLD_S, MIN_STEREO_FRAC, SCALE_RANGE, carry_offsets, runs_of,  # noqa: E402
                                       shift_to_keypoints)
from handbench.fit import refit_global  # noqa: E402
from handbench.hand_fallback import better, choose_fallback, is_plausible, plausibility, seam_offsets  # noqa: E402
from handbench.pipeline import hamer_pass  # noqa: E402
from handbench.mano import Mano  # noqa: E402
from handbench.render import depth_map  # noqa: E402

SIDES = ((0, "left", False), (1, "right", True))  # HaWoR index 0 = left, 1 = right
MIN_MASK_PX = 400


def video(vdir, cam, ep):
    return Path(vdir) / f"observation.images.{cam}" / f"episode_{int(ep):06d}.mp4"


def decode_depth(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    raw = raw.astype(np.float32)
    d = 65535.0 / np.maximum(raw, 1.0) - 1.0  # v2d inverse-depth encoding (ego_stereo_depth.encode_depth)
    d[(raw >= 65535) | (raw <= 0)] = 0
    return d


def run_hawor(a, raw_a):
    out = Path(a.work) / "hawor"
    if (out / "hawor" / "export.npz").exists():
        return out / "hawor" / "export.npz"
    py = HB / "methods" / ".venv-hawor" / "bin" / "python"
    hw = HB / "methods" / "HaWoR"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu),
               PYTHONPATH=f"{hw}:{hw}/thirdparty/DROID-SLAM:{hw}/thirdparty/DROID-SLAM/droid_slam:{HB}")
    cmd = [str(py), "-m", "handbench.pipeline.hawor_stage", "--video", str(raw_a), "--out", str(out), "--rig", a.rig]
    print("[multiview] HaWoR:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=HB)
    return out / "hawor" / "export.npz"


def hawor_hands(ex, mano, n):
    """-> {side: (J (n,21,3), V (n,778,3), betas (n,10), hand_pose (n,45))} camera frame, NaN where HaWoR has no hand."""
    out = {}
    for h, side, is_right in SIDES:
        J = np.full((n, 21, 3), np.nan)
        V = np.full((n, 778, 3), np.nan)
        B = np.zeros((n, 10))
        P = np.zeros((n, 45))
        m = min(n, ex["observed"].shape[1])
        for i in range(m):
            if not ex["observed"][h, i] or not np.isfinite(ex["cam_verts"][h, i, 0, 0]):
                continue
            v = ex["cam_verts"][h, i].astype(float)
            if not (v[:, 2] > 0.05).all():
                continue
            hp = to_ours(ex["cam_pose"][h, i], is_right)
            go, tr, s = refit_global(mano, hp, ex["cam_betas"][h, i], is_right, v)
            Vf, Jf = mano.forward(go, hp, ex["cam_betas"][h, i], tr, is_right, s)
            V[i], J[i], B[i], P[i] = Vf[0], Jf[0], ex["cam_betas"][h, i], hp
        out[side] = (J, V, B, P)
    return out


def assign_tracks(mask_dir, hands, A, mano, n):
    """SAM2 hand track id -> 'left' / 'right' by mask overlap with HaWoR's projected hands (sampled frames)."""
    from handbench.render import silhouette

    tracks = {}
    for td in sorted(p for p in Path(mask_dir).iterdir() if p.is_dir()):
        votes = {"left": 0.0, "right": 0.0}
        files = sorted(td.glob("*.png"))
        for mp in files[:: max(1, len(files) // 40)]:
            i = int(mp.stem)
            if i >= n:
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None or (m > 0).sum() < MIN_MASK_PX:
                continue
            m = m > 0
            for _, side, is_right in SIDES:
                V = hands[side][1][i]
                if not np.isfinite(V[0, 0]):
                    continue
                uv = V[:, :2] / V[:, 2:3] * np.diag(A.K)[:2] + A.K[:2, 2]
                sil = silhouette(uv, mano.faces_for(is_right), (m.shape[1], m.shape[0]))
                votes[side] += (sil & m).sum() / max(sil.sum(), 1)
        if max(votes.values()) > 0:
            tracks[td.name] = max(votes, key=votes.get)
        print(f"[multiview] mask track {td.name}: overlap votes {({k: round(v, 1) for k, v in votes.items()})} -> "
              f"{tracks.get(td.name, 'unassigned')}", flush=True)
    return tracks


def side_mask(mask_dir, tracks, side, i):
    m = None
    for t, s in tracks.items():
        if s != side:
            continue
        x = cv2.imread(str(Path(mask_dir) / t / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if x is not None:
            m = (x > 0) if m is None else (m | (x > 0))
    return m


def keypoints(a, cams, tracks, n, hands):
    """RTMPose keypoints in cam_a and cam_c (view b left empty), gated by the cam_a masks -> kp, sc (n,2,3,21[,2]).

    Crops are placed on HaWoR's projected hand (hand-only, as in the HOT3D study), not on the SAM2 masks: pipeline hand
    masks can include the whole forearm and the same hand is often tracked several times (re-prompted at keyframes), so
    their union makes crops far too large. The masks are used for the crop only where HaWoR has no hand."""
    rtm, _, _ = _models()
    A, C = cams["ego_cam_a"], cams["ego_cam_c"]
    kp = np.full((n, 2, 3, 21, 2), np.nan, np.float32)
    sc = np.zeros((n, 2, 3, 21), np.float32)
    caps = {c: cv2.VideoCapture(str(video(a.video_dir, c, a.episode))) for c in ("ego_cam_a", "ego_cam_c")}
    for i in range(n):
        fr = {}
        for c, cap in caps.items():
            ok, im = cap.read()
            if not ok:
                return kp, sc
            fr[c] = im
        for s, side, _ in SIDES:
            mb = side_mask(a.hand_masks, tracks, side, i)
            if mb is None or mb.sum() < MIN_MASK_PX:
                continue
            Vh = hands[side][1][i]
            if np.isfinite(Vh[0, 0]):  # HaWoR's hand vertices -> rays (hand only)
                rays_a = Vh[::4] / Vh[::4, 2:3]
            else:
                ys, xs = np.nonzero(mb)
                pts = np.c_[xs, ys][:: max(1, len(xs) // 300)].astype(float)
                rays_a = np.c_[(pts - A.K[:2, 2]) / np.diag(A.K)[:2], np.ones(len(pts))]
            for v, cn in ((0, "ego_cam_a"), (2, "ego_cam_c")):
                c = cams[cn]
                rays = rays_a if cn == "ego_cam_a" else c.to_cam(np.concatenate([rays_a * z for z in DEPTH_BAND]))
                if (rays[:, 2] <= 0.05).any():
                    continue
                crop = PerspectiveCrop.around(c, rays, 256, 1.3)
                got = _detect(rtm, crop, fr[cn], cn != "ego_cam_a")
                if got is None:
                    continue
                k, s_, _ = got
                good = s_ >= 0.3
                if good.sum() >= 5:  # pass 2: re-crop tightly on the confident pass-1 keypoints
                    n2 = crop.crop_px_to_norm(k[good])
                    crop2 = PerspectiveCrop.around(c, np.c_[n2, np.ones(len(n2))], 256, 1.6)
                    got2 = _detect(rtm, crop2, fr[cn], cn != "ego_cam_a")
                    if got2 is not None and np.median(got2[1]) >= np.median(s_) - 0.05:
                        crop, (k, s_, _) = crop2, got2
                kp[i, s, v] = crop.crop_px_to_img(k)
                sc[i, s, v] = s_
            # mask-consistency gate on cam_a (keypoints on the other hand are dropped)
            na = A.undistort_px(np.nan_to_num(kp[i, s, 0]))
            f = _inside_frac(mb, na * np.diag(A.K)[:2] + A.K[:2, 2], sc[i, s, 0])
            if f < GATE_MIN_INSIDE:
                sc[i, s, 0] = 0
        if i % 50 == 0:
            print(f"[multiview] keypoints frame {i}/{n}", flush=True)
    return kp, sc


def stereo_scale(V, faces, A, D, M):
    K = A.K.copy()
    K[:2] *= 0.5
    K[:2, 2] = (A.K[:2, 2] + 0.5) * 0.5 - 0.5
    Dh = cv2.resize(D, (D.shape[1] // 2, D.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    Mh = cv2.resize(M.astype(np.uint8), (Dh.shape[1], Dh.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    uv = V[:, :2] / V[:, 2:3] * np.diag(K)[:2] + K[:2, 2]
    R = depth_map(uv, V[:, 2], faces, (Dh.shape[1], Dh.shape[0]))
    m = (cv2.erode((R > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & Mh
    g = m & (Dh > 0)
    if m.sum() < 30 or g.sum() < 30:
        return np.nan, g.sum() / max(m.sum(), 1)
    return float(np.median(Dh[g] / R[g])), g.sum() / m.sum()


def fill_gaps(J, V, P, B, obs, max_gap):
    """Linear fill (vertices / joints / finger pose) of gaps <= max_gap frames between observed frames."""
    filled = np.zeros(len(obs), bool)
    idx = np.nonzero(obs)[0]
    for a_, b_ in zip(idx[:-1], idx[1:]):
        if 1 < b_ - a_ <= max_gap + 1:
            for i in range(a_ + 1, b_):
                w = (i - a_) / (b_ - a_)
                J[i] = (1 - w) * J[a_] + w * J[b_]
                V[i] = (1 - w) * V[a_] + w * V[b_]
                P[i] = (1 - w) * P[a_] + w * P[b_]
                B[i] = B[a_]
                filled[i] = True
    return filled


def fallback(hm, J, V, obs, T, usable, kp_a, kp_sc, side, faces, a, tracks, A, depth, fill_missing):
    """One side. hm = HaMeR (J, V, betas, pose), not metric; J, V = the corrected HaWoR hand (before smoothing).
    HaMeR's depth is corrected like HaWoR's: keypoint shift where >= 4 joints are triangulated, else a per-frame stereo
    scale about the camera; frames with neither are not usable. Returns (use (n,) bool, info incl. "hands")."""
    JH, VH, BH, PH = (x.copy() for x in hm)
    has = np.isfinite(JH[:, 0, 0])
    off = shift_to_keypoints(JH, T, usable, has)
    ok_depth = np.isfinite(off[:, 0])
    JH[ok_depth] += off[ok_depth, None]
    VH[ok_depth] += off[ok_depth, None]
    for i in np.nonzero(has & ~ok_depth)[0]:
        D, M = depth(i), side_mask(a.hand_masks, tracks, side, i)
        if D is None or M is None:
            continue
        k, frac = stereo_scale(VH[i], faces, A, D, M)
        if frac >= MIN_STEREO_FRAC and np.isfinite(k) and SCALE_RANGE[0] < k < SCALE_RANGE[1]:
            JH[i] *= k
            VH[i] *= k
            ok_depth[i] = True
    has_f = has & ok_depth
    n = len(obs)
    hit_p, kp_p, hit_f, kp_f = (np.full(n, np.nan) for _ in range(4))
    for i in np.nonzero(obs | has_f)[0]:
        M = side_mask(a.hand_masks, tracks, side, i)
        for X, hit, kpx, have in ((J, hit_p, kp_p, obs), (JH, hit_f, kp_f, has_f)):
            if have[i]:
                uv = X[i, :, :2] / X[i, :, 2:3] * np.diag(A.K)[:2] + A.K[:2, 2]
                hit[i], kpx[i] = plausibility(uv, M, kp_a[i], kp_sc[i])
    ok_p, ok_f = is_plausible(hit_p, kp_p), is_plausible(hit_f, kp_f)
    use = choose_fallback(obs, ok_p, has_f, ok_f, better(hit_p, kp_p, hit_f, kp_f), fill_missing)
    info = {"hamer_frames": int(has.sum()), "hamer_metric_frames": int(has_f.sum()),
            "hawor_implausible": int((obs & ~ok_p).sum()), "fallback_replaced": int((use & obs).sum()),
            "fallback_added": int((use & ~obs).sum()), "hands": (JH, VH, BH, PH)}
    return use, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--episode", required=True, type=int)
    ap.add_argument("--stereo-dir", required=True, help="tools/ego_stereo_depth.py output (intrinsics_undist.json, "
                                                         "stereo_depth/)")
    ap.add_argument("--hand-masks", required=True, help="SAM2 hand masks: <track_id>/<frame>.png, undistorted cam_a")
    ap.add_argument("--out", required=True, help="writes OUT/left, OUT/right (ego_hands.py hands_by_side layout)")
    ap.add_argument("--work", default="", help="intermediate files (default OUT/../multiview_work)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-gap", type=int, default=15)
    ap.add_argument("--rig", default=str(HB / "refs" / "calib" / "rig.json"))
    ap.add_argument("--no-hamer-fallback", action="store_true", help="HaWoR hands only (no per-frame HaMeR fallback)")
    ap.add_argument("--fallback-replace-only", action="store_true",
                    help="use HaMeR only where HaWoR's hand is implausible, not where HaWoR has no hand")
    ap.add_argument("--hamer-python", default="", help="python of an env with video_to_data's v2d.hamer.lib "
                                                       "(default: $HAMER_PYTHON or the first found, see hamer_pass.py)")
    ap.add_argument("--hamer-weights", default="", help="HaMeR weights folder with _DATA/ (default: $HAMER_WEIGHTS ...)")
    a = ap.parse_args()
    t0 = time.time()
    a.work = a.work or str(Path(a.out).parent / "multiview_work")
    Path(a.work).mkdir(parents=True, exist_ok=True)
    cams = load_rig(a.rig)
    A = cams["ego_cam_a"]
    intr = json.load(open(Path(a.stereo_dir) / "intrinsics_undist.json"))
    if abs(intr["fx"] - A.K[0, 0]) > 0.5 or abs(intr["cx"] - A.K[0, 2]) > 0.5:
        raise SystemExit(f"cam_a intrinsics differ from the rig: {intr} vs K {A.K.tolist()}")
    raw_a = video(a.video_dir, "ego_cam_a", a.episode)
    n = int(cv2.VideoCapture(str(raw_a)).get(cv2.CAP_PROP_FRAME_COUNT))
    mano = Mano()

    ex = np.load(run_hawor(a, raw_a))
    hands = hawor_hands(ex, mano, n)
    tracks = assign_tracks(a.hand_masks, hands, A, mano, n)
    kp_path = Path(a.work) / "keypoints_ac.npz"
    if kp_path.exists():
        z = np.load(kp_path)
        kp, sc = z["kp"], z["score"]
    else:
        kp, sc = keypoints(a, cams, tracks, n, hands)
        np.savez_compressed(kp_path, kp=kp, score=sc, tracks=json.dumps(tracks))
    tri = triangulate_kp([cams[c] for c in ("ego_cam_a", "ego_cam_b", "ego_cam_c")], kp, sc)
    T, U = tri["tri_abc"].astype(float), tri["tri_abc_views"]  # cam_b has no keypoints: a+c triangulation
    usable = np.isfinite(T[..., 0]) & U[..., 2]
    hold = int(round(HOLD_S * float(cv2.VideoCapture(str(raw_a)).get(cv2.CAP_PROP_FPS) or 20.0)))
    info = {"tracks": tracks}
    hamer = None
    if not a.no_hamer_fallback:
        hdir = hamer_pass.run(Path(a.work), raw_a, A, tracks, n, a.hand_masks, a.gpu, a.hamer_python, a.hamer_weights)
        if hdir is not None:
            hamer = hamer_pass.hands(hdir, tracks, mano, A.K, n, a.hand_masks)
    info["hamer_fallback"] = hamer is not None
    # cam_a keypoints in the undistorted cam_a pixels the hands are projected into (for the fallback's checks)
    kp_a = np.stack([A.undistort_px(np.nan_to_num(kp[:, s, 0]).reshape(-1, 2)).reshape(n, 21, 2) for s in range(2)], 1)
    kp_a = kp_a * np.diag(A.K)[:2] + A.K[:2, 2]
    stereo_cache = {}

    def depth(i):
        if i not in stereo_cache:
            stereo_cache[i] = decode_depth(Path(a.stereo_dir) / "stereo_depth" / f"{i:06d}.png")
        return stereo_cache[i]
    out = Path(a.out)
    for s, side, is_right in SIDES:
        J0, V0, B, P = hands[side]
        obs = np.isfinite(J0[:, 0, 0])
        faces = mano.faces_for(is_right)
        off = shift_to_keypoints(J0, T[:, s], usable[:, s], obs)
        oi = carry_offsets(off, obs, hold)
        cov = np.isfinite(oi[:, 0])
        J, V = J0.copy(), V0.copy()
        J[cov] += oi[cov, None]
        V[cov] += oi[cov, None]
        step = np.where(np.isfinite(off[:, 0]), "keypoints", np.where(cov, "carried", "none")).astype(object)
        for run in runs_of(obs & ~cov):
            ss = []
            for i in run:
                D = depth(i)
                M = side_mask(a.hand_masks, tracks, side, i)
                if D is None or M is None:
                    continue
                k, frac = stereo_scale(V0[i], faces, A, D, M)
                if frac >= MIN_STEREO_FRAC and np.isfinite(k):
                    ss.append(k)
            if ss and SCALE_RANGE[0] < np.median(ss) < SCALE_RANGE[1]:
                J[run] = J0[run] * np.median(ss)
                V[run] = V0[run] * np.median(ss)
                step[run] = "stereo"
        B, P = B.copy(), P.copy()
        tracked, fb = obs.copy(), {}
        if hamer is not None:
            use, fb = fallback(hamer[side], J, V, obs, T[:, s], usable[:, s], kp_a[:, s], sc[:, s, 0], side, faces,
                               a, tracks, A, depth, not a.fallback_replace_only)
            JH, VH, BH, PH = fb.pop("hands")
            J[use], V[use], B[use], P[use] = JH[use], VH[use], BH[use], PH[use]
            step[use] = "fallback"
            tracked = obs | use
            d = seam_offsets(np.nan_to_num(J.mean(1)), tracked, use)
            J += d[:, None]
            V += d[:, None]
        for run in runs_of(tracked):
            if len(run) >= 7:
                J[run] = savgol_filter(J[run], 7, 2, axis=0, mode="interp")
                V[run] = savgol_filter(V[run], 7, 2, axis=0, mode="interp")
        filled = fill_gaps(J, V, P, B, tracked, a.max_gap)
        d = out / side
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.json"):
            f.unlink()
        written = 0
        for i in np.nonzero(tracked | filled)[0]:
            go, tr, sc_ = refit_global(mano, P[i], B[i], is_right, V[i])
            uv = V[i, :, :2] / V[i, :, 2:3] * np.diag(A.K)[:2] + A.K[:2, 2]
            rec = {"track_id": s, "is_right": is_right, "frame_idx": int(i), "image_size": list(A.size),
                   "bbox": {"x0": float(uv[:, 0].min()), "y0": float(uv[:, 1].min()),
                            "x1": float(uv[:, 0].max()), "y1": float(uv[:, 1].max())},
                   "mano": {"betas": B[i].tolist(), "global_orient": go.tolist(), "hand_pose": P[i].reshape(15, 3).tolist()},
                   "camera": {"cam_t": tr.tolist(), "intrinsics": {k: intr[k] for k in ("fx", "fy", "cx", "cy")}},
                   "hand_scale": float(sc_), "interpolated": bool(filled[i]),
                   "source": "multiview_hands", "correction": "filled" if filled[i] else str(step[i])}
            json.dump(rec, open(d / f"{i:06d}.json", "w"))
            written += 1
        k_ = max(int(tracked.sum()), 1)
        info[side] = {"frames_written": written, "hawor_frames": int(obs.sum()), "filled_frames": int(filled.sum()),
                      **{f"{st}_frac": float((step[tracked] == st).sum() / k_)
                         for st in ("keypoints", "carried", "stereo", "none", "fallback")}, **fb}
        print(f"[multiview] {side}: {json.dumps(info[side])}", flush=True)
    info["wall_time_s"] = time.time() - t0
    json.dump(info, open(out / "multiview_hands_info.json", "w"), indent=1)
    print(f"[multiview] done in {info['wall_time_s']:.0f}s -> {out}/left, {out}/right")


if __name__ == "__main__":
    main()
