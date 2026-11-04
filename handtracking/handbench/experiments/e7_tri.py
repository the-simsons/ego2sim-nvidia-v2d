"""E7: Track 3 port of the HOT3D winner (hot3d_fusion "tri-interp+sgbm+savgol"): move hands onto triangulated RTMPose
joints, carry the move across frames without joints, SGBM stereo where nothing else applies, Savitzky-Golay.

Held-out discipline: on Track 3 the a+b+c triangulated joints ARE the benchmark reference, so this method triangulates
from cam_a + cam_c ONLY (3.8 cm apart, well conditioned on our rig) and never uses cam_b's keypoints or masks. cam_b
keypoints / SAM2 masks therefore stay an independent judge; the tri_abc / tri_ab 3D errors share the cam_a (+cam_c)
detections with this method and are flagged as non-independent. The SGBM fallback (cam_c + cam_b pixels) is used only
on stretches without any triangulated joint; its share is reported.

Per hand, over runs of consecutive observed frames of the base method:
  shift     frames with >= MIN_JOINTS joints triangulated from a+c: translate by the median joint offset (rejected if the
            median residual after the shift exceeds MAX_RESIDUAL_M: wrong-hand matches)
  carry     offsets cleaned against a 9-frame running median, interpolated between frames that have one, held HOLD_S
            seconds past the first / last
  sgbm      stretches still uncovered: one stereo scale about the camera per stretch (median of the frames whose rendered
            silhouette has >= MIN_STEREO_FRAC of its pixels with stereo depth), if within SCALE_RANGE
  savgol    Savitzky-Golay (w7, o2) on joints and verts, as E1best

Usage: python -m handbench.experiments.e7_tri --src E2-hawor-cam --name E7tri-hawor [--jobs 14]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.signal import savgol_filter

from handbench import format as hf
from handbench import paths
from handbench.build_refs import VIEWS, gated_scores, triangulate_kp
from handbench.calib import load_rig
from handbench.datasets.v2d import StereoCrops
from handbench.experiments.e1 import _render_crop, commit, set_row
from handbench.hand_correction import (HOLD_S, MAX_RESIDUAL_M, MIN_JOINTS, MIN_STEREO_FRAC, SCALE_RANGE,
                                       carry_offsets, runs_of, shift_to_keypoints)
from handbench.mano import Mano

SIDES = ("L", "R")


def stereo_scale(V, faces, A, st, i, side):
    got = st.get(i, side)
    if got is None:
        return np.nan, 0.0
    D, M, box = got
    R, _, _ = _render_crop(V, faces, A, st.scale, box, D.shape)
    m = cv2.erode((R > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    m &= M  # Track 3 has SAM2 hand masks: use mask & silhouette, as E1b
    g = m & (D > 0)
    if m.sum() < 30 or g.sum() < 30:
        return np.nan, g.sum() / max(m.sum(), 1)
    return float(np.median(D[g] / R[g])), g.sum() / m.sum()


def run_episode(args):
    src, name, ep = args
    t0 = time.time()
    mano, cams = Mano(), load_rig(paths.rig_json())
    A = cams["ego_cam_a"]
    cl = [cams[v] for v in VIEWS]
    base = hf.load(paths.RESULTS / src / ep / "hands.npz")
    a = {k: v.copy() for k, v in base.arrays.items()}
    if "verts_cam" not in a:
        a["verts_cam"] = np.full((len(a["frame_idx"]), 778, 3), np.nan, np.float32)
    a["verts_cam"] = a["verts_cam"].astype(np.float64)
    d = np.load(paths.REFS / ep / "kp2d.npz")
    sc = gated_scores(ep).copy()
    sc[:, :, VIEWS.index("ego_cam_b")] = 0  # held out: cam_b keypoints are never used
    tri = triangulate_kp(cl, d["kp"], sc)
    T, U = tri["tri_abc"].astype(float), tri["tri_abc_views"]  # with cam_b zeroed this is a+c only
    usable = np.isfinite(T[..., 0]) & U[..., VIEWS.index("ego_cam_c")]
    st = StereoCrops(paths.REFS / ep / "stereo.npz")
    n = len(a["frame_idx"]) // 2
    hold = int(round(HOLD_S * paths.fps()))
    info = {}
    for s, side in enumerate(SIDES):
        rows = np.arange(s, 2 * n, 2)
        obs = a["valid"][rows] & (a["source"][rows] == "observed")
        faces = mano.faces_for(side == "R")
        J0 = a["joints_cam"][rows].astype(float)
        V0 = np.full((n, 778, 3), np.nan)
        for i in np.nonzero(obs)[0]:
            r = rows[i]
            v, _ = mano.forward(a["global_orient"][r], a["hand_pose"][r], a["betas"][r], a["transl"][r], side == "R",
                                a["scale"][r])
            V0[i] = v[0]
        off = shift_to_keypoints(J0, T[:n, s], usable[:n, s], obs)
        oi = carry_offsets(off, obs, hold)
        cov = np.isfinite(oi[:, 0])
        J, V = J0.copy(), V0.copy()
        J[cov] += oi[cov, None]
        V[cov] += oi[cov, None]
        sg = np.zeros(n, bool)
        for run in runs_of(obs & ~cov):
            ss = []
            for i in run:
                k, frac = stereo_scale(V0[i], faces, A, st, i, side)
                if frac >= MIN_STEREO_FRAC and np.isfinite(k):
                    ss.append(k)
            if ss and SCALE_RANGE[0] < np.median(ss) < SCALE_RANGE[1]:
                J[run] = J0[run] * np.median(ss)
                V[run] = V0[run] * np.median(ss)
                sg[run] = True
        for run in runs_of(obs):
            if len(run) >= 7:
                J[run] = savgol_filter(J[run], 7, 2, axis=0, mode="interp")
                V[run] = savgol_filter(V[run], 7, 2, axis=0, mode="interp")
        for i in np.nonzero(obs)[0]:
            set_row(a, rows[i], V[i], J[i], mano)
        k = max(int(obs.sum()), 1)
        info[side] = {"observed": int(obs.sum()), "tri_frames": float(np.isfinite(off[:, 0])[obs].sum() / k),
                      "carried_frames": float(cov[obs].sum() / k), "sgbm_frames": float(sg[obs].sum() / k),
                      "untouched_frames": float((obs & ~cov & ~sg).sum() / k)}
    a["verts_cam"] = a["verts_cam"].astype(np.float16)
    meta = dict(base.meta)
    meta.update(method=name, version=f"{base.meta.get('version')} + E7 tri(a+c)-shift + carry + sgbm + savgol",
                args={"src": src, "min_joints": MIN_JOINTS, "max_res_m": MAX_RESIDUAL_M, "hold_s": HOLD_S,
                      "min_stereo": MIN_STEREO_FRAC, "scale_range": SCALE_RANGE},
                adapter_commit=commit(), info=info, consumed_stereo_depth=True,
                uses_cam_a_keypoints=True, uses_cam_c_keypoints=True, uses_cam_b_keypoints=False,
                derived_wall_time_s=time.time() - t0)
    hf.save(paths.RESULTS / name / ep / "hands.npz", a, meta)
    return ep, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", default="")
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    eps = [paths.ep_name(e) for e in a.episodes.split(",")] if a.episodes else paths.episodes()
    eps = [e for e in eps if (paths.RESULTS / a.src / e / "hands.npz").exists()]
    tot = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for ep, info in ex.map(run_episode, [(a.src, a.name, e) for e in eps]):
            print(ep, json.dumps({s: {k: round(v, 2) for k, v in d.items()} for s, d in info.items()}), flush=True)
            for s, d in info.items():
                for k, v in d.items():
                    tot.setdefault(k, []).append(v)
    print("mean over episode-sides:", {k: round(float(np.mean(v)), 3) for k, v in tot.items()})


if __name__ == "__main__":
    main()
