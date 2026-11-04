"""Adapter: HaWoR raw export (data/hawor/<ep>/export.npz, handbench.runners.hawor_run) -> hands.npz.

  --variant e2e   E2(i)  HaWoR end to end: infilled world-space hands brought to the camera with HaWoR's own SLAM
                         (DROID-SLAM + Metric3D scale). Frames inside HaWoR's detection chunks are `observed`,
                         frames produced by the motion infiller are `infilled` (never scored).
  --variant cam   E2 cam-space: HaWoR's per-chunk camera-frame hands (no SLAM, no infiller); observed frames only.
Then E2(ii) = `cam` + stereo alignment: python -m handbench.experiments.e1 align --mode scale --src E2-hawor-cam ...

HaWoR's camera = benchmark camera (same centre/orientation, factory focal; principal point re-centred in the image
only). Left hands: HaWoR uses the MANO_LEFT model (here the mirrored-right stand-in); our convention is the right model
mirrored, whose params are HaWoR's left params with the y, z axis-angle components negated. Global orientation /
translation / scale are re-fitted to HaWoR's own vertices and the fit residual is recorded in meta.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from handbench import format as hf
from handbench import paths
from handbench.adapters.upstream_hamer import git_commit
from handbench.fit import refit_global
from handbench.mano import Mano
from handbench.refs_io import episode_times

HAWOR_SIDE = {0: "L", 1: "R"}


def to_ours(pose45, is_right):
    p = np.asarray(pose45, float).reshape(15, 3).copy()
    if not is_right:
        p[:, 1:] *= -1
    return p.reshape(45)


def convert(ep, variant, mano):
    d = np.load(paths.RESULTS.parent / "data" / "hawor" / ep / "export.npz")
    meta_raw = json.loads(str(d["meta"]))
    n = paths.num_frames(ep)
    a = hf.empty(n, episode_times(ep), with_verts=True)
    obs = d["observed"]
    resid = []
    for h, side in HAWOR_SIDE.items():
        is_right = side == "R"
        T = min(n, obs.shape[1])
        for i in range(T):
            if variant == "e2e":
                if not d["world_valid_filled"][h, i]:
                    continue
                R, t = d["R_w2c"][i], d["t_w2c"][i]
                V = d["world_verts"][h, i].astype(float) @ R.T + t
                hp, be = d["world_pose"][h, i], d["world_betas"][h, i]
                src = "observed" if obs[h, i] else "infilled"
            else:
                if not obs[h, i] or not np.isfinite(d["cam_verts"][h, i]).all():
                    continue
                V = d["cam_verts"][h, i].astype(float)
                hp, be = d["cam_pose"][h, i], d["cam_betas"][h, i]
                src = "observed"
            if not (V[:, 2] > 0.05).all():
                continue
            hp_o = to_ours(hp, is_right)
            go, tr, s = refit_global(mano, hp_o, be, is_right, V)
            Vf, Jf = mano.forward(go, hp_o, be, tr, is_right, s)
            resid.append(np.abs(Vf[0] - V).max())
            r = hf.row_index(i, side)
            a["valid"][r], a["source"][r] = True, src
            a["conf"][r] = 1.0 if src == "observed" else 0.0
            a["global_orient"][r], a["hand_pose"][r], a["betas"][r], a["transl"][r], a["scale"][r] = go, hp_o, be, tr, s
            a["joints_cam"][r], a["verts_cam"][r] = Jf[0], V
            if variant == "e2e":
                a.setdefault("T_world_cam", np.tile(np.eye(4), (2 * n, 1, 1)))
                Tc = np.eye(4)
                Tc[:3, :3], Tc[:3, 3] = d["R_w2c"][i].T, -d["R_w2c"][i].T @ d["t_w2c"][i]
                a["T_world_cam"][r] = Tc
                a.setdefault("joints_world", np.full((2 * n, 21, 3), np.nan, np.float32))
                a["joints_world"][r] = Jf[0] @ Tc[:3, :3].T + Tc[:3, 3]
    meta = {"method": None, "version": f"HaWoR@{meta_raw['hawor_commit']} ({variant})",
            "checkpoint": meta_raw["checkpoint"], "args": {"variant": variant, "img_focal": meta_raw["img_focal"]},
            "gpu": meta_raw["gpu"], "wall_time_s": meta_raw["wall_time_s"], "fps": paths.fps(),
            "fps_processing": round(meta_raw["fps"], 3), "consumed_stereo_depth": False,
            "mano_left": meta_raw["mano_left"], "adapter_commit": git_commit(),
            "refit_max_vertex_err_mm_median": float(np.median(resid) * 1000) if resid else None}
    return a, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["e2e", "cam"], required=True)
    ap.add_argument("--method", default="")
    ap.add_argument("--episodes", default="")
    a = ap.parse_args()
    name = a.method or {"e2e": "E2-hawor", "cam": "E2-hawor-cam"}[a.variant]
    eps = [paths.ep_name(e) for e in a.episodes.split(",")] if a.episodes else paths.episodes()
    mano = Mano()
    for ep in eps:
        if not (paths.RESULTS.parent / "data" / "hawor" / ep / "export.npz").exists():
            print(f"{ep}: no HaWoR export, skipped")
            continue
        t0 = time.time()
        arr, meta = convert(ep, a.variant, mano)
        meta["method"] = name
        hf.save(paths.RESULTS / name / ep / "hands.npz", arr, meta)
        v = arr["valid"]
        print(f"{ep}: {v.sum()} valid ({(arr['source'] == 'infilled').sum()} infilled), refit "
              f"{meta['refit_max_vertex_err_mm_median']:.2f} mm, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
