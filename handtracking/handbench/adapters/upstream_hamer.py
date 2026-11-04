"""Adapter: cached upstream ego pipeline hand records (track3_handfix/<ep>/<dir>/<track>/<frame>.json) -> hands.npz.

dirs: hamer_aligned_filled (E0 = the hand stage output: WiLoR det -> SAM2 masks -> HaMeR MANO -> MoGe depth
alignment -> gap interpolation) or hamer_refined_simple (E0-refined: + gsplat joint refinement).
Records with "interpolated": true are written as source=infilled.

Usage: python -m handbench.adapters.upstream_hamer --method E0 --dir hamer_aligned_filled [--episodes ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np

from handbench import format as hf
from handbench import paths
from handbench.adapters.v2d_upstream import UpstreamCamera, anycalib_dir
from handbench.calib import load_rig
from handbench.fit import refit_global
from handbench.mano import Mano
from handbench.refs_io import episode_times, hand_track_ids


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT, text=True).strip()
    except Exception:
        return "unknown"


def convert_episode(ep, src_dir: str, mano: Mano, cams: dict, with_verts: bool = True):
    epd = paths.CACHE / paths.ep_name(ep)
    U = UpstreamCamera(epd, cams["ego_cam_a"], anycalib_dir(ep))
    n = paths.num_frames(ep)
    a = hf.empty(n, episode_times(ep), with_verts=with_verts)
    tracks = hand_track_ids(ep)
    for tid, side in tracks.items():
        d = epd / src_dir / str(tid)
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            rec = json.load(open(f))
            i = int(rec.get("frame_idx", int(f.stem)))
            if i >= n:
                continue
            r = hf.row_index(i, side)
            if a["valid"][r]:  # two tracks of one side on one frame: keep the first (prompt order), like ego_hands.py
                continue
            m = rec["mano"]
            go, hp, be = (np.asarray(m[k], float).ravel() for k in ("global_orient", "hand_pose", "betas"))
            s = float(rec.get("hand_scale", 1.0))
            V, J = mano.forward(go, hp, be, np.asarray(rec["camera"]["cam_t"], float), rec["is_right"], s)
            Vb, Jb = U.points_to_bench(V[0]), U.points_to_bench(J[0])
            go_b, tr_b, s_b = refit_global(mano, hp, be, side == "R", Vb)
            interp = bool(rec.get("interpolated", False))
            dg = rec.get("diagnostics", {})
            npx = int(dg.get("n_pixels", 0))
            a["valid"][r] = True
            a["source"][r] = "infilled" if interp else "observed"
            a["conf"][r] = 0.0 if interp else (1.0 if npx >= 256 else 0.5)
            a["global_orient"][r], a["hand_pose"][r], a["betas"][r] = go_b, hp, be
            a["transl"][r], a["scale"][r] = tr_b, s_b
            a["joints_cam"][r] = Jb
            if with_verts:
                a["verts_cam"][r] = Vb
            if (rec["is_right"]) != (side == "R"):
                print(f"  note {paths.ep_name(ep)} track {tid} frame {i}: record is_right={rec['is_right']} vs track {side}")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dir", default="hamer_aligned_filled")
    ap.add_argument("--set", default="v2d_dev")
    ap.add_argument("--episodes", default="")
    ap.add_argument("--no-verts", action="store_true")
    a = ap.parse_args()
    eps = [paths.ep_name(e) for e in a.episodes.split(",")] if a.episodes else paths.episodes(a.set)
    mano, cams = Mano(), load_rig(paths.rig_json())
    for ep in eps:
        t0 = time.time()
        arr = convert_episode(ep, a.dir, mano, cams, with_verts=not a.no_verts)
        meta = {"method": a.method, "version": "video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer "
                                              "(cached run outputs/track3_handfix)",
                "checkpoint": "HaMeR (v2d_hamer weights) + WiLoR detector", "args": {"source_dir": a.dir},
                "gpu": "n/a (cached run; adapter on CPU)", "wall_time_s": None, "fps": paths.fps(),
                "consumed_stereo_depth": False, "depth_source": "MoGe (monocular), GeoCalib/AnyCalib intrinsics",
                "adapter_commit": git_commit(), "adapter_wall_time_s": time.time() - t0}
        out = paths.RESULTS / a.method / ep / "hands.npz"
        hf.save(out, arr, meta)
        v = arr["valid"]
        print(f"{ep}: {v.sum()} valid rows ({(arr['source'] == 'infilled').sum()} infilled) -> {out}")


if __name__ == "__main__":
    main()
