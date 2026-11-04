"""Adapter: the team's current hands (Rian's objfill_v2s: WiLoR + SAM2 + stereo/MoGe-fused depth alignment, glitch
cleaning, gaps filled by carrying the hand with the object) -> hands.npz.

Source (read only, snapshotted first so later reruns there do not change our results):
  /home/rborah/v2d/ego2sim/ep<N>/hands/objfill_v2s/{left,right}/<frame>.json  ->  data/team_objfill_v2s/ep<N>/...
Records use the upstream layout (mano, camera.cam_t, hand_scale, interpolated, cleaned) and the camera is the undistorted
cam_a with the factory K, i.e. the benchmark camera itself: no camera conversion. `interpolated: true` (gap and object
fills) -> source=infilled (never scored); `cleaned: true` (glitch replaced on a visible frame) -> observed, counted in meta.

Usage: python -m handbench.adapters.team_objfill [--method TEAM-objfill_v2s]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from handbench import format as hf
from handbench import paths
from handbench.adapters.upstream_hamer import git_commit
from handbench.mano import Mano
from handbench.refs_io import episode_times

SRC = Path("/home/rborah/v2d/ego2sim")
SNAP = paths.ROOT / "data" / "team_objfill_v2s"
EPISODES = (2, 11, 13, 18, 21, 23, 26, 40, 41)


def snapshot():
    for e in EPISODES:
        src = SRC / f"ep{e}" / "hands" / "objfill_v2s"
        dst = SNAP / f"ep{e}"
        if not dst.exists() and src.exists():
            shutil.copytree(src, dst)


def convert(e, mano, root=None):
    """Any ego_hands-layout folder (root/{left,right}/<frame>.json, benchmark camera) -> hands.npz arrays."""
    root = Path(root) if root else SNAP / f"ep{e}"
    ep = paths.ep_name(e)
    n = paths.num_frames(ep)
    a = hf.empty(n, episode_times(ep), with_verts=True)
    stats = {"cleaned": 0, "interpolated": 0}
    for side, folder in (("L", "left"), ("R", "right")):
        for f in sorted((root / folder).glob("*.json")):
            rec = json.load(open(f))
            i = int(rec.get("frame_idx", int(f.stem)))
            if i >= n:
                continue
            r = hf.row_index(i, side)
            m = rec["mano"]
            go, hp, be = (np.asarray(m[k], float).ravel() for k in ("global_orient", "hand_pose", "betas"))
            s = float(rec.get("hand_scale", 1.0))
            tr = np.asarray(rec["camera"]["cam_t"], float)
            V, J = mano.forward(go, hp, be, tr, side == "R", s)
            if not (V[0, :, 2] > 0.05).all():
                continue
            interp = bool(rec.get("interpolated", False))
            stats["interpolated"] += interp
            stats["cleaned"] += bool(rec.get("cleaned", False)) and not interp
            a["valid"][r], a["source"][r] = True, "infilled" if interp else "observed"
            a["conf"][r] = 0.0 if interp else 1.0
            a["global_orient"][r], a["hand_pose"][r], a["betas"][r], a["transl"][r], a["scale"][r] = go, hp, be, tr, s
            a["joints_cam"][r], a["verts_cam"][r] = J[0], V[0]
    return a, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="TEAM-objfill_v2s")
    a = ap.parse_args()
    snapshot()
    mano = Mano()
    for e in EPISODES:
        if not (SNAP / f"ep{e}").exists():
            print(f"ep{e}: not available, skipped")
            continue
        arr, stats = convert(e, mano)
        meta = {"method": a.method, "version": "team pipeline objfill_v2s (Rian): WiLoR + SAM2 + depth alignment + "
                "clean_hand_tracks + fill_hand_gaps_object", "checkpoint": "WiLoR", "args": {"source": str(SRC)},
                "gpu": "n/a (team run)", "wall_time_s": None, "fps": paths.fps(), "consumed_stereo_depth": True,
                "adapter_commit": git_commit(), **stats}
        hf.save(paths.RESULTS / a.method / paths.ep_name(e) / "hands.npz", arr, meta)
        print(f"ep{e}: {arr['valid'].sum()} valid rows, {stats}")


if __name__ == "__main__":
    main()
