"""Standalone E0 runs on AssemblyHands / HOT3D -> hands.npz for tools/score_groundtruth.py (--source hands).

Per hand track (hand_tracks.json, handedness reconciled), each hamer_aligned_filled record is posed with handbench's
MANO (record betas / pose / cam_t, scaled by its hand_scale), as handbench.adapters.upstream_hamer does on Track 3.
The records live in E0's camera (MoGe's stabilised intrinsics, K_stable); like that adapter's points_to_bench, each
point is moved along its pixel ray to the input camera's true K (pixel and depth kept). Interpolated records (gap
fill) are kept but not `predicted` (never scored). Two tracks of one side on one frame: the first (prompt order).

  ../.venv-bench/bin/python tools/e0_standalone_to_groundtruth_hands.py --dataset hot3d
Writes data/e0_hands/<dataset>/<clip>/hands.npz: joints_cam (T,2,21,3), verts_cam (T,2,778,3), predicted (T,2),
  filled (T,2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from handbench.mano import Mano  # noqa: E402


def _K(d):
    return np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1.0]])


def to_true_K(X, K_from, K_to):
    z = X[..., 2:3]
    uv = X[..., :2] / z * np.diag(K_from)[:2] + K_from[:2, 2]
    return np.concatenate([(uv - K_to[:2, 2]) / np.diag(K_to)[:2] * z, z], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["assemblyhands", "hot3d"])
    a = ap.parse_args()
    mano = Mano()
    runs = ROOT / "data" / "e0_standalone" / a.dataset
    for run in sorted(p for p in runs.iterdir() if (p / "hamer_aligned_filled").exists()):
        clip = run.name
        meta = json.loads((ROOT / "data" / f"{a.dataset}_input" / f"{clip}.json").read_text())
        T, K = meta["n_frames"], np.array(meta["K"])
        Ks = _K(json.loads((run / "intrinsics_stable.json").read_text()))
        J = np.full((T, 2, 21, 3), np.nan, np.float32)
        V = np.full((T, 2, 778, 3), np.nan, np.float32)
        pred, filled = np.zeros((T, 2), bool), np.zeros((T, 2), bool)
        for tr in json.loads((run / "hand_tracks.json").read_text())["tracks"]:
            s = int(bool(tr["is_right"]))
            for f in sorted((run / "hamer_aligned_filled" / str(tr["object_id"])).glob("*.json")):
                rec = json.loads(f.read_text())
                i = int(rec.get("frame_idx", int(f.stem)))
                if i >= T or pred[i, s] or filled[i, s]:
                    continue
                m = rec["mano"]
                go, hp, be = (np.asarray(m[k], float).ravel() for k in ("global_orient", "hand_pose", "betas"))
                v, j = mano.forward(go, hp, be, np.asarray(rec["camera"]["cam_t"], float), s == 1,
                                    float(rec.get("hand_scale", 1.0)))
                J[i, s], V[i, s] = to_true_K(j[0], Ks, K), to_true_K(v[0], Ks, K)
                interp = bool(rec.get("interpolated", False))
                pred[i, s], filled[i, s] = not interp, interp
        out = ROOT / "data" / "e0_hands" / a.dataset / clip
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "hands.npz", joints_cam=J, verts_cam=V, predicted=pred, filled=filled)
        print(f"{clip}: {int(pred.sum())} predicted, {int(filled.sum())} gap-filled hand-frames")


if __name__ == "__main__":
    main()
