"""HaWoR runs on AssemblyHands / HOT3D clips (tools/run_hawor_on_clips.py) -> hands.npz for tools/score_groundtruth.py.

Uses HaWoR's camera-space per-chunk hands (cam_verts) on the frames its motion estimation covers ("observed"; its
infiller's frames are kept as `filled`, never scored) - the single-camera output, no SLAM. Joints = MANO's joint
regressor on HaWoR's posed vertices + the 5 fingertip vertices, in handbench (OpenPose) order. HaWoR's camera is the
input camera with the principal point moved to the image centre, so every point is moved along its pixel ray back to
the input camera's K (pixel and depth kept).

  ../.venv-bench/bin/python tools/hawor_to_groundtruth_hands.py --dataset assemblyhands
Writes data/hawor_hands/<dataset>/<clip>/hands.npz: joints_cam (T,2,21,3), verts_cam (T,2,778,3), predicted, filled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from handbench.mano import JOINT_ORDER, TIP_VERTS, Mano  # noqa: E402


def to_K(X, K_from, K_to):
    z = X[..., 2:3]
    uv = X[..., :2] / z * np.diag(K_from)[:2] + K_from[:2, 2]
    return np.concatenate([(uv - K_to[:2, 2]) / np.diag(K_to)[:2] * z, z], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["assemblyhands", "hot3d"])
    a = ap.parse_args()
    mano = Mano()
    Jreg = np.asarray(mano.J_regressor)
    runs = ROOT / "data" / "hawor" / a.dataset
    for run in sorted(p for p in runs.iterdir() if (p / "export.npz").exists()):
        clip = run.name
        meta = json.loads((ROOT / "data" / f"{a.dataset}_input" / f"{clip}.json").read_text())
        T, K = meta["n_frames"], np.array(meta["K"])
        Kh = np.array(json.loads((run / "input.json").read_text())["K"])
        e = np.load(run / "export.npz")
        n = min(T, e["observed"].shape[1])
        J = np.full((T, 2, 21, 3), np.nan, np.float32)
        V = np.full((T, 2, 778, 3), np.nan, np.float32)
        pred, filled = np.zeros((T, 2), bool), np.zeros((T, 2), bool)
        for h in range(2):  # HaWoR order: 0 = left, 1 = right = handbench sides
            for i in range(n):
                v = e["cam_verts"][h, i].astype(np.float64)
                if not np.isfinite(v).all():
                    continue
                j16 = Jreg @ v
                j = np.concatenate([j16, v[TIP_VERTS]], 0)[JOINT_ORDER]
                J[i, h], V[i, h] = to_K(j, Kh, K), to_K(v, Kh, K)
                pred[i, h] = bool(e["observed"][h, i])
                filled[i, h] = not pred[i, h]
        out = ROOT / "data" / "hawor_hands" / a.dataset / clip
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "hands.npz", joints_cam=J, verts_cam=V, predicted=pred, filled=filled)
        print(f"{clip}: {int(pred.sum())} predicted, {int(filled.sum())} other hand-frames")


if __name__ == "__main__":
    main()
