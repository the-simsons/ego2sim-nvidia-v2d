"""HOT3D ground-truth hands (motion-capture MANO) in the pinhole benchmark camera, for the clips in data/hot3d_input.

Uses the earlier HOT3D check's code unchanged (handbench.experiments.hot3d_check.gt_in_P: HOT3D's own MANO layer, world
-> RGB camera -> upright pinhole P) and its visibility rule: a hand is scored when every GT joint is > 0.1 m in front
of the camera and the wrist projects inside the image. Joint order = handbench (OpenPose). Needs torch + smplx, so it
runs in handtracking/methods/.venv-hawor.

  ../methods/.venv-hawor/bin/python tools/export_hot3d_gt.py
Writes data/hot3d_input/<clip>_gt.npz: joints_cam (T,2,21,3) m, side 0 = L, 1 = R; visible (T,2); valid (T,2,21).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from handbench.datasets import hot3d as H  # noqa: E402
from handbench.experiments.hot3d_check import SIDES, gt_in_P  # noqa: E402

IN = ROOT / "data" / "hot3d_input"


def main():
    mano = H.Hot3dMano()
    K = H.pinhole_K()
    for meta in sorted(p for p in IN.glob("clip-*.json") if not p.stem.endswith("_rootK")):
        name = meta.stem
        clip = H.Clip(H.ROOT / "train_aria" / f"{name}.tar")
        n = json.loads(meta.read_text())["n_frames"]
        J = np.zeros((n, 2, 21, 3), np.float32)
        vis = np.zeros((n, 2), bool)
        for h, (side, is_right) in SIDES.items():
            for i in range(n):
                g, _ = gt_in_P(clip, mano, i, side, is_right)
                if g is None or not (g[:, 2] > 0.1).all():
                    continue
                w = g[0, :2] / g[0, 2] * K[0, 0] + K[:2, 2]
                if (0 <= w).all() and (w < H.SIZE_PIN).all():
                    J[i, h], vis[i, h] = g, True
        np.savez_compressed(IN / f"{name}_gt.npz", joints_cam=J, visible=vis,
                            valid=np.repeat(vis[..., None], 21, -1))
        print(f"{name}: {int(vis.sum())} visible hand-frames")


if __name__ == "__main__":
    main()
