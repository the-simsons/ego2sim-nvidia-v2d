"""Overlay video for an AssemblyHands / HOT3D clip: one row per method, [input camera] [second camera].

Each method's hand mesh (camera frame, metres) is drawn with the TRUE camera (input K; second camera = rigid transform
+ its K), so 3D errors show as misplacement; ground-truth joints are green dots in both views. AssemblyHands' second
camera is the extra headset camera (held out for E0 / F0; F2 / F3 used it for triangulation). EgoFound3R's
interpolated in-between frames are drawn like predicted ones (not scored).

  ../.venv-bench/bin/python tools/overlay_groundtruth.py --dataset assemblyhands --clip 9012-c07c_005770 \
      --methods E0,F0,F2,F3 [--out results/assemblyhands/videos/<clip>.mp4]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from handbench.mano import Mano  # noqa: E402
from handbench.overlay import COL, draw_mesh  # noqa: E402

SRC = {"E0": ("hands", "data/e0_hands"), "F0": ("egofound3r", "data/egofound3r"),
       "F2": ("egofound3r", "data/egofound3r_f2"), "F3": ("egofound3r", "data/egofound3r_f3"),
       "F0-known-K": ("egofound3r", "data/egofound3r_known_k")}


def load_verts(dataset, clip, method):
    kind, d = SRC[method]
    if kind == "hands":
        e = np.load(ROOT / d / dataset / clip / "hands.npz")
        return e["verts_cam"], e["predicted"] | e["filled"]
    e = np.load(ROOT / d / dataset / clip / "export.npz")
    return e["verts778"], e["valid"]


def project(K, X):
    p = X @ K.T
    return p[:, :2] / p[:, 2:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["assemblyhands", "hot3d"])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--methods", default="E0,F0,F2,F3")
    ap.add_argument("--out", default="")
    ap.add_argument("--height", type=int, default=300, help="row height (px)")
    ap.add_argument("--crf", type=int, default=28)
    a = ap.parse_args()
    methods = a.methods.split(",")
    inp = ROOT / "data" / f"{a.dataset}_input"
    meta = json.loads((inp / f"{a.clip}.json").read_text())
    g = np.load(inp / f"{a.clip}_gt.npz")
    K, fps = np.array(meta["K"]), meta["fps"]
    mano = Mano()
    verts = {m: load_verts(a.dataset, a.clip, m) for m in methods}
    if a.dataset == "assemblyhands":
        T21, K2 = np.median(g["T_cam2_from_cam"], 0), g["K2"]
        src = Path(meta["source"]).parent / f"{meta['second_camera']}_mono10bit"
        frame_b = lambda i: (cv2.imread(str(f)) if (f := src / f"{meta['frames'][i]:06d}.jpg").exists() else None)  # noqa: E731
    else:
        from handbench.datasets import hot3d as H3
        from handbench.experiments import hot3d_check as C
        hc = H3.Clip(H3.ROOT / "train_aria" / f"{a.clip}.tar")
        cams, maps, (_, right), _ = C.rig(hc)
        Bc = cams["ego_cam_b"]
        T21, K2 = Bc.T_from_a, Bc.K
        frame_b = lambda i: cv2.remap(hc.image(i, right), *maps[right], cv2.INTER_LINEAR)  # noqa: E731
    to2 = lambda X: X @ T21[:3, :3].T + T21[:3, 3]  # noqa: E731
    out = Path(a.out or ROOT / "results" / a.dataset / "videos" / f"{a.clip}_{'_'.join(methods)}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(inp / f"{a.clip}.mp4"))
    ff = None
    i = 0
    while True:
        ok, fa = cap.read()
        if not ok:
            break
        fb = frame_b(i)
        if fb is None:
            fb = np.zeros((meta["size"][0], meta["size"][1], 3), np.uint8)
        rows = []
        for m in methods:
            V, valid = verts[m]
            ia, ib = fa.copy(), fb.copy()
            for s, side in enumerate("LR"):
                if i < len(valid) and valid[i, s] and np.isfinite(V[i, s]).all():
                    X = V[i, s].astype(np.float64)
                    faces = mano.faces_for(side == "R")
                    ia = draw_mesh(ia, project(K, X), faces, COL[side])
                    X2 = to2(X)
                    if (X2[:, 2] > 0.02).all():
                        ib = draw_mesh(ib, project(K2, X2), faces, COL[side])
                if g["visible"][i, s]:
                    for q in project(K, g["joints_cam"][i, s].astype(np.float64)):
                        cv2.circle(ia, tuple(int(x) for x in q), 2, (0, 220, 0), -1)
                    G2 = to2(g["joints_cam"][i, s].astype(np.float64))
                    if (G2[:, 2] > 0.02).all():
                        for q in project(K2, G2):
                            cv2.circle(ib, tuple(int(x) for x in q), 2, (0, 220, 0), -1)
            ha = a.height
            ia = cv2.resize(ia, (int(ia.shape[1] * ha / ia.shape[0]), ha), interpolation=cv2.INTER_AREA)
            ib = cv2.resize(ib, (int(ib.shape[1] * ha / ib.shape[0]), ha), interpolation=cv2.INTER_AREA)
            cv2.putText(ia, f"{m} input f{i}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(ib, "second camera, GT = green", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            rows.append(np.hstack([ia, ib]))
        frame = np.vstack(rows)
        frame = frame[: frame.shape[0] - frame.shape[0] % 2, : frame.shape[1] - frame.shape[1] % 2]
        if ff is None:
            ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                   f"{frame.shape[1]}x{frame.shape[0]}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                                   "-crf", str(a.crf), "-preset", "medium", "-pix_fmt", "yuv420p", str(out)],
                                  stdin=subprocess.PIPE)
        ff.stdin.write(np.ascontiguousarray(frame).tobytes())
        i += 1
    ff.stdin.close()
    ff.wait()
    print(f"{out}: {i} frames, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
