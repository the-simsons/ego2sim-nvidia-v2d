"""Overlay video per episode, side by side across methods (one row per method):
  [cam_a undistorted + mesh] [cam_b + projected mesh + cam_b detector keypoints] [depth vs pseudo-GT / stereo plot]

  python -m handbench.overlay --episode 12 --methods E0,E1b-scale [--out results/overlays/ep12.mp4] [--width 1600]

The plot shows, per hand, the method's mean joint depth minus the triangulated pseudo-GT (solid) and minus SGBM
stereo at the hand (dotted), with a cursor at the current frame. H.264 via ffmpeg, CRF chosen to stay < 30 MB.
"""
from __future__ import annotations

import argparse
import subprocess

import cv2
import numpy as np

from handbench import format as hf
from handbench import paths
from handbench.calib import load_rig
from handbench.mano import Mano
from handbench.refs_io import VideoReader

COL = {"L": (40, 140, 255), "R": (60, 60, 230)}  # BGR: L orange, R red


def draw_mesh(img, uv, faces, color, alpha=0.45):
    ov = img.copy()
    ok = np.isfinite(uv).all(1) & (np.abs(uv) < 1e5).all(1)
    f = faces[ok[faces].all(1)]
    if len(f):
        cv2.fillPoly(ov, list(np.round(uv[f] * 4).astype(np.int32)), color, lineType=cv2.LINE_AA, shift=2)
    return cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)


def plot_panel(w, h, series, cursor, n, title):
    """series: list of (values (n,), color, dashed). y range +-150 mm."""
    p = np.full((h, w, 3), 250, np.uint8)
    y0 = h // 2
    cv2.line(p, (0, y0), (w, y0), (180, 180, 180), 1)
    for mm in (-100, -50, 50, 100):
        y = int(y0 - mm / 150 * (h / 2 - 12))
        cv2.line(p, (0, y), (w, y), (230, 230, 230), 1)
        cv2.putText(p, f"{mm:+d}", (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    for vals, color, dashed in series:
        pts = [(int(i / max(n - 1, 1) * (w - 1)), int(y0 - np.clip(v, -150, 150) / 150 * (h / 2 - 12)))
               for i, v in enumerate(vals) if np.isfinite(v)]
        for k in range(1, len(pts)):
            if pts[k][0] - pts[k - 1][0] > 3 * w / n + 2 or (dashed and k % 2):
                continue
            cv2.line(p, pts[k - 1], pts[k], color, 2 if not dashed else 1, cv2.LINE_AA)
    x = int(cursor / max(n - 1, 1) * (w - 1))
    cv2.line(p, (x, 0), (x, h), (0, 0, 0), 1)
    cv2.putText(p, title, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    return p


def depth_series(tr, ep, n):
    """Per side: mean joint depth - mean pseudo-GT depth on the joints with a reference (mm); and vs stereo."""
    tri = np.load(paths.REFS / paths.ep_name(ep) / "tri.npz")["tri_abc"]
    J = tr["joints_cam"].reshape(-1, 2, 21, 3)[:n].astype(float)
    sc = tr.scored_mask().reshape(-1, 2)[:n]
    out = {}
    for s, side in enumerate("LR"):
        d = np.full(n, np.nan)
        for i in range(n):
            g = np.isfinite(tri[i, s, :, 0])
            if sc[i, s] and g.sum() >= 5:
                d[i] = 1000 * (J[i, s, g, 2] - tri[i, s, g, 2]).mean()
        out[side] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--start", type=int, default=0, help="first frame to render")
    ap.add_argument("--end", type=int, default=-1, help="last frame to render (inclusive; -1 = end of episode)")
    ap.add_argument("--labels", default="", help="row labels shown instead of the method names (comma-separated)")
    ap.add_argument("--fps", type=float, default=0, help="output frame rate (default: the video's); lower = slow motion")
    a = ap.parse_args()
    ep = paths.ep_name(a.episode)
    methods = a.methods.split(",")
    out = a.out or str(paths.RESULTS / "overlays" / f"{ep}_{'_vs_'.join(methods)}.mp4")
    cams = load_rig(paths.rig_json())
    A, B = cams["ego_cam_a"], cams["ego_cam_b"]
    mano = Mano()
    n = paths.num_frames(ep)
    trs = {m: hf.load(paths.RESULTS / m / ep / "hands.npz") for m in methods}
    series = {m: depth_series(trs[m], ep, n) for m in methods}
    kp = np.load(paths.REFS / ep / "kp2d.npz")
    mapa = cv2.initUndistortRectifyMap(A.K, A.dist, np.eye(3), A.K, A.size, cv2.CV_32FC1)
    # panel sizes: cam_a 4:3, cam_b 16:10, plot
    ha = int(a.width * 0.36 * 3 / 4)
    wa, wb = int(ha * 4 / 3), int(ha * 16 / 10)
    wp = a.width - wa - wb
    Wt, Ht = wa + wb + wp, ha * len(methods)
    Wt, Ht = Wt - Wt % 2, Ht - Ht % 2
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                           f"{Wt}x{Ht}", "-r", f"{(a.fps or paths.fps()):.4f}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", str(a.crf), "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    labels = dict(zip(methods, a.labels.split(","))) if a.labels else {}
    for i, fr in VideoReader(ep, ("ego_cam_a", "ego_cam_b")):
        if i < a.start:
            continue
        if i >= n or (a.end >= 0 and i > a.end):
            break
        ua = cv2.remap(fr["ego_cam_a"], *mapa, cv2.INTER_LINEAR)
        rows = []
        for m in methods:
            tr = trs[m]
            ia, ib = ua.copy(), fr["ego_cam_b"].copy()
            for s, side in enumerate("LR"):
                r = 2 * i + s
                if not tr["valid"][r]:
                    continue
                V, _ = mano.forward(tr["global_orient"][r], tr["hand_pose"][r], tr["betas"][r], tr["transl"][r],
                                    side == "R", tr["scale"][r])
                V = V[0]
                faces = mano.faces_for(side == "R")
                col = COL[side] if tr["source"][r] == "observed" else (160, 160, 160)
                ia = draw_mesh(ia, V[:, :2] / V[:, 2:3] * np.diag(A.K)[:2] + A.K[:2, 2], faces, col)
                ib = draw_mesh(ib, B.project(V), faces, col)
                k, sc = kp["kp"][i, s, 1], kp["score"][i, s, 1]
                for (x, y), c in zip(k, sc):
                    if c >= 0.5 and np.isfinite(x):
                        cv2.circle(ib, (int(x), int(y)), 3, (0, 220, 0), -1)
            ia = cv2.resize(ia, (wa, ha), interpolation=cv2.INTER_AREA)
            ib = cv2.resize(ib, (wb, ha), interpolation=cv2.INTER_AREA)
            cv2.putText(ia, f"{labels.get(m, m)}  cam_a  f{i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(ib, "cam_b (held out) + detector kps", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
            sr = series[m]
            pp = plot_panel(wp, ha, [(sr["L"], COL["L"], False), (sr["R"], COL["R"], False)], i, n,
                            "depth - tri pseudo-GT (mm, clipped +-150), L orange / R red")
            rows.append(np.hstack([ia, ib, pp]))
        frame = np.vstack(rows)[:Ht, :Wt]
        ff.stdin.write(np.ascontiguousarray(frame).tobytes())
    ff.stdin.close()
    ff.wait()
    import os

    mb = os.path.getsize(out) / 1e6
    print(f"{out}: {mb:.1f} MB")
    if mb > 30 and a.crf < 40:
        print("over 30 MB, re-encoding with higher CRF")
        subprocess.run(["python", "-m", "handbench.overlay", "--episode", a.episode, "--methods", a.methods, "--out", out,
                        "--width", str(a.width), "--crf", str(a.crf + 4), "--start", str(a.start), "--end", str(a.end),
                        "--labels", a.labels, "--fps", str(a.fps)], cwd=paths.ROOT)


if __name__ == "__main__":
    main()
