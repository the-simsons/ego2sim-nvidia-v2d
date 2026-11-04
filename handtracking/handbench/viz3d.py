"""Three-panel video: original cam_a | cam_a + method's hand meshes | the hands alone in 3D (virtual third-person view).

  python -m handbench.viz3d --episode 12 --method E1best-sgbm [--out results/overlays/ep12_E1best-sgbm_3d.mp4]

The 3D panel is in the benchmark (head-mounted cam_a) frame: a fixed virtual camera beside and above the head looks at
the median hand position; flat-shaded MANO meshes (painter's algorithm, CPU only), a 1 s wrist trail per hand, a 5 cm
grid just below the lowest hand and the head camera drawn as a small frustum. Grey = infilled rows (not scored).
"""
from __future__ import annotations

import argparse
import os
import subprocess

import cv2
import numpy as np

from handbench import format as hf
from handbench import paths
from handbench.calib import load_rig
from handbench.mano import Mano
from handbench.overlay import COL, draw_mesh
from handbench.refs_io import VideoReader


def look_at(eye, target, up=np.array([0.0, -1.0, 0.0])):
    """World(=cam_a frame) -> virtual camera rotation R and translation t (OpenCV convention)."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


class View3D:
    def __init__(self, target, size=(640, 480), f=620.0, offset=(0.42, -0.38, -0.12), up=(0.0, -1.0, 0.0)):
        self.W, self.H = size
        self.K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
        self.R, self.t = look_at(np.asarray(target) + np.asarray(offset), np.asarray(target), np.asarray(up, float))
        self.light = self.R.T @ np.array([0.3, -0.5, -0.8])  # light from the viewer's upper left, in world coords
        self.light /= np.linalg.norm(self.light)

    def proj(self, X):
        Xc = np.asarray(X) @ self.R.T + self.t
        uv = Xc[..., :2] / Xc[..., 2:3] * self.K[0, 0] + self.K[:2, 2]
        return uv, Xc[..., 2]

    def mesh(self, img, V, faces, color, depth_sorted=True):
        uv, z = self.proj(V)
        tri = V[faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = 0.35 + 0.65 * np.abs(n @ self.light)
        zf = z[faces].mean(1)
        order = np.argsort(-zf) if depth_sorted else np.arange(len(faces))
        pts = np.round(uv[faces] * 4).astype(np.int32)
        col = np.asarray(color, float)
        for k in order:
            if zf[k] <= 0.01:
                continue
            cv2.fillConvexPoly(img, pts[k], tuple(int(c) for c in col * shade[k]), lineType=cv2.LINE_AA, shift=2)

    def polyline(self, img, P, color, thick=2):
        uv, z = self.proj(P)
        ok = z > 0.01
        for a, b in zip(uv[:-1][ok[:-1] & ok[1:]], uv[1:][ok[:-1] & ok[1:]]):
            cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, thick, cv2.LINE_AA)

    def grid(self, img, y, x_rng, z_rng, step=0.05):
        for x in np.arange(x_rng[0], x_rng[1] + 1e-9, step):
            self.polyline(img, np.array([[x, y, z_rng[0]], [x, y, z_rng[1]]]), (205, 205, 205), 1)
        for z in np.arange(z_rng[0], z_rng[1] + 1e-9, step):
            self.polyline(img, np.array([[x_rng[0], y, z], [x_rng[1], y, z]]), (205, 205, 205), 1)

    def head(self, img, s=0.05):
        c = np.zeros(3)
        corners = np.array([[-s, -0.75 * s, 1.5 * s], [s, -0.75 * s, 1.5 * s], [s, 0.75 * s, 1.5 * s], [-s, 0.75 * s, 1.5 * s]])
        for q in corners:
            self.polyline(img, np.array([c, q]), (90, 90, 90), 2)
        self.polyline(img, np.vstack([corners, corners[:1]]), (90, 90, 90), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--method", default="E1best-sgbm")
    ap.add_argument("--out", default="")
    ap.add_argument("--panel-h", type=int, default=480)
    ap.add_argument("--crf", type=int, default=28)
    a = ap.parse_args()
    ep = paths.ep_name(a.episode)
    out = a.out or str(paths.RESULTS / "overlays" / f"{ep}_{a.method}_3d.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cams = load_rig(paths.rig_json())
    A = cams["ego_cam_a"]
    mano = Mano()
    tr = hf.load(paths.RESULTS / a.method / ep / "hands.npz")
    n = tr.n_frames
    fps = paths.fps()
    # precompute meshes
    V = np.full((n, 2, 778, 3), np.nan)
    for r in np.nonzero(tr["valid"])[0]:
        i, s = r // 2, r % 2
        v, _ = mano.forward(tr["global_orient"][r], tr["hand_pose"][r], tr["betas"][r], tr["transl"][r],
                            s == 1, tr["scale"][r])
        V[i, s] = v[0]
    wrist = tr["joints_cam"].reshape(-1, 2, 21, 3)[:, :, 0].astype(float)
    wrist[~tr["valid"].reshape(-1, 2)] = np.nan
    allv = V[np.isfinite(V[..., 0])]
    target = np.nanmedian(wrist.reshape(-1, 3), 0)
    target[2] = max(target[2], 0.25)
    view = View3D(target, (int(a.panel_h * 4 / 3), a.panel_h))
    y_grid = np.nanpercentile(allv[:, 1], 99) + 0.03
    x_rng = (np.nanpercentile(allv[:, 0], 1) - 0.1, np.nanpercentile(allv[:, 0], 99) + 0.1)
    z_rng = (0.0, np.nanpercentile(allv[:, 2], 99) + 0.1)
    mapa = cv2.initUndistortRectifyMap(A.K, A.dist, np.eye(3), A.K, A.size, cv2.CV_32FC1)
    ph, pw = a.panel_h, int(a.panel_h * 4 / 3)
    Wt, Ht = 3 * pw, ph
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Wt}x{Ht}",
                           "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
                           "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    trail = int(round(fps))
    for i, fr in VideoReader(ep, ("ego_cam_a",)):
        if i >= n:
            break
        raw = cv2.remap(fr["ego_cam_a"], *mapa, cv2.INTER_LINEAR)
        ov = raw.copy()
        p3 = np.full((ph, pw, 3), 245, np.uint8)
        view.grid(p3, y_grid, x_rng, z_rng)
        view.head(p3)
        for s, side in enumerate("LR"):
            r = 2 * i + s
            lo = max(0, i - trail)
            view.polyline(p3, wrist[lo:i + 1, s][np.isfinite(wrist[lo:i + 1, s, 0])], COL[side], 2)
            if not tr["valid"][r]:
                continue
            col = COL[side] if tr["source"][r] == "observed" else (160, 160, 160)
            faces = mano.faces_for(side == "R")
            ov = draw_mesh(ov, V[i, s, :, :2] / V[i, s, :, 2:3] * np.diag(A.K)[:2] + A.K[:2, 2], faces, col)
            view.mesh(p3, V[i, s], faces, col)
        tiles = []
        for img, label in ((raw, "cam_a (input)"), (ov, f"{a.method} overlay"), (None, f"{a.method}: hands in 3D")):
            t = p3 if img is None else cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
            col = (40, 40, 40) if img is None else (255, 255, 255)
            cv2.putText(t, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
            tiles.append(t)
        cv2.putText(tiles[0], f"{ep}  f{i}  t={i / fps:5.2f}s", (10, ph - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tiles[2], "head-camera frame; L orange / R red, 1 s wrist trail; grid 5 cm", (10, ph - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
        ff.stdin.write(np.ascontiguousarray(np.hstack(tiles)).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__" and not (len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "compare"):
    main()


# ----------------------------------------------------------------------------------------------------- comparison mode
CMP_COL = [(150, 150, 150), (40, 140, 255), (200, 120, 40)]  # BGR per method: grey, orange, blue


def compare_main():
    """cam_a | side view | top view, several methods' hands together + triangulated pseudo-GT joints (green).

    python -m handbench.viz3d compare --episode 12 --methods E0,E1best-sgbm
    Views look ACROSS the head camera's line of sight, so depth errors are visible: side view from the right of the
    head (depth runs left->right, head camera at depth 0 in frame), oblique view from above-right of the hands looking
    down-left at them. Head-camera frame.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--methods", default="E0,E1best-sgbm")
    ap.add_argument("--out", default="")
    ap.add_argument("--panel-h", type=int, default=480)
    ap.add_argument("--crf", type=int, default=28)
    a = ap.parse_args()
    ep = paths.ep_name(a.episode)
    methods = a.methods.split(",")
    out = a.out or str(paths.RESULTS / "overlays" / f"{ep}_3d_{'_vs_'.join(methods)}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cams = load_rig(paths.rig_json())
    A = cams["ego_cam_a"]
    mano = Mano()
    fps = paths.fps()
    tri = np.load(paths.REFS / ep / "tri.npz")["tri_abc"]
    V, ok = {}, {}
    for m in methods:
        tr = hf.load(paths.RESULTS / m / ep / "hands.npz")
        n = tr.n_frames
        Vm = np.full((n, 2, 778, 3), np.nan)
        for r in np.nonzero(tr["valid"] & (tr["source"] == "observed"))[0]:
            v, _ = mano.forward(tr["global_orient"][r], tr["hand_pose"][r], tr["betas"][r], tr["transl"][r],
                                r % 2 == 1, tr["scale"][r])
            Vm[r // 2, r % 2] = v[0]
        V[m] = Vm
    n = min(v.shape[0] for v in V.values())
    pts = np.concatenate([V[m][np.isfinite(V[m][..., 0])] for m in methods] + [tri[np.isfinite(tri[..., 0])]])
    lo, hi = np.percentile(pts, 1, 0), np.percentile(pts, 99, 0)
    target = (lo + hi) / 2
    span = float(np.max(hi - lo))
    ph, pw = a.panel_h, int(a.panel_h * 4 / 3)
    # side view: frame everything from the head camera (depth 0) to the farthest hand, so nothing runs into the camera
    lo_s, hi_s = np.minimum(lo, 0.0), np.maximum(hi, 0.0)
    target_s = (lo_s + hi_s) / 2
    span_s = float(max(hi_s[2] - lo_s[2], hi_s[1] - lo_s[1])) + 0.3
    d = 1.5
    side = View3D(target_s, (pw, ph), f=0.85 * pw * d / span_s, offset=(d, 0, 0), up=(0, -1, 0))
    # oblique view: above and to the right of the hands, slightly behind them, looking down-left at the hands
    d2 = 1.0
    top = View3D(target, (pw, ph), f=0.75 * pw * d2 / max(span, 0.35), offset=np.array([0.55, -0.65, -0.35]) * d2,
                 up=(0, -1, 0))
    mapa = cv2.initUndistortRectifyMap(A.K, A.dist, np.eye(3), A.K, A.size, cv2.CV_32FC1)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                           f"{3 * pw}x{ph}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", str(a.crf), "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    z_far = max(hi[2], target[2]) + 0.1
    for i, fr in VideoReader(ep, ("ego_cam_a",)):
        if i >= n:
            break
        raw = cv2.resize(cv2.remap(fr["ego_cam_a"], *mapa, cv2.INTER_LINEAR), (pw, ph), interpolation=cv2.INTER_AREA)
        panels = []
        for view, name in ((side, "side view (depth ->)"), (top, "oblique view (from above-right)")):
            p = np.full((ph, pw, 3), 245, np.uint8)
            # depth ticks every 10 cm along the camera axis, head camera, optical axis
            for z in np.arange(0.1, z_far + 1e-9, 0.1):
                view.polyline(p, np.array([[target[0] - 0.02, target[1], z], [target[0] + 0.02, target[1], z]]),
                              (190, 190, 190), 1)
                uv, _ = view.proj(np.array([[target[0], target[1], z]]))
                cv2.putText(p, f"{z:.1f}m", tuple(np.round(uv[0] + [4, -4]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (150, 150, 150), 1, cv2.LINE_AA)
            view.polyline(p, np.array([[0, 0, 0], [0, 0, z_far]]), (200, 200, 200), 1)
            view.head(p, 0.04)
            for k, m in enumerate(methods):
                for s, side_ in enumerate("LR"):
                    if np.isfinite(V[m][i, s, 0, 0]):
                        view.mesh(p, V[m][i, s], mano.faces_for(side_ == "R"), CMP_COL[k % len(CMP_COL)])
            g = tri[i][np.isfinite(tri[i][..., 0])]
            if len(g):
                uv, zz = view.proj(g)
                for q in uv[zz > 0]:
                    cv2.circle(p, tuple(np.round(q).astype(int)), 3, (40, 170, 40), -1, cv2.LINE_AA)
            cv2.putText(p, name, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2, cv2.LINE_AA)
            y = ph - 12
            for k, m in reversed(list(enumerate(methods))):
                cv2.putText(p, m, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CMP_COL[k % len(CMP_COL)], 2, cv2.LINE_AA)
                y -= 20
            cv2.putText(p, "green dots = triangulated reference joints", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (40, 170, 40), 1, cv2.LINE_AA)
            panels.append(p)
        cv2.putText(raw, f"{ep}  f{i}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        ff.stdin.write(np.ascontiguousarray(np.hstack([raw] + panels)).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "compare":
    compare_main()
