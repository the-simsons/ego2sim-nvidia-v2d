"""Slide media for one episode (stills + short MP4s), one subcommand per pipeline stage.
  python ppt_media.py frames  EP_DIR OUT_DIR [--frame 100]          RGB clip, still, cam_a / cam_b / cam_c triplet
  python ppt_media.py depth   EP_DIR OUT_DIR [--frame 100]          FoundationStereo depth (turbo) next to RGB
  python ppt_media.py track   EP_DIR OUT_DIR NAME=FP_DIR=MESH ...   FoundationPose meshes drawn over the video
  python ppt_media.py turn    OUT_DIR NAME=MESH ... [--cpu]         mesh turntables (pyrender on the GPU; --cpu: matplotlib)
  python ppt_media.py traj    PARQUET_DIR OUT_DIR                   3D plot of object and wrist trajectories (sim frame)
  python ppt_media.py trajvid PARQUET_DIR OUT_DIR MESH_DIR [--ep EP_DIR] [--table Z]   animated 3D reference (CPU)
Run in the v2d_foundation_pose venv (cv2, trimesh, pyrender, matplotlib). Videos are H.264 via ffmpeg."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
COL = [(77, 84, 237), (102, 191, 77), (230, 128, 77), (64, 191, 242)]  # BGR, same order as ego_overlay / render_recon_gl
W = 960


def writer(path, size, fps=20.0):
    return cv2.VideoWriter(str(path) + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def finish(path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path) + ".tmp.mp4", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "22", "-movflags", "+faststart", str(path)], check=True)
    os.remove(str(path) + ".tmp.mp4")


def frames_iter(video):
    cap = cv2.VideoCapture(str(video))
    while True:
        ok, im = cap.read()
        if not ok:
            return
        yield im


def resize(im, w=W):
    return cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_AREA)


def label(im, text):
    cv2.putText(im, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(im, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def cmd_frames(a):
    E, O = Path(a.ep), Path(a.out)
    vid = E / "stereo/cam_a_undist.mp4"
    w = None
    for i, im in enumerate(frames_iter(vid)):
        im = resize(im)
        if w is None:
            w = writer(O / "01_input_ego_rgb.mp4", im.shape[1::-1])
        w.write(im)
        if i == a.frame:
            cv2.imwrite(str(O / "01_input_ego_rgb.jpg"), im)
    w.release(); finish(O / "01_input_ego_rgb.mp4")
    ep = int(E.name[2:])
    vd = Path(a.videos)
    tiles = []
    for cam in "abc":
        f = next((vd / f"observation.images.ego_cam_{cam}").glob(f"episode_{ep:06d}.mp4"))
        for i, im in enumerate(frames_iter(f)):
            if i == a.frame:
                im = cv2.resize(im, (int(im.shape[1] * 400 / im.shape[0]), 400))
                tiles.append(label(im, f"cam_{cam}" + (" (RGB)" if cam == "a" else " (grayscale)")))
                break
    cv2.imwrite(str(O / "01_input_three_cameras.jpg"), np.hstack(tiles))


def cmd_depth(a):
    E, O = Path(a.ep), Path(a.out)
    D = sorted((E / "stereo_fs/stereo_depth").glob("*.png"))
    w = None
    for i, im in enumerate(frames_iter(E / "stereo/cam_a_undist.mp4")):
        if i >= len(D):
            break
        v = cv2.imread(str(D[i]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        m = 65535.0 / np.maximum(v, 1) - 1.0  # inverse-depth encoding (ego_stereo_depth.py); 65535 = invalid
        valid = v < 65535
        near, far = 0.25, 1.6
        n = np.clip((m - near) / (far - near), 0, 1)
        col = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_TURBO)
        col[~valid] = 0
        pair = np.hstack([label(resize(im, W // 2 * 2 // 2), "RGB (cam_a)"),
                          label(resize(col, W // 2 * 2 // 2), f"FoundationStereo depth {near}-{far} m")])
        if w is None:
            w = writer(O / "02_stereo_depth.mp4", pair.shape[1::-1])
        w.write(pair)
        if i == a.frame:
            cv2.imwrite(str(O / "02_stereo_depth.jpg"), pair)
    w.release(); finish(O / "02_stereo_depth.mp4")


def cmd_track(a):
    import trimesh
    from fp_track_diag import load_track
    E, O = Path(a.ep), Path(a.out)
    K = json.load(open(E / "stereo/intrinsics_undist.json"))
    n = len(list((E / "masks/0").glob("*.png")))
    objs = []
    for spec in a.objs:
        name, d, mp = spec.split("=", 2)
        m = trimesh.load(mp, force="mesh")
        if len(m.faces) > 8000:
            m = m.simplify_quadric_decimation(face_count=8000)
        objs.append((name, load_track(d, n), m))
    s = W / K["width"]
    w = None
    for i, im in enumerate(frames_iter(E / "stereo/cam_a_undist.mp4")):
        im = resize(im)
        over = im.copy()
        for k, (name, T, m) in enumerate(objs):
            if i >= n or np.isnan(T[i, 0, 0]):
                continue
            v = m.vertices @ T[i, :3, :3].T + T[i, :3, 3]
            ok = (v[m.faces][:, :, 2] > 0.01).all(1)
            uv = (v[:, :2] / np.maximum(v[:, 2:3], 1e-6)) * [K["fx"], K["fy"]] + [K["cx"], K["cy"]]
            uv = (uv * s).astype(np.int32)
            sil = np.zeros(im.shape[:2], np.uint8)
            cv2.fillPoly(sil, list(uv[m.faces[ok]]), 1)
            over[sil > 0] = COL[k % 4]
            cs, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cs, -1, COL[k % 4], 2)
        im = cv2.addWeighted(im, 0.55, over, 0.45, 0)
        label(im, "FoundationPose: " + ", ".join(nm for nm, _, _ in objs))
        if w is None:
            w = writer(O / "05_object_pose_tracking.mp4", im.shape[1::-1])
        w.write(im)
        if i == a.frame:
            cv2.imwrite(str(O / "05_object_pose_tracking.jpg"), im)
    w.release(); finish(O / "05_object_pose_tracking.mp4")


def cmd_turn_mpl(a):
    """CPU-only turntable (matplotlib shaded polygons), for when no GPU is free."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    O = Path(a.out)
    for spec in a.objs:
        name, mp = spec.split("=", 1)
        tm = trimesh.load(mp, force="mesh")
        if len(tm.faces) > 6000:
            tm = tm.simplify_quadric_decimation(face_count=6000)
        tm.apply_translation(-tm.bounds.mean(0))
        _, v = np.linalg.eigh(np.cov(tm.vertices.T))
        tm.vertices = tm.vertices @ v[:, ::-1]  # long axis along x
        r = float(np.abs(tm.vertices).max())
        light = np.array([0.4, -0.5, 0.8]); light /= np.linalg.norm(light)
        shade = 0.35 + 0.65 * np.clip(tm.face_normals @ light, 0, 1)
        cols = np.c_[np.outer(shade, [0.82, 0.66, 0.45]), np.ones(len(shade))]
        fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Poly3DCollection(tm.triangles, facecolors=cols, edgecolor="none"))
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r); ax.set_box_aspect((1, 1, 1)); ax.axis("off")
        frames = []
        for j in range(48):
            ax.view_init(elev=25, azim=360 * j / 48)
            fig.canvas.draw()
            im = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(label(cv2.cvtColor(im, cv2.COLOR_RGB2BGR).copy(), name))
        plt.close(fig)
        h, w_ = frames[0].shape[:2]
        w = writer(O / f"04_mesh_turntable_{name}.mp4", (w_, h), fps=16)
        for f in frames:
            w.write(f)
        w.release(); finish(O / f"04_mesh_turntable_{name}.mp4")
        cv2.imwrite(str(O / f"04_mesh_{name}.jpg"), frames[6])


def cmd_turn(a):
    if a.cpu:
        return cmd_turn_mpl(a)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh
    O = Path(a.out)
    for k, spec in enumerate(a.objs):
        name, mp = spec.split("=", 1)
        tm = trimesh.load(mp, force="mesh")
        tm.apply_translation(-tm.bounds.mean(0))
        r = float(np.linalg.norm(np.ptp(tm.bounds, 0))) / 2
        try:
            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
        except Exception:
            mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(tm.vertices, tm.faces), smooth=False)
        rend = pyrender.OffscreenRenderer(640, 640)
        frames = []
        for j in range(72):
            sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.45, 0.45, 0.45])
            ang = 2 * np.pi * j / 72
            rot = trimesh.transformations.rotation_matrix(ang, [0, 0, 1]) @ trimesh.transformations.rotation_matrix(-0.5, [1, 0, 0])
            sc.add(mesh, pose=rot)
            cam = np.eye(4); cam[2, 3] = 3.0 * r
            sc.add(pyrender.PerspectiveCamera(yfov=np.pi / 5), pose=cam)
            sc.add(pyrender.DirectionalLight(intensity=3.0), pose=cam)
            col, _ = rend.render(sc)
            frames.append(label(cv2.cvtColor(col, cv2.COLOR_RGB2BGR).copy(), name))
        rend.delete()
        w = writer(O / f"04_mesh_turntable_{name}.mp4", (640, 640), fps=24)
        for f in frames:
            w.write(f)
        w.release(); finish(O / f"04_mesh_turntable_{name}.mp4")
        cv2.imwrite(str(O / f"04_mesh_{name}.jpg"), frames[9])


def cmd_traj(a):
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq
    r = pq.read_table(glob.glob(f"{a.parquet}/*.parquet")[0]).to_pylist()[0]
    P = np.array(r["object_body_position"]); names = list(r["object_body_names"])
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    for k, n in enumerate(names):
        c = np.array(COL[k % 4][::-1]) / 255
        ax.plot(P[:, k, 0], P[:, k, 1], P[:, k, 2], color=c, lw=2.5, label=n)
        ax.scatter(*P[0, k], color=c, s=40, marker="o"); ax.scatter(*P[-1, k], color=c, s=60, marker="x")
    for side, c in (("left", "0.35"), ("right", "0.65")):
        wp = r.get(f"robot_{side}_wrist_position")
        if wp:
            wp = np.array(wp); ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], color=c, lw=1.2, ls="--", label=f"{side} wrist (Sharpa)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("Reference trajectories in the simulator frame\n(o = start, x = end)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(Path(a.out) / "09_reference_trajectories.png", dpi=160)


def cmd_trajvid(a):
    """Animated 3D view of the reference (object meshes at their poses, trails, wrists, table) next to the video. CPU."""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial.transform import Rotation as R
    r = pq.read_table(glob.glob(f"{a.parquet}/*.parquet")[0]).to_pylist()[0]
    names = list(r["object_body_names"]); P = np.array(r["object_body_position"]); Q = np.array(r["object_body_wxyz"])
    T = len(P)
    meshes = []
    for n in names:
        m = trimesh.load(Path(a.meshes) / f"{n}.obj", force="mesh")
        if len(m.faces) > 1500:
            m = m.simplify_quadric_decimation(face_count=1500)
        meshes.append((np.array(m.vertices), np.array(m.faces), np.array(m.face_normals)))
    wr = {s: np.array(r[f"robot_{s}_wrist_position"]) for s in ("left", "right") if r.get(f"robot_{s}_wrist_position")}
    pts = P.reshape(-1, 3)  # frame the objects' paths (wrists may leave the box)
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2; half = max(np.max(hi - lo) / 2 + a.margin, 0.12)
    lo3, hi3 = c - half, c + half
    lo3[2] = min(lo3[2], a.table - 0.02)
    light = np.array([0.3, -0.4, 0.85]); light /= np.linalg.norm(light)
    th = np.linspace(0, 2 * np.pi, 60)
    vid = list(frames_iter(Path(a.ep) / "stereo/cam_a_undist.mp4")) if a.ep else []
    fig = plt.figure(figsize=(6.4, 6.0), dpi=100)
    w = None
    for f in range(T):
        fig.clf(); ax = fig.add_subplot(111, projection="3d")
        ax.plot(c[0] + 0.3 * np.cos(th), c[1] + 0.3 * np.sin(th), np.full_like(th, a.table), color="#b07a45", lw=1.5)
        for k, (n, (mv, mf, mn)) in enumerate(zip(names, meshes)):
            rot = R.from_quat(np.array(Q[f, k][[1, 2, 3, 0]]))
            tri = rot.apply(mv)[mf] + P[f, k]
            nrm = rot.apply(mn)
            base = np.array(COL[k % 4][::-1]) / 255
            shade = 0.35 + 0.65 * np.clip(nrm @ light, 0, 1)
            ax.add_collection3d(Poly3DCollection(tri, facecolors=np.c_[np.outer(shade, base), np.ones(len(shade))], edgecolor="none"))
            ax.plot(P[:f + 1, k, 0], P[:f + 1, k, 1], P[:f + 1, k, 2], color=base, lw=1.5, alpha=0.8)
        for s, wp in wr.items():  # wrists: only the part inside the framed box
            g = "0.25" if s == "left" else "0.55"
            q = wp[:f + 1].copy(); q[(np.abs(q - c) > half).any(1)] = np.nan
            ax.plot(q[:, 0], q[:, 1], q[:, 2], color=g, lw=1, ls="--")
            if not np.isnan(q[-1]).any():
                ax.scatter(*q[-1], color=g, s=25)
        ax.set_xlim(lo3[0], hi3[0]); ax.set_ylim(lo3[1], hi3[1]); ax.set_zlim(lo3[2], lo3[2] + 2 * half)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=a.elev, azim=a.azim)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_title(f"Reference trajectory (sim frame)  frame {f}", fontsize=10)
        fig.canvas.draw()
        im = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
        if vid:
            v = vid[min(f, len(vid) - 1)]
            v = cv2.resize(v, (int(v.shape[1] * im.shape[0] / v.shape[0]), im.shape[0]))
            im = np.hstack([label(v, "Ego video"), im])
        im = im[: im.shape[0] // 2 * 2, : im.shape[1] // 2 * 2]
        if w is None:
            w = writer(Path(a.out) / a.name, im.shape[1::-1])
        w.write(im)
    plt.close(fig)
    w.release(); finish(Path(a.out) / a.name)


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)
p = sp.add_parser("frames"); p.add_argument("ep"); p.add_argument("out"); p.add_argument("--frame", type=int, default=100)
p.add_argument("--videos", default=os.path.expanduser("~/v2d/v2d_data/track_3/evaluation/videos/chunk-000"))
p = sp.add_parser("depth"); p.add_argument("ep"); p.add_argument("out"); p.add_argument("--frame", type=int, default=100)
p = sp.add_parser("track"); p.add_argument("ep"); p.add_argument("out"); p.add_argument("objs", nargs="+")
p.add_argument("--frame", type=int, default=100)
p = sp.add_parser("turn"); p.add_argument("out"); p.add_argument("objs", nargs="+")
p.add_argument("--cpu", action="store_true", help="matplotlib render, no GPU")
p = sp.add_parser("traj"); p.add_argument("parquet"); p.add_argument("out")
p = sp.add_parser("trajvid"); p.add_argument("parquet"); p.add_argument("out"); p.add_argument("meshes")
p.add_argument("--ep", help="episode dir: put the ego video next to the 3D view")
p.add_argument("--azim", type=float, default=90, help="90 = from the person's side (+y), like the ego view")
p.add_argument("--elev", type=float, default=35); p.add_argument("--margin", type=float, default=0.06)
p.add_argument("--table", type=float, default=0.75); p.add_argument("--name", default="09_reference_trajectory.mp4")
a = ap.parse_args()
Path(a.out).mkdir(parents=True, exist_ok=True)
{"frames": cmd_frames, "depth": cmd_depth, "track": cmd_track, "turn": cmd_turn, "traj": cmd_traj,
 "trajvid": cmd_trajvid}[a.cmd](a)
print(f"{a.cmd} -> {a.out}")
