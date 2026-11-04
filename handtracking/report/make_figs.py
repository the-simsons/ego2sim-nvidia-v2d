"""Figures for handtracking_summary.tex added on 30 Oct 2026 (the earlier figs/ were made by hand-run snippets):

  figs/fig_triangulation.pdf   how the Track 3 reference joints are built (episode 12, frame 2)
  figs/fig_models.pdf          WiLoR, HaMeR, MANO and HaWoR in the V2D hand pipeline
  figs/fig_hot3d.pdf           HOT3D cameras, floor and fusion results, one fused frame from above

Run from handtracking/:  PYTHONPATH=. .venv-bench/bin/python report/make_figs.py
"""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from handbench import build_refs as B
from handbench import format as hf
from handbench import paths
from handbench.calib import load_rig
from handbench.datasets import hot3d as H
from handbench.experiments import hot3d_check as C
from handbench.experiments import hot3d_floor as F
from handbench.experiments import hot3d_fusion as FU
from handbench.mano import Mano
from handbench.overlay import draw_mesh
from handbench.refs_io import read_frames

FIGS = paths.ROOT / "report" / "figs"
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13),
         (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
CAMCOL = {"ego_cam_c": (31, 119, 180), "ego_cam_a": (255, 127, 14), "ego_cam_b": (44, 160, 44)}  # RGB
CAMNAME = {"ego_cam_c": "cam_c (left, mono)", "ego_cam_a": "cam_a (centre, colour)", "ego_cam_b": "cam_b (right, mono)"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.spines.top": False, "axes.spines.right": False})


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def skeleton(img, uv, color, th):
    for a, b in BONES:
        if np.isfinite(uv[[a, b]]).all():
            cv2.line(img, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), color, th, cv2.LINE_AA)


def crop_box(pts, shape, pad=0.25, aspect=None):
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - pad * w, x1 + pad * w, y0 - pad * h, y1 + pad * h
    if aspect:  # width / height
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        if w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    H_, W_ = shape[:2]
    return int(max(0, x0)), int(max(0, y0)), int(min(W_, x1)), int(min(H_, y1))


# --------------------------------------------------------------------------------------------------- triangulation
def fig_triangulation(ep="episode_000012", i=2, s=0):
    """One hand (s: 0 left, 1 right) in the three cameras, and its rays from above."""
    cams = load_rig(paths.rig_json())
    fr = read_frames(ep, i)
    kp = np.load(paths.REFS / ep / "kp2d.npz")["kp"][i, s]
    sc = B.gated_scores(ep)[i, s]
    t = np.load(paths.REFS / ep / "tri.npz")
    X, U = t["tri_abc"][i, s].astype(float), t["tri_abc_views"][i, s]
    fig = plt.figure(figsize=(12, 3.3))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1.1], wspace=0.06, figure=fig)
    for col, cn in enumerate(("ego_cam_c", "ego_cam_a", "ego_cam_b")):
        v = B.VIEWS.index(cn)
        img = fr[cn].copy()
        P = cams[cn].project(X)
        good = np.isfinite(P[:, 0])
        x0, y0, x1, y1 = crop_box(P[good], img.shape, 0.45, aspect=1.0)
        th = max(1, (x1 - x0) // 120)
        skeleton(img, P, (255, 255, 255), th)
        for j in range(21):
            q = kp[v, j]
            if not np.isfinite(q).all():
                continue
            if U[j, v]:
                cv2.circle(img, tuple(np.round(q).astype(int)), 3 * th, tuple(int(x) for x in CAMCOL[cn][::-1]), -1,
                           cv2.LINE_AA)
            elif sc[v, j] >= 0.5:
                cv2.drawMarker(img, tuple(np.round(q).astype(int)), (40, 40, 220), cv2.MARKER_TILTED_CROSS, 7 * th, th)
        for j in np.nonzero(good)[0]:
            cv2.circle(img, tuple(np.round(P[j]).astype(int)), 2 * th, (0, 0, 0), th, cv2.LINE_AA)
        ax = fig.add_subplot(gs[col])
        ax.imshow(rgb(img[y0:y1, x0:x1]))
        ax.set_title(CAMNAME[cn], color=np.array(CAMCOL[cn]) / 255)
        ax.axis("off")
    # from above, sideways exaggerated so the rays separate (stretching x keeps straight lines straight)
    EX = 6.0
    ax = fig.add_subplot(gs[3])
    for cn, dx in (("ego_cam_c", -14), ("ego_cam_a", 0), ("ego_cam_b", 14)):
        c = cams[cn].center_in_a
        v = B.VIEWS.index(cn)
        for j in range(21):
            if np.isfinite(X[j, 0]) and U[j, v]:
                ax.plot([EX * c[0], EX * X[j, 0]], [c[2], X[j, 2]], color=np.array(CAMCOL[cn]) / 255, lw=0.6, alpha=0.5)
        ax.plot(EX * c[0], c[2], marker="^", ms=8, color=np.array(CAMCOL[cn]) / 255, mec="k", mew=0.5)
        ax.annotate(cn.replace("ego_", ""), (EX * c[0], c[2]), xytext=(dx, -13), textcoords="offset points",
                    ha="center", fontsize=7)
    for a, b in BONES:
        if np.isfinite(X[[a, b], 0]).all():
            ax.plot(EX * X[[a, b], 0], X[[a, b], 2], "k-", lw=1)
    ax.plot(EX * X[:, 0], X[:, 2], "o", ms=2.5, color="k")
    ax.set_xlabel(f"sideways (stretched {EX:.0f}x so the rays can be told apart)")
    ax.set_ylabel("distance from cam_a (m)")
    ax.set_title("from above: each camera's rays meet at the joints")
    ax.set_ylim(-0.06, np.nanmax(X[:, 2]) + 0.05)
    ax.set_xticks([])
    fig.savefig(FIGS / "fig_triangulation.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------- models
def render_mano(V, faces, J, R, size=320):
    """Shaded orthographic render of a MANO mesh rotated by R (rows = image x, image y, viewing direction)."""
    Vr, Jr = V @ R.T, J @ R.T
    lo, hi = Vr[:, :2].min(0), Vr[:, :2].max(0)
    s = 0.85 * size / (hi - lo).max()
    off = size / 2 - s * (lo + hi) / 2
    uv, ju = Vr[:, :2] * s + off, Jr[:, :2] * s + off
    img = np.full((size, size, 3), 255, np.uint8)
    tri = Vr[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.35 + 0.65 * np.abs(n[:, 2])
    for f in np.argsort(-tri[:, :, 2].mean(1)):
        c = (np.array([235, 200, 170]) * shade[f]).astype(int).tolist()
        cv2.fillPoly(img, [np.round(uv[faces[f]] * 4).astype(np.int32)], c, cv2.LINE_AA, shift=2)
    skeleton(img, ju, (90, 90, 90), 1)
    for j in range(21):
        cv2.circle(img, tuple(np.round(ju[j]).astype(int)), 3, (200, 40, 40) if j in (0, 4, 8, 12, 16, 20) else (60, 60, 60),
                   -1, cv2.LINE_AA)
    return img


def flow_panel(ax):
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.55)
    boxes = [(0.05, "ego video\n(cam_a)", "#eeeeee"), (1.75, "WiLoR\nfinds each hand\n(box, left/right)", "#fde2c8"),
             (3.55, "SAM2\nhand mask\nin every frame", "#eeeeee"), (5.35, "HaMeR\nMANO hand\nper frame", "#fde2c8"),
             (7.15, "MoGe depth\nsets the hand's\ndistance", "#eeeeee"), (8.85, "3D hand\ntracks", "#d9ead3")]
    for x, txt, colr in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.15), 1.35 if x < 8.8 else 1.1, 0.9, boxstyle="round,pad=0.03", fc=colr, ec="0.4"))
        ax.text(x + (0.675 if x < 8.8 else 0.55), 1.6, txt, ha="center", va="center", fontsize=7)
    for a, b in ((1.4, 1.75), (3.1, 3.55), (4.9, 5.35), (6.7, 7.15), (8.5, 8.85)):
        ax.add_patch(FancyArrowPatch((a, 1.6), (b, 1.6), arrowstyle="-|>", mutation_scale=9, color="0.3"))
    ax.add_patch(FancyBboxPatch((1.75, 0.08), 6.75, 0.75, boxstyle="round,pad=0.03", fc="#dde7f5", ec="0.4"))
    ax.text(5.125, 0.45, "HaWoR (alternative, E2): one network takes the whole video clip and outputs MANO hands for every "
            "frame,\nplus the camera's motion; it also fills in frames where a hand is briefly hidden",
            ha="center", va="center", fontsize=7)
    ax.add_patch(FancyArrowPatch((8.5, 0.45), (9.4, 1.13), arrowstyle="-|>", mutation_scale=9, color="0.3"))
    ax.text(0.05, 2.5, "Upstream V2D hand stage (our baseline E0), and where each model sits", fontsize=8.5,
            fontweight="bold", va="top")


def fig_models(ep="episode_000012", i=40, clip_name="clip-003147", hi=60):
    cams = load_rig(paths.rig_json())
    A = cams["ego_cam_a"]
    mano = Mano()
    raw = read_frames(ep, i, ("ego_cam_a",))["ego_cam_a"]
    fig = plt.figure(figsize=(12, 4.6))
    gs = GridSpec(2, 4, height_ratios=[0.5, 1], hspace=0.05, wspace=0.05, figure=fig)
    flow_panel(fig.add_subplot(gs[0, :]))
    # (a) WiLoR boxes
    img = raw.copy()
    corners = []
    for d in json.load(open(paths.CACHE / ep / "wilor_raw" / f"{i:06d}.json")):
        b = d["bbox"]
        c = (60, 60, 230) if d["is_right"] else (40, 140, 255)
        cv2.rectangle(img, (int(b["x0"]), int(b["y0"])), (int(b["x1"]), int(b["y1"])), c, 7)
        cv2.putText(img, f"{'right' if d['is_right'] else 'left'} {d['score']:.2f}", (int(b["x0"]) + 8, int(b["y0"]) - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, c, 5, cv2.LINE_AA)
        corners += [(b["x0"], b["y0"] - 80), (b["x1"], b["y1"])]
    x0, y0, x1, y1 = crop_box(np.array(corners, float), img.shape, 0.3, aspect=1.25)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(rgb(img[y0:y1, x0:x1]))
    ax.set_title("(a) WiLoR: hand boxes in cam_a")
    ax.axis("off")
    # (b) HaMeR meshes (E0) on the undistorted cam_a
    tr = hf.load(paths.RESULTS / "E0" / ep / "hands.npz")
    und = cv2.remap(raw, *cv2.initUndistortRectifyMap(A.K, A.dist, np.eye(3), A.K, A.size, cv2.CV_32FC1), cv2.INTER_LINEAR)
    pts = []
    posed = None
    for s in range(2):
        r = hf.row_index(i, B.SIDES[s])
        if not tr["valid"][r]:
            continue
        V, J = mano.forward(tr["global_orient"][r], tr["hand_pose"][r], tr["betas"][r], tr["transl"][r], s == 1,
                            tr["scale"][r])
        uv = V[0][:, :2] / V[0][:, 2:3] * np.diag(A.K)[:2] + A.K[:2, 2]
        und = draw_mesh(und, uv, mano.faces_for(s == 1), (60, 60, 230) if s else (40, 140, 255), 0.65)
        pts.append(uv)
        if s == 1:
            posed = tr["hand_pose"][r], tr["betas"][r]
    x0, y0, x1, y1 = crop_box(np.concatenate(pts), und.shape, 0.2, aspect=1.25)
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(rgb(und[y0:y1, x0:x1]))
    ax.set_title("(b) HaMeR: one MANO hand per frame")
    ax.axis("off")
    # (c) MANO, flat and posed, same viewpoint
    faces = mano.faces_for(True)
    V0, J0 = mano.forward(np.zeros(3), np.zeros(45), np.zeros(10), None, True, 1.0)
    V0, J0 = V0[0], J0[0]
    _, _, Vt = np.linalg.svd(V0 - V0.mean(0))
    R = Vt.copy()
    if np.linalg.det(R) < 0:
        R[2] *= -1
    V1, J1 = mano.forward(np.zeros(3), posed[0], posed[1], None, True, 1.0)
    pan = np.concatenate([render_mano(V0, faces, J0, R), render_mano(V1[0], faces, J1[0], R)], 1)
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(pan)
    ax.set_title("(c) MANO: flat, and posed as in (b)")
    ax.axis("off")
    # (d) HaWoR on HOT3D, with the motion-capture ground truth
    cp = next(c for c in H.clips() if c.stem == clip_name)
    clip = H.Clip(cp)
    hmano = H.Hot3dMano()
    ex = np.load(C.OUT / cp.stem / "export.npz")
    K = H.pinhole_K()
    maps = H.pinhole_from_fisheye_maps(clip.cameras(0)[H.RGB]["params"], K, (H.SIZE_PIN, H.SIZE_PIN), H.R_CAM_FROM_PIN)
    img = cv2.remap(clip.image(hi, H.RGB), *maps, cv2.INTER_LINEAR)
    pts = []
    for s, side, ir in F.SIDES:
        Jh, Vh = F.hawor_hands(clip, ex, hmano, s, ir, min(clip.n, ex["observed"].shape[1]))
        if np.isfinite(Vh[hi, 0, 0]):
            uv = Vh[hi][:, :2] / Vh[hi][:, 2:3] * K[0, 0] + K[:2, 2]
            img = draw_mesh(img, uv, hmano.mano.faces_for(ir), (40, 140, 255), 0.65)
            pts.append(uv)
        g, _ = C.gt_in_P(clip, hmano, hi, side, ir)
        if g is not None:
            gu = g[:, :2] / g[:, 2:3] * K[0, 0] + K[:2, 2]
            skeleton(img, gu, (40, 200, 40), 4)
            pts.append(gu)
    x0, y0, x1, y1 = crop_box(np.concatenate(pts), img.shape, 0.25, aspect=1.25)
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(rgb(img[y0:y1, x0:x1]))
    ax.set_title("(d) HaWoR on HOT3D (green: true hand)")
    ax.axis("off")
    fig.savefig(FIGS / "fig_models.pdf", bbox_inches="tight", dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------- HOT3D
def fig_hot3d(clip_name="clip-003147"):
    floor = json.load(open(paths.RESULTS / "hot3d_floor.json"))
    fus = json.load(open(paths.RESULTS / "hot3d_fusion.json"))
    cp = next(c for c in H.clips() if c.stem == clip_name)
    clip = H.Clip(cp)
    cams, maps, (left, right), _ = C.rig(clip)
    K = H.pinhole_K()
    name, _, _, joints = FU.fuse_clip(cp, keep_joints=True)
    # frame where the fused hand moved most toward the GT
    best, gain = None, 0
    for side, (G, J) in joints.items():
        e_raw = np.linalg.norm(J["raw"] - G, axis=-1).mean(1)
        e_fus = np.linalg.norm(J["tri-interp+sgbm+savgol"] - G, axis=-1).mean(1)
        g = np.nan_to_num(e_raw - e_fus, nan=-1)
        if g.max() > gain:
            best, gain = (side, int(np.argmax(g))), g.max()
    side, fi = best
    fig = plt.figure(figsize=(12, 6.2))
    gs = GridSpec(2, 4, height_ratios=[1, 1.05], hspace=0.38, wspace=0.42, figure=fig)
    # top: the three Aria views at that frame, and their layout
    maps["a"] = H.pinhole_from_fisheye_maps(clip.cameras(0)[H.RGB]["params"], K, (H.SIZE_PIN, H.SIZE_PIN), H.R_CAM_FROM_PIN)
    views = (("left mono (cam_c role)", cv2.remap(clip.image(fi, left), *maps[left], cv2.INTER_LINEAR), "ego_cam_c"),
             ("RGB (cam_a role)", cv2.remap(clip.image(fi, H.RGB), *maps["a"], cv2.INTER_LINEAR), "ego_cam_a"),
             ("right mono (cam_b role)", cv2.remap(clip.image(fi, right), *maps[right], cv2.INTER_LINEAR), "ego_cam_b"))
    for k, (title, img, cn) in enumerate(views):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(rgb(img))
        ax.set_title(title, color=np.array(CAMCOL[cn]) / 255)
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 3])
    for cn, lab, dx in (("ego_cam_c", "left\nmono", -16), ("ego_cam_a", "RGB", 10), ("ego_cam_b", "right\nmono", 0)):
        c = 100 * cams[cn].center_in_a
        ax.plot(c[0], c[2], "^", ms=10, color=np.array(CAMCOL[cn]) / 255, mec="k", mew=0.5)
        ax.annotate(lab, (c[0], c[2]), xytext=(dx, 9), textcoords="offset points", ha="center", fontsize=7)
    ca, cb, cc = (100 * cams[v].center_in_a for v in ("ego_cam_a", "ego_cam_b", "ego_cam_c"))
    for p, q, y in ((ca, cc, -1.5), (ca, cb, -3.0)):
        ax.annotate("", (p[0], y), (q[0], y), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text((p[0] + q[0]) / 2, y - 0.9, f"{np.linalg.norm(p - q):.1f} cm", ha="center", fontsize=7)
    ax.set_xlim(min(ca[0], cb[0], cc[0]) - 3, max(ca[0], cb[0], cc[0]) + 3)
    ax.set_ylim(-5, 3)
    ax.set_xlabel("sideways (cm)")
    ax.set_yticks([])
    ax.set_title("Aria camera positions, from above")
    # bottom left: floor by camera combination
    ax = fig.add_subplot(gs[1, 0])
    bc = floor["floor"]["gt-mask"]["by_combo"]
    labels = {"abc": "all three", "ab": "RGB +\nright mono", "ac": "RGB +\nleft mono"}
    xs = np.arange(3)
    for k, (stat, colr) in enumerate((("mean", "#4c72b0"), ("median", "#9db8dd"))):
        ax.bar(xs + (k - 0.5) * 0.38, [bc[c][stat] for c in ("abc", "ab", "ac")], 0.38, color=colr, label=stat)
    ax.set_xticks(xs, [f"{labels[c]}\n{100 * bc[c]['share']:.0f}% of joints" for c in ("abc", "ab", "ac")],
                  fontsize=6)
    ax.set_ylabel("reference joint error (mm)")
    ax.set_title("Reference error, by cameras used")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    # bottom middle: fusion variants
    ax = fig.add_subplot(gs[1, 1:3])
    S = fus["summary"]
    rows = [("raw HaWoR", "raw"), ("+ shift (frames with\ntriangulated joints)", "tri-shift"),
            ("+ shift carried\nacross gaps", "tri-interp"), ("+ SGBM fallback\n+ smoothing", "tri-interp+sgbm+savgol")]
    ys = np.arange(len(rows))[::-1]
    ax.barh(ys + 0.19, [S[m]["mpjpe"] for _, m in rows], 0.38, color="#4c72b0", label="mean")
    ax.barh(ys - 0.19, [S[m]["mpjpe_median"] for _, m in rows], 0.38, color="#9db8dd", label="median")
    for y, (_, m) in zip(ys, rows):
        ax.text(S[m]["mpjpe"] + 0.4, y + 0.19, f"{S[m]['mpjpe']:.1f}", va="center", fontsize=7)
        ax.text(S[m]["mpjpe_median"] + 0.4, y - 0.19, f"{S[m]['mpjpe_median']:.1f}", va="center", fontsize=7)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlabel("3D joint error against the true hands (mm), 1,433 hand-frames, 7 clips")
    ax.set_title("Correcting HaWoR with the three cameras")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    # bottom right: the chosen frame from above (true scale)
    G, J = joints[side]
    ax = fig.add_subplot(gs[1, 3])
    for lab, P, colr in (("true hand", G[fi], "#2ca02c"), ("raw HaWoR", J["raw"][fi], "0.55"),
                         ("corrected", J["tri-interp+sgbm+savgol"][fi], "#1f4e9c")):
        for a, b in BONES:
            ax.plot(100 * P[[a, b], 0], 100 * P[[a, b], 2], "-", color=colr, lw=1.2)
        ax.plot([], [], "-", color=colr, label=lab)
    c = 100 * G[fi].mean(0)
    u = c / np.linalg.norm(c)
    ax.plot([c[0] - 14 * u[0], c[0] + 10 * u[0]], [c[2] - 14 * u[2], c[2] + 10 * u[2]], ":", color="0.4", lw=0.8)
    ax.annotate("to the camera", (c[0] - 12 * u[0], c[2] - 12 * u[2]), fontsize=6.5, color="0.35", ha="right")
    allp = 100 * np.concatenate([G[fi], J["raw"][fi], J["tri-interp+sgbm+savgol"][fi]])
    m = allp[:, [0, 2]].mean(0)
    r = np.abs(allp[:, [0, 2]] - m).max() + 4
    ax.set_xlim(m[0] - r, m[0] + r)
    ax.set_ylim(m[1] - r, m[1] + r)
    ax.set_aspect("equal")
    ax.set_xlabel("sideways (cm)")
    ax.set_ylabel("distance from RGB camera (cm)")
    e_r = 1000 * np.linalg.norm(J["raw"][fi] - G[fi], axis=-1).mean()
    e_f = 1000 * np.linalg.norm(J["tri-interp+sgbm+savgol"][fi] - G[fi], axis=-1).mean()
    ax.set_title(f"One frame from above: {e_r:.0f} mm -> {e_f:.0f} mm")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.savefig(FIGS / "fig_hot3d.pdf", bbox_inches="tight", dpi=170)
    plt.close(fig)
    return name, side, fi, e_r, e_f


if __name__ == "__main__":
    fig_triangulation()
    fig_models()
    print("hot3d example:", fig_hot3d())
