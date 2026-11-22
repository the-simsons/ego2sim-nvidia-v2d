"""Trajectory metrics of a reconstructed object against motion capture.

Alignment: the reconstruction lives in its own world (camera frame 0, or a gravity-aligned table frame). One rigid
transform maps it onto the mocap world, estimated ONLY on the first still stretch of the ground truth, so drift and
errors later in the clip are not absorbed. With a single object this also defines frame 0 as correct, so frame-0
error is reported only when the alignment comes from somewhere else (camera markers, a second object).

Per frame: position error of the mesh origin, rotation error, ADD (mean distance of corresponding mesh points) and
ADD-S (mean distance to the closest point, which forgives symmetric look-alikes), rotation error modulo the declared
symmetry, and look-alike flips: frames whose rotation error exceeds 30 deg while a 180-degree turn of the mesh about
one of its axes brings it within 15 deg."""
import numpy as np
from scipy.spatial import cKDTree

from e2s.geometry import box_symmetries, mean_pose, rotation_angle_deg, runs, still_mask, symmetry_rotations

AUC_MAX_CM = 10.0


def load_obj_vertices(path, max_points=2000, seed=0):
    v = np.array([[float(x) for x in l.split()[1:4]] for l in open(path) if l.startswith("v ")])
    if len(v) > max_points:
        v = v[np.random.default_rng(seed).choice(len(v), max_points, replace=False)]
    return v


def auc(errors_cm, max_cm=AUC_MAX_CM):
    """Area under the accuracy-vs-threshold curve for thresholds 0..max_cm, normalised to [0, 1]."""
    e = np.sort(np.asarray(errors_cm))
    th = np.linspace(0, max_cm, 201)
    acc = np.searchsorted(e, th, side="right") / max(len(e), 1)
    area = float(np.sum((acc[1:] + acc[:-1]) * np.diff(th)) / 2)   # trapezoid rule (np.trapz is gone in numpy 2)
    return area / max_cm


def align_on_first_still(T_rec, T_gt, fps, min_s=0.25, candidates=None):
    """(G, frames used, how): G maps the reconstruction's world onto the mocap world.

    A tracker that sees a symmetric mesh may start in a look-alike orientation; aligning on that start would rotate
    the whole reconstructed world by the look-alike turn and inflate every later error. So one alignment is built per
    candidate symmetry S of the mesh (G_S from T_gt S T_rec^-1 on the still frames), and the one with the lowest
    median rotation error (deg) + position error (cm) over the clip is kept. This is a discrete choice among a few turns, not a fit to the clip."""
    valid = ~np.isnan(T_rec[:, 0, 0])
    still = still_mask(T_gt[:, :3, 3], fps) & valid
    st = runs(still, int(min_s * fps))
    idx = np.arange(st[0][0], st[0][1]) if st else np.nonzero(valid)[0][:10]
    how = "first still stretch" if st else "first 10 valid frames (no still stretch)"
    best = None
    Tr = np.nan_to_num(T_rec)
    for k, S in enumerate(candidates or [np.eye(4)]):
        G = mean_pose(T_gt[idx] @ S @ np.linalg.inv(T_rec[idx]))
        A = (G @ Tr @ S)[valid]
        # rotation alone ties when the motion commutes with the turn (e.g. spinning about the flip axis); positions don't
        err = (np.median(rotation_angle_deg(A[:, :3, :3], T_gt[valid][:, :3, :3]))
               + 100 * np.median(np.linalg.norm(A[:, :3, 3] - T_gt[valid][:, :3, 3], axis=1)))
        if best is None or err < best[0]:
            best = (err, k, G, S)
    if candidates and len(candidates) > 1:
        how += f", start orientation = symmetry {best[1]} of {len(candidates)}"
    return best[2], best[3], idx, how


def object_metrics(T_rec, T_gt, vertices, fps, symmetry=None):
    """Per-frame arrays and a summary dict for one object in one clip."""
    n = min(len(T_rec), len(T_gt))
    T_rec, T_gt = T_rec[:n], T_gt[:n]
    valid = ~np.isnan(T_rec[:, 0, 0])
    G, S0, idx, how = align_on_first_still(T_rec, T_gt, fps, candidates=box_symmetries())
    T = np.where(valid[:, None, None], G @ np.nan_to_num(T_rec) @ S0, np.nan)
    v = vertices
    pos = np.linalg.norm(T[:, :3, 3] - T_gt[:, :3, 3], axis=1) * 100
    rot = rotation_angle_deg(np.nan_to_num(T[:, :3, :3]), T_gt[:, :3, :3])
    P = np.einsum("nij,mj->nmi", np.nan_to_num(T[:, :3, :3]), v) + np.nan_to_num(T[:, None, :3, 3])
    Q = np.einsum("nij,mj->nmi", T_gt[:, :3, :3], v) + T_gt[:, None, :3, 3]
    add = np.linalg.norm(P - Q, axis=2).mean(1) * 100
    adds = np.array([cKDTree(Q[i]).query(P[i])[0].mean() for i in range(n)]) * 100
    syms = symmetry_rotations(symmetry)
    rot_sym = np.min([rotation_angle_deg(np.nan_to_num(T[:, :3, :3]) @ S[:3, :3], T_gt[:, :3, :3]) for S in syms], axis=0)
    flip_alt = np.min([rotation_angle_deg(np.nan_to_num(T[:, :3, :3]) @ S[:3, :3], T_gt[:, :3, :3]) for S in box_symmetries()[1:]], axis=0)
    flip = valid & (rot > 30) & (flip_alt < 15)
    flip_runs = runs(flip)
    for a in (pos, rot, add, adds, rot_sym):
        a[~valid] = np.nan
    vv = valid
    end = np.nan
    st = runs(still_mask(T_gt[:, :3, 3], fps) & valid, int(0.25 * fps))
    if len(st) > 1:
        a, b = st[-1]
        end = float(np.nanmedian(pos[a:b]))
    summary = {
        "frames": int(n), "coverage": float(vv.mean()), "alignment": how,
        "pos_cm_median": float(np.nanmedian(pos)), "pos_cm_p95": float(np.nanpercentile(pos, 95)), "pos_cm_max": float(np.nanmax(pos)),
        "rot_deg_median": float(np.nanmedian(rot)), "rot_deg_p95": float(np.nanpercentile(rot, 95)), "rot_deg_max": float(np.nanmax(rot)),
        "rot_sym_deg_median": float(np.nanmedian(rot_sym)), "rot_sym_deg_p95": float(np.nanpercentile(rot_sym, 95)),
        "add_cm_median": float(np.nanmedian(add)), "add_cm_p95": float(np.nanpercentile(add, 95)),
        "adds_cm_median": float(np.nanmedian(adds)), "adds_cm_p95": float(np.nanpercentile(adds, 95)),
        "add_auc": auc(np.where(vv, add, np.inf)), "adds_auc": auc(np.where(vv, adds, np.inf)),
        "flip_frac": float(flip[vv].mean()) if vv.any() else np.nan,
        "flip_longest_s": float(max((b - a for a, b in flip_runs), default=0) / fps),
        "end_still_pos_cm": end,
    }
    per_frame = {"pos_cm": pos, "rot_deg": rot, "add_cm": add, "adds_cm": adds, "rot_sym_deg": rot_sym, "flip": flip}
    return summary, per_frame


SUMMARY_KEYS = ["coverage", "pos_cm_median", "pos_cm_p95", "pos_cm_max", "rot_deg_median", "rot_deg_p95", "rot_sym_deg_median",
                "add_cm_median", "add_cm_p95", "adds_cm_median", "add_auc", "adds_auc", "flip_frac", "flip_longest_s",
                "end_still_pos_cm"]


def pool(per_frame_list, fps_list):
    """Summary over several clips, from the pooled per-frame errors (each valid frame counts once)."""
    cat = {k: np.concatenate([p[k] for p in per_frame_list]) for k in per_frame_list[0]}
    ok = ~np.isnan(cat["pos_cm"])
    return {
        "coverage": float(ok.mean()),
        "pos_cm_median": float(np.nanmedian(cat["pos_cm"])), "pos_cm_p95": float(np.nanpercentile(cat["pos_cm"], 95)),
        "pos_cm_max": float(np.nanmax(cat["pos_cm"])),
        "rot_deg_median": float(np.nanmedian(cat["rot_deg"])), "rot_deg_p95": float(np.nanpercentile(cat["rot_deg"], 95)),
        "rot_sym_deg_median": float(np.nanmedian(cat["rot_sym_deg"])),
        "add_cm_median": float(np.nanmedian(cat["add_cm"])), "add_cm_p95": float(np.nanpercentile(cat["add_cm"], 95)),
        "adds_cm_median": float(np.nanmedian(cat["adds_cm"])),
        "add_auc": auc(np.where(ok, cat["add_cm"], np.inf)), "adds_auc": auc(np.where(ok, cat["adds_cm"], np.inf)),
        "flip_frac": float(cat["flip"][ok].mean()) if ok.any() else np.nan,
    }
