"""Rebuild the tracks of two nested cups (ep10) where FoundationPose merged them: while nested, the pair is one rigid
stack. The nesting offset is computed from the meshes: the inner cup coaxial with the outer, open ends up, lifted along
the outer cup's axis until it no longer overlaps the outer cup's collision hulls (the CoACD hulls the simulator uses).
Per frame segment, one cup's (good) track is kept and the other is placed from it; the outer cup's frames before the
first stack frame are interpolated from its last good pose; changes are blended in over --blend frames.
  python nest_fix.py OUT_DIR HULL_DIR EP OUTER INNER --interp A B --inner-leads C D --outer-leads E F [--blend 6]
    --interp A B       outer cup: slerp/lerp from frame A to the stack pose at frame B (A, B inclusive)
    --inner-leads C D  frames where the inner cup's track is kept and the outer cup is placed under it
    --outer-leads E F  frames where the outer cup's track is kept and the inner cup is placed in it (F = -1: to the end)
Edits OUT_DIR/package in place (meshclean venv); the input parquet is kept as OUT_DIR/epEP_prenest.parquet."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("hulls"); ap.add_argument("ep"); ap.add_argument("outer"); ap.add_argument("inner")
ap.add_argument("--interp", type=int, nargs=2, required=True)
ap.add_argument("--inner-leads", type=int, nargs=2, required=True)
ap.add_argument("--outer-leads", type=int, nargs=2, required=True)
ap.add_argument("--blend", type=int, default=6); ap.add_argument("--tol-cm", type=float, default=0.15)
ap.add_argument("--static-from", type=int, help="frame from which the stack stands still upright: fit the inner cup "
                "(upright, x/y/z) to its mask over the remaining frames and hold the stack there; frames from the end of "
                "--inner-leads to this one are interpolated (overrides --outer-leads)")
ap.add_argument("--fit", nargs=5, metavar=("EP_DIR", "RAW_PKG", "INNER_MASK_ID", "INNER_FP_DIR", "INNER_MESH"),
                help="inputs for --static-from: episode dir (masks, intrinsics), the raw package, the inner cup's SAM2 "
                     "mask id, its raw FoundationPose dir and mesh (camera <- sim per frame from that raw track)")
a = ap.parse_args()
OUT = Path(a.out)


def load(name):
    d = np.load(Path(a.hulls) / f"{name}.npz")
    m = trimesh.Trimesh(d["mesh_v"], d["mesh_f"], process=False)
    pts = np.array(trimesh.sample.sample_surface(m, 3000, seed=0)[0])
    c = pts.mean(0); w, v = np.linalg.eigh(np.cov((pts - c).T))
    gaps = [abs(w[1] - w[2]), abs(w[0] - w[2]), abs(w[0] - w[1])]
    ax = v[:, int(np.argmin(gaps))].copy()
    h = (pts - c) @ ax; r = np.linalg.norm((pts - c) - np.outer(h, ax), axis=1)
    if np.percentile(r[h > h.max() - 0.02], 90) < np.percentile(r[h < h.min() + 0.02], 90):
        ax = -ax  # axis points to the open (wider) end
    planes = []
    for i in range(int(d["n"])):
        hh = trimesh.Trimesh(d[f"v{i}"], d[f"f{i}"], process=False)
        n = hh.face_normals; planes.append((n, (hh.triangles[:, 0] * n).sum(1)))
    return dict(pts=pts, c=c, ax=ax, planes=planes)


def depth(p, planes):
    best = 0.0
    for n, d in planes:
        s = p @ n.T - d; ins = (s <= 0).all(1)
        if ins.any():
            best = max(best, float((-s[ins].max(1)).max()))
    return best


def align(u, v):
    """Rotation taking unit vector u to unit vector v."""
    c = np.cross(u, v); s = np.linalg.norm(c)
    if s < 1e-9:
        return R.identity() if u @ v > 0 else R.from_rotvec(np.pi * np.array([1.0, 0, 0]))
    return R.from_rotvec(c / s * np.arctan2(s, u @ v))


Oo, Oi = load(a.outer), load(a.inner)
Ra = align(Oi["ax"], Oo["ax"])  # inner body -> outer body orientation (spin about the axis is free)


def inner_in_outer(dz):
    """Inner cup's body pose in the outer cup's body frame, centroid dz above the outer centroid along its axis."""
    t = Oo["c"] + dz * Oo["ax"] - Ra.apply(Oi["c"])
    return Ra, t


def overlap(dz):
    Rr, t = inner_in_outer(dz)
    pin = Rr.apply(Oi["pts"]) + t  # inner surface in the outer frame
    pout = Rr.inv().apply(Oo["pts"] - t)  # outer surface in the inner frame
    return max(depth(pin, Oo["planes"]), depth(pout, Oi["planes"]))


lo, hi = 0.0, 0.20
assert overlap(hi) <= a.tol_cm / 100, "inner cup still overlaps 20 cm up"
for _ in range(20):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if overlap(mid) > a.tol_cm / 100 else (lo, mid)
dz = hi
Rn, tn = inner_in_outer(dz)
print(f"nesting offset: inner centroid {100 * dz:.1f} cm above the outer centroid along the outer axis")

p = next((OUT / f"package/sim_reference/processed/sequence_id=ep{a.ep}/robot_name=sharpa_wave").glob("*.parquet"))
bak = OUT / f"ep{a.ep}_prenest.parquet"
if not bak.exists():
    shutil.copyfile(p, bak)
t = pq.read_table(bak); r = t.to_pylist()[0]
names = list(r["object_body_names"]); io, ii = names.index(a.outer), names.index(a.inner)
P = np.asarray(r["object_body_position"], float); Q = np.asarray(r["object_body_wxyz"], float); T = len(P)
rot = [R.from_quat(Q[:, k][:, [1, 2, 3, 0]]) for k in range(len(names))]
newP = P.copy()
Ro, Ri = list(rot[io]), list(rot[ii])
src = ["track"] * T, ["track"] * T  # (outer, inner) provenance per frame


def place_inner(f):  # inner from the outer pose
    return Ro[f] * Rn, P[f, io] + Ro[f].apply(tn)


def place_outer(f):  # outer from the inner pose
    Rinv = Rn.inv()
    return Ri[f] * Rinv, P[f, ii] - (Ri[f] * Rinv).apply(tn)


c0, c1 = a.inner_leads
e0, e1 = a.outer_leads; e1 = T - 1 if e1 < 0 else e1
outR, outP = list(Ro), P[:, io].copy()
inR, inP = list(Ri), P[:, ii].copy()
for f in range(c0, c1 + 1):
    outR[f], outP[f] = place_outer(f); src[0][f] = "from inner"
for f in range(e0, e1 + 1):
    inR[f], inP[f] = place_inner(f); src[1][f] = "from outer"
fit_rep = None
if a.static_from is not None:
    import sys

    import cv2
    sys.path.insert(0, str(Path(__file__).parent))
    from fp_track_diag import load_track, silhouette  # noqa: E402
    ep_dir, raw_pkg, mid, fp_dir, mesh_path = a.fit
    K = json.load(open(Path(ep_dir) / "stereo/intrinsics_undist.json"))
    s = 0.25; hw = (int(K["height"] * s), int(K["width"] * s))
    rr = pq.read_table(next((Path(raw_pkg) / f"sim_reference/processed/sequence_id=ep{a.ep}/robot_name=sharpa_wave").glob("*.parquet"))).to_pylist()[0]
    rP = np.asarray(rr["object_body_position"])[:, ii]; rQ = np.asarray(rr["object_body_wxyz"])[:, ii]
    Tfp = load_track(fp_dir, T)
    mesh = trimesh.load(mesh_path, force="mesh")
    if len(mesh.faces) > 6000:
        mesh = mesh.simplify_quadric_decimation(face_count=6000)
    S0 = a.static_from
    frames = [f for f in range(S0, T, max(1, (T - S0) // 10)) if not np.isnan(Tfp[f, 0, 0])]
    cams, masks = [], []
    for f in frames:  # camera <- sim world, from the raw track (the bundle maps every object with the same transform)
        Ts = np.eye(4); Ts[:3, :3] = R.from_quat(rQ[f][[1, 2, 3, 0]]).as_matrix(); Ts[:3, 3] = rP[f]
        cams.append(Tfp[f] @ np.linalg.inv(Ts))
        mk = cv2.imread(str(Path(ep_dir) / f"masks/{mid}/{f:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        masks.append(cv2.resize(mk.astype(np.uint8), hw[::-1], interpolation=cv2.INTER_NEAREST))
    Rup = align(Oi["ax"], np.array([0.0, 0.0, 1.0]))  # upright inner cup (spin about the axis is invisible)
    rj0 = OUT / f"ep{a.ep}_report.json"
    table = json.loads(rj0.read_text()).get("table_top", 0.75) if rj0.exists() else 0.75
    Rout_up = Rup * Rn.inv()  # outer cup orientation under the upright inner cup

    def on_table(pos):
        """Height that puts the outer cup's lowest point on the table (one camera barely constrains depth)."""
        outer_pos = pos - Rout_up.apply(tn)
        zmin = (Rout_up.apply(Oo["pts"]) + outer_pos)[:, 2].min()
        q = pos.copy(); q[2] += table - zmin
        return q

    def score(pos):
        pos = on_table(pos)
        Tw = np.eye(4); Tw[:3, :3] = Rup.as_matrix(); Tw[:3, 3] = pos
        v = []
        for C, mk in zip(cams, masks):
            sil = silhouette(mesh, C @ Tw, K, hw, s)
            v.append((sil & mk).sum() / max((sil | mk).sum(), 1))
        return float(np.mean(v))

    best = (score(inP[S0]), inP[S0].copy())  # start: inner pose at the first static frame (as placed so far)
    for step in (0.04, 0.02, 0.01, 0.005, 0.0025):  # coordinate search over x, y (z from the table), shrinking steps
        improved = True
        while improved:
            improved = False
            for k in range(2):
                for sgn in (1, -1):
                    cand = best[1].copy(); cand[k] += sgn * step
                    sc = score(cand)
                    if sc > best[0] + 1e-4:
                        best, improved = (sc, cand), True
    best = (best[0], on_table(best[1]))
    fit_rep = {"static_from": S0, "frames_scored": len(frames), "inner_iou_start": round(score(inP[S0]), 3),
               "inner_iou_fit": round(best[0], 3), "inner_pos": np.round(best[1], 4).tolist(), "table_top": table}
    print("static stack fit:", json.dumps(fit_rep))
    e0, e1 = S0, T - 1
    for f in range(e0, e1 + 1):  # hold the fitted stack: inner fixed, outer under it
        inR[f], inP[f] = Rup, best[1].copy(); src[1][f] = "static fit"
        Ri[f] = inR[f]; P[f, ii] = inP[f]
        outR[f], outP[f] = place_outer(f); src[0][f] = "from inner (static)"
    for f in range(c1 + 1, e0):  # between the inner-led run and the static stack: interpolate both
        w = (f - c1) / (e0 - c1)
        inR[f] = Slerp([0, 1], R.concatenate([inR[c1], inR[e0]]))([w])[0]; inP[f] = (1 - w) * inP[c1] + w * inP[e0]
        outR[f] = Slerp([0, 1], R.concatenate([outR[c1], outR[e0]]))([w])[0]; outP[f] = (1 - w) * outP[c1] + w * outP[e0]
        src[0][f] = src[1][f] = "interp"
i0, i1 = a.interp  # outer: from its last good pose (i0) to the stack pose at i1
sl = Slerp([i0, i1], R.concatenate([outR[i0], outR[i1]]))
for f in range(i0 + 1, i1):
    w = (f - i0) / (i1 - i0)
    outR[f] = sl([f])[0]; outP[f] = (1 - w) * outP[i0] + w * outP[i1]; src[0][f] = "interp"
# the stack must not go into the table: lift both cups where the placed outer cup's base is below the table top
rj1 = OUT / f"ep{a.ep}_report.json"
table_z = json.loads(rj1.read_text()).get("table_top", 0.75) if rj1.exists() else 0.75
lifted = 0
for f in range(T):
    if src[0][f] == "track" and src[1][f] == "track":
        continue
    zmin = (outR[f].apply(Oo["pts"]) + outP[f])[:, 2].min()
    if zmin < table_z:
        dzt = table_z - zmin
        outP[f] = outP[f] + [0, 0, dzt]; inP[f] = inP[f] + [0, 0, dzt]; lifted += 1
print(f"stack lifted onto the table in {lifted} frames")
# blend the edges of every replaced run with the original track (no jumps at segment ends)
for k, (Rl, Pl, orig_R, orig_P) in enumerate(((outR, outP, Ro, P[:, io]), (inR, inP, Ri, P[:, ii]))):
    changed = np.array([s != "track" for s in src[k]])
    for f in range(T):
        if changed[f]:
            continue
        dist = [abs(f - g) for g in np.flatnonzero(changed)] if changed.any() else []
        if dist and min(dist) <= a.blend:  # near a replaced run: mix towards it
            g = np.flatnonzero(changed)[int(np.argmin(dist))]
            w = 1 - min(dist) / (a.blend + 1)
            target = Rl[g] * (orig_R[g].inv() * orig_R[f])  # replaced pose at g, carried by the track's own motion
            Rl[f] = Slerp([0, 1], R.concatenate([orig_R[f], target]))([w])[0]
            Pl[f] = (1 - w) * orig_P[f] + w * (Pl[g] + (orig_P[f] - orig_P[g]))
newP[:, io], newP[:, ii] = outP, inP
q_o = R.concatenate(outR).as_quat()[:, [3, 0, 1, 2]]; q_i = R.concatenate(inR).as_quat()[:, [3, 0, 1, 2]]
newQ = Q.copy(); newQ[:, io], newQ[:, ii] = q_o, q_i


def set_col(t, name, value):
    i = t.column_names.index(name)
    return t.set_column(i, t.schema.field(i), pa.array([value], type=t.schema.field(i).type))


t = set_col(t, "object_body_position", newP.tolist()); t = set_col(t, "object_body_wxyz", newQ.tolist())
t = set_col(t, "object_root_position", newP[:, 0].tolist())
t = set_col(t, "object_root_axis_angle", R.concatenate(outR if io == 0 else inR).as_rotvec().tolist())
pq.write_table(t, p)
ov = []
for f in range(0, T, 2):
    Rrel = outR[f].inv() * inR[f]
    trel = outR[f].inv().apply(inP[f] - outP[f])
    pin = Rrel.apply(Oi["pts"]) + trel
    ov.append(depth(pin, Oo["planes"]))
rep = {"nesting_offset_cm": round(100 * dz, 2), "interp": [i0, i1], "inner_leads": [c0, c1], "outer_leads": [e0, e1],
       "max_overlap_after_cm": round(100 * max(ov), 2), "static_fit": fit_rep}
rj = OUT / f"ep{a.ep}_report.json"
if rj.exists():
    R_ = json.loads(rj.read_text()); R_["nest_fix"] = rep; rj.write_text(json.dumps(R_, indent=1) + "\n")
print(json.dumps(rep))
