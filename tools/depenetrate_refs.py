"""Stage 3 of the reference cleaning (after clean_refs_v2.py stage2): push apart objects whose collision shapes overlap
in the reference. In the simulator overlapping objects either lock together (ep1 pot + lid "fused") or are forced apart
(ep10 nested cups "not fitting"). Per frame and object pair, the depth of one object's surface inside the other's
collision hulls (the CoACD hulls FlashCHORD builds, dumped by collision_hulls.py) is measured; above --tol-cm the smaller
object is moved by the shortest push that clears it, over candidate directions (centroid line, world up, both ways along
each object's principal axes; bisection, at most --max-cm). Frames that no direction clears within --max-cm are left as
they are. Offsets are smoothed over time (window max, then box mean). Rotations, hands and robot
are untouched. Edits the cleaned package in place; the pre-stage-3 parquet is kept as OUT/epEP_stage2.parquet.
  python depenetrate_refs.py OUT_DIR HULL_DIR EP [EP ...] [--tol-cm 0.15] [--max-cm 2] [--window 5]   (meshclean venv)
OUT_DIR is clean_refs_v2.py's output dir (package/ inside it)."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("hulls"); ap.add_argument("eps", nargs="+")
ap.add_argument("--tol-cm", type=float, default=0.15); ap.add_argument("--max-cm", type=float, default=2.0)
ap.add_argument("--window", type=int, default=5)
ap.add_argument("--mover", help="object that moves (default: the smaller of each pair)")
ap.add_argument("--direction", choices=["any", "up", "axis"], default="any",
                help="up: world +z only; axis: the other object's symmetry axis, its upward end (nested cups, "
                     "whose tracks can coincide); any: shortest of all candidates")
a = ap.parse_args()
OUT = Path(a.out); PKG = OUT / "package"
_obj = {}


def obj(name):
    """Surface samples, centroid (body frame), hull face planes (normals, offsets), surface area."""
    if name not in _obj:
        d = np.load(Path(a.hulls) / f"{name}.npz")
        m = trimesh.Trimesh(d["mesh_v"], d["mesh_f"], process=False)
        planes = []
        for i in range(int(d["n"])):
            h = trimesh.Trimesh(d[f"v{i}"], d[f"f{i}"], process=False)
            n = h.face_normals
            planes.append((n, (h.triangles[:, 0] * n).sum(1)))
        pts = np.array(trimesh.sample.sample_surface(m, 1200, seed=0)[0])
        w, axes = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
        gaps = [abs(w[1] - w[2]), abs(w[0] - w[2]), abs(w[0] - w[1])]  # symmetry axis as in clean_refs.py
        _obj[name] = dict(pts=pts, c=np.array(m.centroid), axes=axes, sym=axes[:, int(np.argmin(gaps))].copy(),
                          planes=planes, area=float(m.area))
    return _obj[name]


def depth(pts_local, planes):
    """Max penetration (m) of points into the union of convex hulls given by face planes."""
    best = 0.0
    for n, d in planes:
        s = pts_local @ n.T - d
        ins = (s <= 0).all(1)
        if ins.any():
            best = max(best, float((-s[ins].max(1)).max()))
    return best


def pair_depth(A, PA, RA, B, PB, RB):
    """Symmetric: A's surface inside B's hulls and B's surface inside A's hulls."""
    wa = RA.apply(A["pts"]) + PA
    wb = RB.apply(B["pts"]) + PB
    return max(depth(RB.inv().apply(wa - PB), B["planes"]), depth(RA.inv().apply(wb - PA), A["planes"]))


def set_col(t, name, value):
    i = t.column_names.index(name)
    return t.set_column(i, t.schema.field(i), pa.array([value], type=t.schema.field(i).type))


for ep in a.eps:
    p = next((PKG / f"sim_reference/processed/sequence_id=ep{ep}/robot_name=sharpa_wave").glob("*.parquet"))
    t = pq.read_table(p); r = t.to_pylist()[0]
    names = [str(n) for n in r["object_body_names"]]
    if len(names) < 2:
        print(f"ep{ep}: single object, nothing to do"); continue
    P = np.asarray(r["object_body_position"], float); Q = np.asarray(r["object_body_wxyz"], float)
    T = len(P)
    rots = [R.from_quat(Q[:, b][:, [1, 2, 3, 0]]) for b in range(len(names))]
    O = [obj(n) for n in names]
    off = np.zeros_like(P)
    rep = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mv, st = (i, j) if O[i]["area"] <= O[j]["area"] else (j, i)  # the smaller object moves
            if a.mover in (names[i], names[j]):
                mv, st = (i, j) if names[i] == a.mover else (j, i)
            before, after, moved, unresolved = [], [], [], 0
            for f in range(T):
                Rm, Rs = rots[mv][f], rots[st][f]
                pm, ps = P[f, mv] + off[f, mv], P[f, st] + off[f, st]
                d0 = pair_depth(O[mv], pm, Rm, O[st], ps, Rs)
                before.append(d0)
                if d0 <= a.tol_cm / 100:
                    after.append(d0); moved.append(0.0); continue
                # candidate push directions: centroid line, world up, both ways along each object's principal axes
                # (nested cups separate along the cup axis, a lid lifts straight up); shortest clearing push wins
                u0 = (Rm.apply(O[mv]["c"]) + pm) - (Rs.apply(O[st]["c"]) + ps)
                if a.direction == "up":
                    cands = [np.array([0.0, 0.0, 1.0])]
                elif a.direction == "axis":
                    ax = Rs.apply(O[st]["sym"])
                    cands = [(np.sign(ax[2]) or 1.0) * ax]  # the upward end of the axis (upright cups; tracks may coincide)
                else:
                    cands = [u0 / np.linalg.norm(u0)] if np.linalg.norm(u0) > 1e-6 else []
                    cands.append(np.array([0.0, 0.0, 1.0]))
                    for Rx, k in ((Rs, st), (Rm, mv)):
                        for ax in O[k]["axes"].T:
                            cands += [Rx.apply(ax), -Rx.apply(ax)]
                best = None
                for u in cands:
                    hi = a.max_cm / 100
                    if pair_depth(O[mv], pm + hi * u, Rm, O[st], ps, Rs) > a.tol_cm / 100:
                        continue  # cannot clear along u within max-cm
                    lo = 0.0
                    for _ in range(12):
                        mid = (lo + hi) / 2
                        if pair_depth(O[mv], pm + mid * u, Rm, O[st], ps, Rs) > a.tol_cm / 100:
                            lo = mid
                        else:
                            hi = mid
                    if best is None or hi < best[0]:
                        best = (hi, u)
                if best is None:  # interlocked beyond max-cm in every direction: leave the frame as it is
                    unresolved += 1; moved.append(0.0); continue
                off[f, mv] += best[0] * best[1]
                moved.append(best[0])
            rep[f"{names[mv]} out of {names[st]}"] = dict(
                frames_overlapping=int((np.array(before) > a.tol_cm / 100).sum()),
                max_overlap_cm=round(100 * max(before), 2), max_shift_cm=round(100 * max(moved), 2),
                frames_left_unresolved=unresolved)
    # smooth offsets in time: largest offset within the window, then a box mean (keeps clearance, removes jitter)
    w = a.window // 2
    sm = np.zeros_like(off)
    for b in range(len(names)):
        nrm = np.linalg.norm(off[:, b], axis=1)
        peak = np.stack([off[max(0, f - w) + int(np.argmax(nrm[max(0, f - w):f + w + 1])), b] for f in range(T)])
        sm[:, b] = np.stack([peak[max(0, f - w):f + w + 1].mean(0) for f in range(T)])
    Pn = P + sm
    resid = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dd = [pair_depth(O[i], Pn[f, i], rots[i][f], O[j], Pn[f, j], rots[j][f]) for f in range(T)]
            resid.append(f"{names[i]}/{names[j]} max overlap after {100 * max(dd):.2f} cm, "
                         f"{int((np.array(dd) > a.tol_cm / 100).sum())} frames > tol")
    bak = OUT / f"ep{ep}_stage2.parquet"
    if not bak.exists():
        shutil.copyfile(p, bak)
    t = set_col(t, "object_body_position", Pn.tolist()); t = set_col(t, "object_root_position", Pn[:, 0].tolist())
    pq.write_table(t, p)
    rj = OUT / f"ep{ep}_report.json"
    if rj.exists():
        R_ = json.loads(rj.read_text()); R_["stage3_depenetrate"] = rep; rj.write_text(json.dumps(R_, indent=1) + "\n")
    print(f"ep{ep}: " + "; ".join(f"{k}: {v['frames_overlapping']} fr, max {v['max_overlap_cm']} cm, shift <= {v['max_shift_cm']} cm, "
                                  f"{v['frames_left_unresolved']} unresolved"
                                  for k, v in rep.items()) + " | " + "; ".join(resid), flush=True)
