"""Smallest uniform shrink of one object's collision shape (about its centroid; visual mesh untouched) that removes its
overlap with the other objects of the given episodes along the reference poses.
  python collision_fit.py PKG HULL_DIR OBJ EP [EP ...] [--tol-cm 0.15] [--every 2]     (meshclean venv)
Scaling the hulls about the centroid c by s = URDF <collision> scale s with origin xyz (1 - s) * c."""
import argparse
import glob
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("pkg"); ap.add_argument("hulls"); ap.add_argument("obj"); ap.add_argument("eps", nargs="+")
ap.add_argument("--tol-cm", type=float, default=0.15); ap.add_argument("--every", type=int, default=2)
a = ap.parse_args()


def load(name):
    d = np.load(Path(a.hulls) / f"{name}.npz")
    m = trimesh.Trimesh(d["mesh_v"], d["mesh_f"], process=False)
    hv = [(np.array(d[f"v{i}"]), np.array(d[f"f{i}"])) for i in range(int(d["n"]))]
    return dict(pts=np.array(trimesh.sample.sample_surface(m, 1500, seed=0)[0]), c=np.array(m.centroid), hulls=hv)


def planes(hv, c, s):
    out = []
    for v, f in hv:
        h = trimesh.Trimesh(c + s * (v - c), f, process=False)
        n = h.face_normals; out.append((n, (h.triangles[:, 0] * n).sum(1)))
    return out


def depth(p, pl):
    best = 0.0
    for n, d in pl:
        sgn = p @ n.T - d; ins = (sgn <= 0).all(1)
        if ins.any():
            best = max(best, float((-sgn[ins].max(1)).max()))
    return best


X = load(a.obj)
for ep in a.eps:
    p = glob.glob(f"{a.pkg}/sim_reference/processed/sequence_id=ep{ep}/robot_name=sharpa_wave/*.parquet")[0]
    r = pq.read_table(p, columns=["object_body_names", "object_body_position", "object_body_wxyz"]).to_pylist()[0]
    names = list(r["object_body_names"]); P = np.array(r["object_body_position"]); Q = np.array(r["object_body_wxyz"])
    if a.obj not in names:
        continue
    i = names.index(a.obj)
    for j, other in enumerate(names):
        if j == i:
            continue
        Y = load(other); plY = planes(Y["hulls"], Y["c"], 1.0)
        res = []
        for s in (1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88):
            plX = planes(X["hulls"], X["c"], s); ptsX = X["c"] + s * (X["pts"] - X["c"])
            worst, nbad = 0.0, 0
            for f in range(0, len(P), a.every):
                Ri, Rj = R.from_quat(Q[f, i][[1, 2, 3, 0]]), R.from_quat(Q[f, j][[1, 2, 3, 0]])
                wx = Ri.apply(ptsX) + P[f, i]; wy = Rj.apply(Y["pts"]) + P[f, j]
                dd = max(depth(Rj.inv().apply(wx - P[f, j]), plY), depth(Ri.inv().apply(wy - P[f, i]), plX))
                worst = max(worst, dd); nbad += dd > a.tol_cm / 100
            res.append(f"s={s:.2f}: max {100 * worst:.2f} cm, {nbad} frames")
            if worst <= a.tol_cm / 100:
                break
        print(f"ep{ep} {a.obj} vs {other} (centroid {np.round(100 * X['c'], 1)} cm): " + " | ".join(res), flush=True)
