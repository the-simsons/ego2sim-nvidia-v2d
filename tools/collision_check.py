"""What the simulator's collision geometry does with our meshes: FlashCHORD replaces each object's collision mesh with a
CoACD convex decomposition (flash_chord.assets.mesh.convex_decompose, its defaults). Per object: hull count and how much
of the empty space inside the object's convex hull the collision hulls fill (a filled cup cannot hold another cup).
Per object pair: how deep their collision shapes interpenetrate in the reference, at frame 0 and worst over the clip.
  python collision_check.py PKG EP [EP ...] [--every 5] [--threshold 0.05]
Hulls come from collision_hulls.py (flash_chord venv) in HULL_DIR; this runs in the meshclean venv (trimesh + embree).
  python collision_check.py PKG HULL_DIR EP [EP ...] [--every 5]"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("pkg"); ap.add_argument("hulls"); ap.add_argument("eps", nargs="+")
ap.add_argument("--every", type=int, default=5)
a = ap.parse_args()
pkg = Path(a.pkg)
_cache = {}


def collision(name):
    """Collision mesh and its CoACD hulls (as trimesh), from collision_hulls.py's dump."""
    if name not in _cache:
        d = np.load(Path(a.hulls) / f"{name}.npz")
        m = trimesh.Trimesh(d["mesh_v"], d["mesh_f"], process=False)
        _cache[name] = (m, [trimesh.Trimesh(d[f"v{i}"], d[f"f{i}"], process=False) for i in range(int(d["n"]))])
    return _cache[name]
    urdf = pkg / "assets" / f"{name}_rigid.urdf"
    col = ET.parse(urdf).getroot().find(".//collision/geometry/mesh")
    m = trimesh.load(urdf.parent / col.get("filename"), force="mesh", process=False)
    s = col.get("scale")
    if s:
        m.apply_scale([float(x) for x in s.split()])
    hulls = [trimesh.Trimesh(h.vertices, h.faces, process=False) for h in
             convex_decompose(m.vertices, m.faces, threshold=a.threshold)]
    _cache[name] = (m, hulls)
    return m, hulls


def inside_any(hulls, p):
    out = np.zeros(len(p), bool)
    for h in hulls:  # convex: inside iff on the inner side of every face plane
        n = h.face_normals; d = (h.triangles[:, 0] * n).sum(1)
        out |= ((p @ n.T) - d <= 0).all(1)
    return out


def depth_inside(hulls, p):
    """Penetration depth of points into the union of convex hulls (max over hulls of min distance to a face plane)."""
    best = np.zeros(len(p))
    for h in hulls:
        n = h.face_normals; d = (h.triangles[:, 0] * n).sum(1)
        s = (p @ n.T) - d  # <= 0 on every plane = inside
        dep = np.where((s <= 0).all(1), -s.max(1), 0.0)
        best = np.maximum(best, dep)
    return best


rng = np.random.default_rng(0)
for ep in a.eps:
    p = glob.glob(str(pkg / f"sim_reference/processed/sequence_id=ep{ep}/robot_name=sharpa_wave/*.parquet"))[0]
    r = pq.read_table(p, columns=["object_body_names", "object_body_position", "object_body_wxyz"]).to_pylist()[0]
    names = list(r["object_body_names"]); P = np.array(r["object_body_position"]); Q = np.array(r["object_body_wxyz"])
    print(f"== ep{ep}: {names}")
    surf = {}
    for n in names:
        m, hulls = collision(n)
        ch = m.convex_hull
        box = rng.uniform(ch.bounds[0], ch.bounds[1], (200000, 3)); pts = box[inside_any([ch], box)][:40000]  # inside its hull
        solid = m.contains(pts) if m.is_watertight else None
        inh = inside_any(hulls, pts)
        if solid is not None:
            empty = ~solid
            fill = inh[empty].mean() if empty.any() else 0.0
            msg = f"empty space inside its hull {empty.mean():.0%} of hull volume, collision hulls fill {fill:.0%} of it"
        else:
            msg = f"mesh not watertight; collision hulls fill {inh.mean():.0%} of its convex hull"
        print(f"  {n}: {len(hulls)} CoACD hulls; {msg}")
        surf[n] = trimesh.sample.sample_surface(m, 4000, seed=0)[0]
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            hj = collision(names[j])[1]
            worst, wt, d0 = 0.0, 0, None
            for t in range(0, len(P), a.every):
                Ri, Rj = R.from_quat(Q[t, i][[1, 2, 3, 0]]), R.from_quat(Q[t, j][[1, 2, 3, 0]])
                w = Ri.apply(surf[names[i]]) + P[t, i]           # i's surface in the world
                local = Rj.inv().apply(w - P[t, j])               # ... in j's body frame
                dep = depth_inside(hj, local).max() * 100
                if t == 0:
                    d0 = dep
                if dep > worst:
                    worst, wt = dep, t
            print(f"  {names[i]} inside {names[j]}'s collision: frame 0 {d0:.1f} cm, worst {worst:.1f} cm at frame {wt}")
