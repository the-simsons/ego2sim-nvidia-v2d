"""Clean a reconstructed mesh before packing: drop duplicate / degenerate faces, union overlapping closed parts
(manifold3d), and remove faces hidden inside the object (no ray from the face, on either side, escapes the mesh).
CD-O normalises by the surface centroid and RMS radius, so hidden faces shift both and attract sample points.

  python mesh_clean.py IN OUT [--mode union|visible|both] [--dirs 96]
union:   boolean union of the watertight parts (non-watertight parts are kept as they are), then hidden-face removal
         on the result (catches open parts buried in the union). visible: hidden-face removal only. both = union.
Writes OUT (.obj/.glb) and prints face counts and the radius change. Run with envs/meshclean.
"""
import argparse

import numpy as np
import trimesh


def surface_moments(v, f):
    """CD-O.py _surface_moments: area-weighted centroid and RMS radius of the surface."""
    t = v[f]
    a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    k = a > 1e-12
    t, a = t[k], a[k]
    c = (t.mean(1) * a[:, None]).sum(0) / a.sum()
    x = t - c
    sq = np.einsum("tij,tij->ti", x, x).sum(1)
    cr = (np.einsum("ti,ti->t", x[:, 0], x[:, 1]) + np.einsum("ti,ti->t", x[:, 0], x[:, 2])
          + np.einsum("ti,ti->t", x[:, 1], x[:, 2]))
    return c, float(np.sqrt((a * (sq + cr) / 6.0).sum() / a.sum()))


def exposed_faces(m, dirs=96, seed=0):
    """True for faces from which, on at least one side, some ray escapes the mesh (no hit)."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(dirs, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    eps = 1e-4 * m.scale
    cen, nrm = m.triangles_center, m.face_normals
    out = np.zeros(len(m.faces), bool)
    ray = m.ray
    for side in (1.0, -1.0):
        todo = np.where(~out)[0]
        for k in range(dirs):
            if not len(todo):
                break
            dk = d[k] * np.where((nrm[todo] @ d[k])[:, None] * side >= 0, 1.0, -1.0)  # into this side's hemisphere
            o = cen[todo] + side * eps * nrm[todo]
            hit = ray.intersects_any(o, dk)
            out[todo[~hit]] = True
            todo = todo[hit]
    return out


def union_parts(m):
    import manifold3d as mf
    parts = m.split(only_watertight=False)
    closed = [p for p in parts if p.is_watertight and p.volume > 0]
    opened = [p for p in parts if not (p.is_watertight and p.volume > 0)]
    if len(closed) < 2:
        return m, len(closed), len(opened)
    acc = None
    for p in closed:
        q = mf.Manifold(mf.Mesh(vert_properties=np.asarray(p.vertices, np.float32), tri_verts=np.asarray(p.faces, np.uint32)))
        acc = q if acc is None else acc + q
    o = acc.to_mesh()
    u = trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)
    return trimesh.util.concatenate([u] + opened), len(closed), len(opened)


def clean(m, mode="union", dirs=96):
    w = trimesh.Trimesh(m.vertices, m.faces, process=True)
    w.update_faces(w.nondegenerate_faces())
    w.update_faces(w.unique_faces())
    w.remove_unreferenced_vertices()
    info = {"faces_in": len(m.faces)}
    if mode in ("union", "both"):
        w, info["closed_parts"], info["open_parts"] = union_parts(w)
    vis = exposed_faces(w, dirs)
    info["hidden_removed"] = int((~vis).sum())
    w.update_faces(vis)
    w.remove_unreferenced_vertices()
    info["faces_out"] = len(w.faces)
    return w, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--mode", default="union", choices=["union", "visible", "both"])
    ap.add_argument("--dirs", type=int, default=96)
    a = ap.parse_args()
    m = trimesh.load(a.inp, force="mesh", process=False)
    _, r0 = surface_moments(np.asarray(m.vertices, float), np.asarray(m.faces))
    c, info = clean(m, a.mode, a.dirs)
    _, r1 = surface_moments(c.vertices, c.faces)
    c.export(a.out)
    print(f"{a.inp.split('/')[-1]} -> {a.out}: {info}, watertight {c.is_watertight}, RMS radius {r0 * 100:.2f} -> "
          f"{r1 * 100:.2f} cm, extents cm {np.round(c.extents * 100, 1)}")


if __name__ == "__main__":
    main()
