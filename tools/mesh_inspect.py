"""Mesh internals: connected parts, watertightness, duplicate faces, and how much surface is hidden inside (never
reached by a ray from outside), plus the CD-O normalisation (surface centroid, RMS radius) with and without hidden faces.

  python mesh_inspect.py MESH [MESH ...]
Run with envs/meshclean.
"""
import sys

import numpy as np
import trimesh

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mesh_clean import exposed_faces, surface_moments  # noqa: E402


def main():
    for p in sys.argv[1:]:
        m = trimesh.load(p, force="mesh", process=False)
        w = trimesh.Trimesh(m.vertices, m.faces, process=True)
        parts = w.split(only_watertight=False)
        key = np.sort(w.faces, axis=1)
        dup = len(key) - len(np.unique(key, axis=0))
        vis = exposed_faces(w)
        area = w.area_faces
        c0, r0 = surface_moments(w.vertices, w.faces)
        c1, r1 = surface_moments(w.vertices, w.faces[vis])
        print(f"{p.split('/')[-2]}/{p.split('/')[-1]}: {len(w.faces)} faces, {len(parts)} parts "
              f"({sum(q.is_watertight for q in parts)} watertight), whole watertight {w.is_watertight}, dup faces {dup}")
        print(f"   hidden faces {np.sum(~vis)} ({100 * area[~vis].sum() / area.sum():.1f}% of area); RMS radius "
              f"{r0 * 100:.2f} -> {r1 * 100:.2f} cm visible-only ({100 * (r0 / r1 - 1):+.1f}%), centroid shift "
              f"{np.linalg.norm(c0 - c1) * 100:.2f} cm")


if __name__ == "__main__":
    main()
