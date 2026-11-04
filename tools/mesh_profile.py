"""Size profile of rotationally symmetric-ish objects (cups, pot, lid): height along the principal symmetry axis and
the outer radius at several heights, for our mesh vs a reference scan, both in their own frames (cm).
  python mesh_profile.py NAME=MESH [NAME=MESH ...] [--bins 6]"""
import argparse

import numpy as np
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("meshes", nargs="+"); ap.add_argument("--bins", type=int, default=6)
a = ap.parse_args()
for spec in a.meshes:
    lab, path = spec.split("=", 1)
    m = trimesh.load(path, force="mesh")
    p = trimesh.sample.sample_surface(m, 60000, seed=0)[0]
    c = p.mean(0); w, v = np.linalg.eigh(np.cov((p - c).T))
    gaps = [abs(w[1] - w[2]), abs(w[0] - w[2]), abs(w[0] - w[1])]
    ax = v[:, int(np.argmin(gaps))]
    h = (p - c) @ ax
    r = np.linalg.norm((p - c) - np.outer(h, ax), axis=1)
    lo, hi = h.min(), h.max()
    edges = np.linspace(lo, hi, a.bins + 1)
    prof = [np.percentile(r[(h >= edges[k]) & (h < edges[k + 1])], 98) * 200 for k in range(a.bins)]
    print(f"{lab:24} height {100 * (hi - lo):5.1f} cm, outer diameter by height (low->high along axis): "
          + " ".join(f"{x:5.1f}" for x in prof))
