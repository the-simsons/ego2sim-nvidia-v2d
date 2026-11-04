"""Dump the simulator's collision hulls for objects: the URDF collision mesh, CoACD-decomposed with FlashCHORD's own
convex_decompose (same defaults as the scene builder), written to OUT/<obj>.npz (n, verts_i, faces_i ...).
  python collision_hulls.py ASSETS_DIR OUT_DIR OBJ [OBJ ...] [--threshold 0.05]     (flash_chord venv)"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from flash_chord.assets.mesh import convex_decompose

ap = argparse.ArgumentParser()
ap.add_argument("assets"); ap.add_argument("out"); ap.add_argument("objs", nargs="+")
ap.add_argument("--threshold", type=float, default=0.05)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for n in a.objs:
    urdf = Path(a.assets) / f"{n}_rigid.urdf"
    col = ET.parse(urdf).getroot().find(".//collision/geometry/mesh")
    m = trimesh.load(urdf.parent / col.get("filename"), force="mesh", process=False)
    if col.get("scale"):
        m.apply_scale([float(x) for x in col.get("scale").split()])
    hulls = convex_decompose(m.vertices, m.faces, threshold=a.threshold)
    d = {"n": len(hulls), "mesh_v": m.vertices, "mesh_f": m.faces}
    for i, h in enumerate(hulls):
        d[f"v{i}"], d[f"f{i}"] = h.vertices, h.faces.reshape(-1, 3)
    np.savez(out / f"{n}.npz", **d)
    print(f"{n}: {len(hulls)} hulls -> {out / (n + '.npz')}")
