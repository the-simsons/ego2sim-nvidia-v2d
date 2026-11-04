"""Clean a mesh extracted from an object-only splat (black background): drop faces whose vertex colours are near black
(background floaters), keep the largest connected component(s), write OBJ (+ GLB with colours).
  python atlas_mesh_clean.py IN.glb OUT_BASENAME [--black 0.12] [--keep 0.9]
--keep: keep components (largest first) until this fraction of the remaining surface area is covered."""
import argparse

import numpy as np
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--black", type=float, default=0.12); ap.add_argument("--keep", type=float, default=0.9)
a = ap.parse_args()
m = trimesh.load(a.inp, force="mesh", process=True)
print(f"in: {len(m.vertices)} v, {len(m.faces)} f, bounds {np.ptp(m.bounds, 0).round(3)}")
try:
    vc = m.visual.to_color().vertex_colors[:, :3] / 255.0
    dark = vc.max(1) < a.black
    fdark = dark[m.faces].all(1)
    m.update_faces(~fdark); m.remove_unreferenced_vertices()
    print(f"dropped {int(fdark.sum())} near-black faces")
except Exception as e:  # untextured: nothing to drop
    print(f"no vertex colours ({e}); colour filter skipped")
parts = sorted(m.split(only_watertight=False), key=lambda p: -p.area)
tot, kept, acc = sum(p.area for p in parts), [], 0.0
for p in parts:
    kept.append(p); acc += p.area
    if acc >= a.keep * tot:
        break
out = trimesh.util.concatenate(kept)
print(f"components {len(parts)} -> kept {len(kept)} ({acc / tot:.0%} of area); out {len(out.vertices)} v, {len(out.faces)} f, "
      f"bounds {np.ptp(out.bounds, 0).round(3)}")
out.export(a.out + ".obj")
out.export(a.out + ".glb")
