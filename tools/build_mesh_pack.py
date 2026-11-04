"""Packing-time mesh set: meshes_final, with union-cleaned meshes (tools/mesh_clean.py --mode union) for the objects
where the local CD-O board says cleaning helps. Cleaning keeps the body frame, so the trajectories tracked with
meshes_final stay valid; only the packed submission meshes change. The tracking assets are not touched.

  python build_mesh_pack.py OUT_DIR
Writes OUT_DIR/<name>.{obj,glb} for every object and OUT_DIR/manifest.json (source, sha256, body-frame check: the
max distance from each cleaned vertex to the original surface and the bounds-centre shift).
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

E2 = Path.home() / "v2d/ego2sim"
FINAL, CLEAN = E2 / "meshes_final", E2 / "meshes_clean_union"
# board (2026-10-27): sweeper 0.333 -> 0.307, pot 0.263 -> 0.261, wooden_piece_2 0.188 -> 0.183; lid worse, spoon unscored
CLEANED = {"mini_sweeper", "white_pot", "wooden_piece_2"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    man = {}
    for f in sorted(FINAL.glob("*.obj")):
        n = f.stem
        src = CLEAN if n in CLEANED else FINAL
        for ext in ("obj", "glb"):
            shutil.copyfile(src / f"{n}.{ext}", out / f"{n}.{ext}")
        e = {"source": str(src / n), "sha256_obj": sha(out / f"{n}.obj"), "sha256_glb": sha(out / f"{n}.glb")}
        if n in CLEANED:
            a = trimesh.load(f, force="mesh")
            b = trimesh.load(out / f"{n}.glb", force="mesh")
            _, d, _ = trimesh.proximity.closest_point(a, b.vertices)
            e.update(max_vertex_off_surface_cm=round(float(d.max()) * 100, 3),
                     bounds_centre_shift_cm=round(float(np.linalg.norm(a.bounds.mean(0) - b.bounds.mean(0))) * 100, 3))
        man[n] = e
        print(n, {k: v for k, v in e.items() if not k.startswith("sha")})
    json.dump(man, open(out / "manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
