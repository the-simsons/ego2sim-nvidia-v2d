"""How low can an honest CD-O go? Score re-tessellated copies of the public scan against the packed scan
with the kit's own CD-O code, at Kaggle's seeds. A copy with the same surface but different triangles
is the best a reconstruction can do; the packed scan itself scores ~0.
  cd_o_floor.py KIT_ROOT OBJECT SCAN_MESH [FACES ...]
Run with envs/kit (trimesh, fast_simplification)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


kit, obj, scan = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
faces = [int(f) for f in sys.argv[4:]] or [500, 1000, 2000, 4000]
sys.path.insert(0, str(kit))
from v2dlb.mesh_budget import budget_mesh  # noqa: E402
cdo = load(kit / "metric_code/track_3/CD-O.py", "cdo")
roster = json.load(open(kit / "data/track_3_evaluation_objects.json"))["episodes"]
slots = [(int(e), names.index(obj)) for e, names in roster.items() if obj in names]

m = trimesh.load(scan, force="mesh")
ref = budget_mesh(scan, 4096, 4096)
ref = (np.asarray(ref[0], float), np.asarray(ref[1], np.int64))


def score(path):
    v, f = budget_mesh(path, 4096, 4096)
    cand = (np.asarray(v, float), np.asarray(f, np.int64))
    return np.mean([cdo._object_chamfer_cm(ref, cand, cdo._object_seed(e, s)) for e, s in slots])


tmp = Path("/tmp") / f"cdo_floor_{obj}"; tmp.mkdir(exist_ok=True)
print(f"{obj}: scan {len(m.faces)} faces, {len(slots)} eval pairs")
print(f"  packed scan itself      : {score(scan):.3f} cm")
for n in faces:
    d = m.simplify_quadric_decimation(face_count=n)
    p = tmp / f"dec{n}.obj"; d.export(p)
    print(f"  scan decimated to {n:5d}: {score(p):.3f} cm")
# same surface, different triangles AND slightly noisy: jitter vertices by 1 mm
j = m.copy(); j.vertices += np.random.default_rng(0).normal(0, 0.001 * m.scale, j.vertices.shape)
p = tmp / "jitter.obj"; j.export(p)
print(f"  scan, 0.1% scale jitter  : {score(p):.3f} cm")
