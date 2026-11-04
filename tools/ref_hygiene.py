"""Per object of a reference package: rotation speed (deg/frame, p95 and max) and frame-0 lowest point vs the table top
(cm; negative = starts inside the table). What clean_refs.py (cup spin / flips / table / settle) would act on.
  python ref_hygiene.py LABEL=PKG_DIR [LABEL=PKG_DIR ...] --eps 8,10,32,33"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("pkgs", nargs="+")
ap.add_argument("--eps", default="8,10,32,33")
ap.add_argument("--table", type=float, default=0.75)
a = ap.parse_args()
print(f"{'ep':>3} {'object':18} " + " ".join(f"{l.split('=')[0]:>22}" for l in a.pkgs))
print(f"{'':22}" + " ".join(f"{'rot p95/max  f0 cm':>22}" for _ in a.pkgs))
for ep in a.eps.split(","):
    rows = {}
    for spec in a.pkgs:
        lab, pkg = spec.split("=", 1)
        p = glob.glob(f"{pkg}/sim_reference/processed/sequence_id=ep{ep}/robot_name=sharpa_wave/*.parquet")[0]
        r = pq.read_table(p, columns=["object_body_names", "object_body_position", "object_body_wxyz"]).to_pylist()[0]
        pos, q = np.array(r["object_body_position"]), np.array(r["object_body_wxyz"])
        for i, n in enumerate(r["object_body_names"]):
            rot = R.from_quat(q[:, i][:, [1, 2, 3, 0]])
            d = np.degrees((rot[:-1].inv() * rot[1:]).magnitude())
            v = trimesh.load(f"{pkg}/assets/{n}.obj", process=False, force="mesh").vertices
            low = (rot[0].apply(v) + pos[0, i])[:, 2].min()
            rows.setdefault(n, []).append(f"{np.percentile(d, 95):6.1f} /{d.max():6.1f}  {100 * (low - a.table):+5.1f}")
    for n, cells in rows.items():
        print(f"{ep:>3} {n:18} " + " ".join(f"{c:>22}" for c in cells))
