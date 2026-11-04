"""Raw vs cleaned reference for one object: per frame lowest point vs table, rotation speed, and cleaned-vs-raw angle.
  python clean_diff.py RAW_PKG CLEAN_PKG EP OBJ [--every 10] [--table 0.75]"""
import argparse
import glob

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("clean"); ap.add_argument("ep"); ap.add_argument("obj")
ap.add_argument("--every", type=int, default=10); ap.add_argument("--table", type=float, default=0.75)
a = ap.parse_args()


def load(pkg):
    p = glob.glob(f"{pkg}/sim_reference/processed/sequence_id=ep{a.ep}/robot_name=sharpa_wave/*.parquet")[0]
    r = pq.read_table(p).to_pylist()[0]
    i = list(r["object_body_names"]).index(a.obj)
    q = np.array(r["object_body_wxyz"])[:, i]
    return np.array(r["object_body_position"])[:, i], R.from_quat(q[:, [1, 2, 3, 0]])


v = trimesh.load(f"{a.raw}/assets/{a.obj}.obj", process=False, force="mesh").vertices
(P0, R0), (P1, R1) = load(a.raw), load(a.clean)
low = lambda P, Rr, t: 100 * ((Rr[t].apply(v) + P[t])[:, 2].min() - a.table)
d0 = np.r_[0, np.degrees((R0[1:] * R0[:-1].inv()).magnitude())]
d1 = np.r_[0, np.degrees((R1[1:] * R1[:-1].inv()).magnitude())]
ang = np.degrees((R1 * R0.inv()).magnitude())
print(f"ep{a.ep} {a.obj}: frame  low_raw low_clean (cm)  rot/frame raw clean (deg)  clean-vs-raw (deg)  dpos (cm)")
for t in range(0, len(P0), a.every):
    print(f"  {t:4d}  {low(P0, R0, t):+7.1f} {low(P1, R1, t):+7.1f}   {d0[t]:7.1f} {d1[t]:6.1f}   {ang[t]:7.1f}   {100 * np.linalg.norm(P1[t] - P0[t]):6.1f}")
