"""Spin vs tilt of rotationally symmetric objects in reference packages: per frame-to-frame rotation, the twist about
the mesh's symmetry axis (unobservable spin) and the swing of the axis itself (real tilting), in deg/frame.
  python cup_spin.py LABEL=PKG [LABEL=PKG ...] [--objs blue_cup,beige_cup]"""
import argparse
import glob
import re

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("pkgs", nargs="+")
ap.add_argument("--objs", default="blue_cup,beige_cup")
a = ap.parse_args()
objs = a.objs.split(",")


def sym_axis(pkg, name):  # same choice as clean_refs.py: the eigenvector whose eigenvalue is unlike the other two
    m = trimesh.load(f"{pkg}/assets/{name}.obj", process=False, force="mesh")
    p = trimesh.sample.sample_surface(m, 20000, seed=0)[0]
    w, v = np.linalg.eigh(np.cov((p - p.mean(0)).T))
    gaps = [abs(w[1] - w[2]), abs(w[0] - w[2]), abs(w[0] - w[1])]
    return v[:, int(np.argmin(gaps))]


rows = {}
for spec in a.pkgs:
    lab, pkg = spec.split("=", 1)
    for p in sorted(glob.glob(f"{pkg}/sim_reference/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet")):
        ep = int(re.search(r"sequence_id=ep(\d+)", p).group(1))
        r = pq.read_table(p, columns=["object_body_names", "object_body_wxyz"]).to_pylist()[0]
        for i, n in enumerate(r["object_body_names"]):
            if n not in objs:
                continue
            ax = sym_axis(pkg, n)
            q = np.array(r["object_body_wxyz"])[:, i]
            rot = R.from_quat(q[:, [1, 2, 3, 0]])
            d = (rot[:-1].inv() * rot[1:]).as_quat()  # body-frame step: twist about ax, swing the rest
            tw = np.c_[(d[:, :3] @ ax)[:, None] * ax, d[:, 3]]
            tw /= np.linalg.norm(tw, axis=1, keepdims=True)
            twist = np.degrees(R.from_quat(tw).magnitude())
            swing = np.degrees((R.from_quat(tw).inv() * R.from_quat(d)).magnitude())
            total_spin = abs(np.degrees(np.sum(np.sign((d[:, :3] @ ax) * d[:, 3]) * np.radians(twist))))
            rows.setdefault((ep, n), {})[lab] = (np.median(twist), np.percentile(twist, 95), np.percentile(swing, 95), total_spin)
labs = [s.split("=", 1)[0] for s in a.pkgs]
print(f"{'ep':>3} {'object':10} " + "".join(f"{l + ': spin med/p95, tilt p95 (deg/fr), net spin (deg)':>62}" for l in labs))
for (ep, n), d in sorted(rows.items()):
    print(f"{ep:>3} {n:10} " + "".join(f"{'%6.2f /%6.2f   %6.2f   %8.0f' % d[l]:>62}" if l in d else f"{'-':>62}" for l in labs))
