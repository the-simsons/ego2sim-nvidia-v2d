"""Copy of a reference package with one object's orientation turned 180 deg about one of its mesh principal axes through
the mesh centroid, in every frame of one episode (a look-alike flip: same silhouette, same centroid path). Everything
else (other objects, hands, robot, meshes, assets) is unchanged. Paths inside the copy are re-pointed (relocate.py).
  python flip_object_ref.py SRC_PKG OUT_PKG EP OBJ [--axis long|mid|short]        (kit or meshclean venv)"""
import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("ep"); ap.add_argument("obj")
ap.add_argument("--axis", default="long", choices=["long", "mid", "short"])
a = ap.parse_args()
src, out = Path(a.src), Path(a.out)
if out.exists():
    raise SystemExit(f"{out} exists")
shutil.copytree(src, out)
if (out / "relocate.py").exists():
    subprocess.run([sys.executable, "-I", str(out / "relocate.py")], check=True, stdout=subprocess.DEVNULL)
m = trimesh.load(out / "assets" / f"{a.obj}.obj", force="mesh", process=False)
pts = trimesh.sample.sample_surface(m, 20000, seed=0)[0]
c = pts.mean(0); _, v = np.linalg.eigh(np.cov((pts - c).T))
F = R.from_rotvec(np.pi * v[:, {"long": 2, "mid": 1, "short": 0}[a.axis]])
p = glob.glob(str(out / f"sim_reference/processed/sequence_id=ep{a.ep}/robot_name=sharpa_wave/*.parquet"))[0]
t = pq.read_table(p); r = t.to_pylist()[0]
i = list(r["object_body_names"]).index(a.obj)
P = np.array(r["object_body_position"], float); Q = np.array(r["object_body_wxyz"], float)
rot = R.from_quat(Q[:, i][:, [1, 2, 3, 0]]); new = rot * F
P[:, i] = P[:, i] + rot.apply(c) - new.apply(c)  # keep the centroid where it was
Q[:, i] = new.as_quat()[:, [3, 0, 1, 2]]


def set_col(t, name, value):
    k = t.column_names.index(name)
    return t.set_column(k, t.schema.field(k), pa.array([value], type=t.schema.field(k).type))


t = set_col(t, "object_body_position", P.tolist()); t = set_col(t, "object_body_wxyz", Q.tolist())
if i == 0:  # the root columns mirror object 0
    t = set_col(t, "object_root_position", P[:, 0].tolist())
    t = set_col(t, "object_root_axis_angle", new.as_rotvec().tolist())
pq.write_table(t, p)
chk = R.from_quat(np.array(pq.read_table(p).to_pylist()[0]["object_body_wxyz"])[:, i][:, [1, 2, 3, 0]])
ang = np.degrees((rot.inv() * chk).magnitude())
cc = np.linalg.norm((rot.apply(c) + np.array(r["object_body_position"])[:, i]) - (chk.apply(c) + P[:, i]), axis=1).max()
print(f"ep{a.ep} {a.obj}: flipped 180 deg about its {a.axis} axis in {len(P)} frames "
      f"(body change {ang.min():.1f}-{ang.max():.1f} deg, centroid moved max {1000 * cc:.3f} mm) -> {out}")
