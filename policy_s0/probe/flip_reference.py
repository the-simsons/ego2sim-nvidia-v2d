"""Copy a reference package and turn one object over in one episode: its body orientation is rotated 180 deg about a body
axis through the mesh's bounding-box centre, in every frame, so the object occupies the same place and follows the same path
but with the opposite face up (fixes a tracker that started the object face-down, e.g. ep36's spoon).
  flip_reference.py <src package> <dst package> <episode> <object> <axis x|y|z>
Copies assets/, meshes/ and sim_reference/reconstructed_stage/ (the container mounts only the package), writes the flipped
parquet for that episode, rewrites root and body poses consistently."""
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R


def main(src, dst, ep, obj, axis):
    src, dst = Path(src), Path(dst)
    for sub in ("assets", "meshes", "sim_reference/reconstructed_stage"):
        if not (dst / sub).exists():
            shutil.copytree(src / sub, dst / sub)
    rel = f"sim_reference/processed/sequence_id=ep{ep}/robot_name=sharpa_wave"
    f = Path(glob.glob(str(src / rel / "*.parquet"))[0]); (dst / rel).mkdir(parents=True, exist_ok=True)
    t = pq.read_table(f)
    names = [str(n) for n in t.column("object_body_names")[0].as_py()]
    b = names.index(obj)
    mesh = trimesh.load(src / "assets" / f"{obj}.obj", force="mesh", process=False)
    c = (mesh.bounds[0] + mesh.bounds[1]) / 2
    F = R.from_rotvec(np.pi * np.eye(3)["xyz".index(axis)])
    pos = np.array(t.column("object_body_position")[0].as_py()); q = np.array(t.column("object_body_wxyz")[0].as_py())
    rb = R.from_quat(q[:, b][:, [1, 2, 3, 0]])
    pos[:, b] = pos[:, b] + rb.apply(c - F.apply(c))
    q[:, b] = (rb * F).as_quat()[:, [3, 0, 1, 2]]
    cols = {"object_body_position": pos.tolist(), "object_body_wxyz": q.tolist()}
    if b == 0:                                    # the root pose columns describe the first body
        cols["object_root_position"] = pos[:, 0].tolist()
        cols["object_root_axis_angle"] = R.from_quat(q[:, 0][:, [1, 2, 3, 0]]).as_rotvec().tolist()
    for k, v in cols.items():
        i = t.schema.get_field_index(k)
        t = t.set_column(i, t.schema.field(i), pa.array([v], type=t.schema.field(i).type))
    pq.write_table(t, dst / rel / f.name)
    print(f"ep{ep} {obj}: flipped 180 deg about body {axis} through {np.round(c, 4).tolist()} in {len(pos)} frames -> {dst / rel / f.name}")


if __name__ == "__main__":
    main(*sys.argv[1:4], sys.argv[4], sys.argv[5])
