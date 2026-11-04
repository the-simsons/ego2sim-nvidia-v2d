"""Rewrite the absolute asset / bundle paths inside this package's reference parquets after copying the package.
  python relocate.py            (rewrites to wherever this file now lives)
Needs pyarrow. URDFs reference their meshes relatively, so only the parquets change."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent


def set_col(t, name, value):
    i = t.column_names.index(name)
    return t.set_column(i, t.schema.field(i), pa.array([value] * t.num_rows, type=t.schema.field(i).type))


for p in sorted((ROOT / "sim_reference/processed").glob("sequence_id=*/robot_name=*/*.parquet")):
    t = pq.read_table(p)
    r = t.to_pylist()[0]
    names = list(r["object_body_names"])
    sid = p.parent.parent.name.split("=", 1)[1]
    t = set_col(t, "object_mesh_paths", [str(ROOT / "assets" / f"{o}.obj") for o in names])
    t = set_col(t, "object_urdf_paths", [str(ROOT / "assets" / f"{o}_rigid.urdf") for o in names])
    t = set_col(t, "raw_motion_file", str(ROOT / "raw" / f"{sid}_{names[0]}" / "result.npz"))
    pq.write_table(t, p)
    print(f"{sid}: paths -> {ROOT}")
