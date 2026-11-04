"""'Our tracks + perfect registration': map a rollout built from an OUR-mesh reference into the scan body frame, using the
constant ours<-scan transform estimated from a second reference tracked with the scan (median over frames). Score the
result with the SCAN meshes: the gap to the scan-tracked score is then tracking only, the rest of the our-mesh gap is the
scorer's mesh->scan registration.

  python oracle_reg_rollout.py ROLLOUT_OURS REF_SCAN REF_OURS OUT
ROLLOUT_OURS: ref_parquet_as_rollout.py output of REF_OURS. All bodies are mapped.
"""
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def ref(p):
    r = pq.read_table(p, columns=["object_body_names", "object_body_position", "object_body_wxyz"]).to_pylist()[0]
    return r["object_body_names"], np.asarray(r["object_body_position"], float), np.asarray(r["object_body_wxyz"], float)


def main():
    roll, ref_s, ref_o, out = sys.argv[1:5]
    names, Ps, Qs = ref(ref_s)
    names_o, Po, Qo = ref(ref_o)
    t = pq.read_table(roll)
    df = t.to_pandas()
    for c in df.columns:
        if "position_" in c or "quaternion_" in c:
            df[c] = df[c].astype(np.float64)
    for b, n in enumerate(names_o):
        s = names.index(n)
        T = min(len(Ps), len(Po))
        Rs = Rotation.from_quat(Qs[:T, s][:, [1, 2, 3, 0]])
        Ro = Rotation.from_quat(Qo[:T, b][:, [1, 2, 3, 0]])
        Xr = (Ro.inv() * Rs).mean()
        Xt = np.median(Ro.inv().apply(Ps[:T, s] - Po[:T, b]), axis=0)
        sel = df["object_body_name"].astype(str) == n
        for pre in ("achieved_", "reference_"):
            p = df.loc[sel, [f"{pre}position_{c}" for c in "xyz"]].to_numpy(float)
            q = df.loc[sel, [f"{pre}quaternion_{c}" for c in "xyzw"]].to_numpy(float)
            R = Rotation.from_quat(q)
            p2 = p + R.apply(Xt)
            q2 = (R * Xr).as_quat()
            for k, c in enumerate("xyz"):
                df.loc[sel, f"{pre}position_{c}"] = p2[:, k]
            for k, c in enumerate("xyzw"):
                df.loc[sel, f"{pre}quaternion_{c}"] = q2[:, k]
        print(f"{n}: X rot {np.degrees(Xr.magnitude()):.1f} deg, trans {np.linalg.norm(Xt) * 100:.1f} cm")
    out_t = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(pa.table({f.name: out_t[f.name].cast(f.type) for f in t.schema}, schema=t.schema), out)


if __name__ == "__main__":
    main()
