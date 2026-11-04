"""Frame-0 sensitivity of the Track 3 pose metrics (the scorer aligns the whole scene on frame 0).

  python frame0_variant.py ROLLOUT OUT --from OTHER_ROLLOUT     # replace step-0 poses with OTHER's step-0 poses
  python frame0_variant.py ROLLOUT OUT --static K               # step-0 pose := robust mean of steps 0..K-1 per body
Both achieved_ and reference_ columns are changed.
"""
import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

POS = [f"position_{c}" for c in "xyz"]
QUAT = [f"quaternion_{c}" for c in "xyzw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout")
    ap.add_argument("out")
    ap.add_argument("--from", dest="other")
    ap.add_argument("--static", type=int)
    a = ap.parse_args()
    t = pq.read_table(a.rollout)
    df = t.to_pandas()
    for c in df.columns:
        if "position_" in c or "quaternion_" in c:
            df[c] = df[c].astype(np.float64)
    other = pq.read_table(a.other).to_pandas() if a.other else None
    for n in df.object_body_name.astype(str).unique():
        sel = df.object_body_name.astype(str) == n
        b = df[sel].sort_values("step")
        i0 = b.index[0]
        for pre in ("achieved_", "reference_"):
            if other is not None:
                o = other[(other.object_body_name.astype(str) == n)].sort_values("step").iloc[0]
                p, q = o[[pre + c for c in POS]].to_numpy(float), o[[pre + c for c in QUAT]].to_numpy(float)
            else:
                k = b.iloc[:a.static]
                p = np.median(k[[pre + c for c in POS]].to_numpy(float), axis=0)
                q = Rotation.from_quat(k[[pre + c for c in QUAT]].to_numpy(float)).mean().as_quat()
            df.loc[i0, [pre + c for c in POS]] = p
            df.loc[i0, [pre + c for c in QUAT]] = q
    out_t = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(pa.table({f.name: out_t[f.name].cast(f.type) for f in t.schema}, schema=t.schema), a.out)


if __name__ == "__main__":
    main()
