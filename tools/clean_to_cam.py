"""Express a cleaned reference object track in the camera, so it can be checked against the SAM2 mask with
fp_track_diag.py: per frame, the body-frame change raw -> cleaned in the sim reference is applied to the raw
FoundationPose camera pose (T_cam_clean = T_cam_fp @ inv(T_sim_raw) @ T_sim_clean).
  python clean_to_cam.py FP_DIR RAW_PKG CLEAN_PKG EP OBJ OUT_DIR"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

fp, raw, clean, ep, obj, out = sys.argv[1:7]


def sim(pkg):
    r = pq.read_table(glob.glob(f"{pkg}/sim_reference/processed/sequence_id=ep{ep}/robot_name=sharpa_wave/*.parquet")[0]).to_pylist()[0]
    i = list(r["object_body_names"]).index(obj)
    P, q = np.array(r["object_body_position"])[:, i], np.array(r["object_body_wxyz"])[:, i]
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3, :3] = R.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    T[:, :3, 3] = P
    return T


A, B = sim(raw), sim(clean)
Path(out).mkdir(parents=True, exist_ok=True)
for t in range(len(A)):
    p = Path(fp) / f"{t:06d}.json"
    if not p.exists():
        continue
    j = json.load(open(p))
    w, x, y, z = j["rotation"]
    C = np.eye(4)
    C[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    C[:3, 3] = j["translation"]
    N = C @ np.linalg.inv(A[t]) @ B[t]
    x, y, z, w = R.from_matrix(N[:3, :3]).as_quat()
    j["rotation"], j["translation"] = [w, x, y, z], N[:3, 3].tolist()
    json.dump(j, open(Path(out) / p.name, "w"))
print(f"-> {out}")
