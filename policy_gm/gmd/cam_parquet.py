"""Camera position and target for a reference parquet (same framing rule as sub/cam_for_pkg.py), plus its frame count.
  cam_parquet.py <parquet dir or file> <back_d> <up>   ->   "cx,cy,cz tx,ty,tz frames"
"""
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

p, back_d, up = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
f = p if os.path.isfile(p) else sorted(glob.glob(os.path.join(p, "*.parquet")))[0]
t = pq.read_table(f, columns=["object_body_position", "robot_left_wrist_position", "robot_right_wrist_position"]).to_pylist()[0]
op = np.asarray(t["object_body_position"], np.float64)                     # (T, B, 3)
obj = op.reshape(-1, 3).mean(0)
hands = np.concatenate([np.asarray(t["robot_left_wrist_position"]), np.asarray(t["robot_right_wrist_position"])]).mean(0)
target = 0.5 * (obj + hands); target[2] = max(target[2], obj[2]) + 0.03
back = hands[:2] - obj[:2]; back /= max(np.linalg.norm(back), 1e-6)
if op.shape[1] >= 2:
    line = op[0, 1, :2] - op[0, 0, :2]
    if np.linalg.norm(line) > 0.03:
        perp = np.array([-line[1], line[0]]) / np.linalg.norm(line); back = perp if perp @ back >= 0 else -perp
cam = np.r_[target[:2] + back_d * back, target[2] + up]
print(",".join(f"{v:.3f}" for v in cam), ",".join(f"{v:.3f}" for v in target), op.shape[0])
