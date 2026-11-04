"""Add mid-task reset frames to an episode's S0 extras: every 3rd frame (0 .. T-11) at which some object of the reference
moves within the next 10 frames (> 1 cm or > 5 deg) becomes a pickup reset frame, on top of the existing ones. Used with
the contact-reward curriculum (cfg/curriculum/tracking_96m_pickup_ray5decay_contact.yaml) so the policy also practises the
middle of the task, e.g. ep19 (sweeper + dust pan) in the 4 Nov submission.
  add_reset_frames.py <reference package> <episode> <extras dir in> <extras dir out>
"""
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R


def main(pkg, ep, src, dst):
    ep = int(ep)
    t = pq.read_table(glob.glob(f"{pkg}/sim_reference/processed/sequence_id=ep{ep}/robot_name=sharpa_wave/*.parquet")[0],
                      columns=["object_body_position", "object_body_wxyz"])
    P = np.array(t.column("object_body_position")[0].as_py()); Q = np.array(t.column("object_body_wxyz")[0].as_py()); T = len(P)
    mv = np.zeros(T, bool)
    for b in range(P.shape[1]):
        r = R.from_quat(Q[:, b][:, [1, 2, 3, 0]])
        for f in range(T):
            g = min(f + 10, T - 1)
            mv[f] |= np.linalg.norm(P[f:g + 1, b] - P[f, b], axis=1).max() > 0.01 or np.degrees((r[f:g + 1] * r[f].inv()).magnitude()).max() > 5
    d = dict(np.load(f"{src}/episode_{ep:06d}.npz", allow_pickle=True))
    old = d["pickup_reset_frames"].tolist()
    d["pickup_reset_frames"] = np.asarray(sorted(set(old) | {f for f in range(0, T - 10, 3) if mv[f]}), np.int32)
    os.makedirs(dst, exist_ok=True)
    np.savez(f"{dst}/episode_{ep:06d}.npz", **d)
    print(f"ep{ep}: reset frames {len(old)} -> {len(d['pickup_reset_frames'])} -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
