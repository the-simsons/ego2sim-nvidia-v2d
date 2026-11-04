"""Symmetry-axis tilt over time (deg, relative to frame 0) for an axially symmetric object: mocap vs reference parquets.

  python axis_tilt.py EPISODE OBJ SCAN_AXIS REF_PARQUET=MESH_AXIS [REF_PARQUET=MESH_AXIS ...] [--every N]
Axes are x|y|z in each body frame (scan frame for mocap). Also prints the world-frame angle between each track's axis and
mocap's after aligning frame 0 (what axial-symmetry scoring sees).
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

AX = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
DATA = Path.home() / "v2d/v2d_data/track_3/public"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--every")]
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 15
    args = [a for a in args if a != str(every)]
    ep, obj, sax = int(args[0]), args[1], AX[args[2]]
    rows = pq.read_table(DATA / "data/chunk-000" / f"episode_{ep:06d}.parquet",
                         columns=["observation.objects"]).column(0).to_pylist()
    q = np.array([next(o for o in r if o["name"] == obj)["pose"][3:] for r in rows])  # wxyz
    Rm = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    am = Rm.apply(sax)
    series = [("mocap", am)]
    for a in args[3:]:
        p, ax = a.rsplit("=", 1)
        r = pq.read_table(p, columns=["object_body_names", "object_body_wxyz"]).to_pylist()[0]
        Q = np.asarray(r["object_body_wxyz"], float)[:, r["object_body_names"].index(obj)]
        series.append((Path(p).parts[-5], Rotation.from_quat(Q[:, [1, 2, 3, 0]]).apply(AX[ax])))
    T = min(len(s) for _, s in series)

    def ang(u, v):
        return np.degrees(np.arccos(np.clip(np.sum(u * v, -1), -1, 1)))

    # align each track's frame-0 axis to mocap's (rotation taking a0 -> m0), then compare world axes
    out = {}
    for lab, a in series:
        a = a[:T]
        if lab == "mocap":
            out[lab] = a
            continue
        rot, _ = Rotation.align_vectors([am[0]], [a[0]])
        out[lab] = rot.apply(a)
    print("frame " + " ".join(f"{lab + ' tilt':>16s}" for lab in out) + "  " +
          " ".join(f"{lab + ' vs mocap':>18s}" for lab in out if lab != "mocap"))
    for i in range(0, T, every):
        print(f"{i:5d} " + " ".join(f"{ang(out[l][i], out[l][0]):16.1f}" for l in out) + "  " +
              " ".join(f"{ang(out[l][i], out['mocap'][i]):18.1f}" for l in out if l != "mocap"))
    for l in out:
        if l != "mocap":
            e = ang(out[l], out["mocap"])
            print(f"{l}: axis error vs mocap median {np.median(e):.1f} p90 {np.percentile(e, 90):.1f} deg")


if __name__ == "__main__":
    main()
