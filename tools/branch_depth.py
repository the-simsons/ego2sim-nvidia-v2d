"""Which of two look-alike orientations of an object does the measured depth support? For a FoundationPose track, per
frame, the tracked pose (A) and the same pose turned 180 deg about one of the mesh's principal axes through its centroid
(B, a look-alike flip: same silhouette) are ray-cast into the camera at the object's mask pixels; the median absolute
difference to the FoundationStereo depth decides. Then, for each given reference package, report which branch its pose
follows per frame (body-frame change from the raw package: ~0 deg = A, ~180 deg = B).
  python branch_depth.py EP_DIR FP_DIR MESH MASK_DIR OBJ [--axis long|mid|short] [--every 3]
                         [--refs RAW_PKG LABEL=PKG ...] [--ep N]          (meshclean venv: trimesh + embree, cv2)"""
import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from fp_track_diag import load_track  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ep"); ap.add_argument("fp"); ap.add_argument("mesh"); ap.add_argument("masks"); ap.add_argument("obj")
ap.add_argument("--axis", default="long", choices=["long", "mid", "short"]); ap.add_argument("--every", type=int, default=3)
ap.add_argument("--refs", nargs="*", default=[]); ap.add_argument("--epnum", type=int)
a = ap.parse_args()
E = Path(a.ep)
K = json.load(open(E / "stereo/intrinsics_undist.json"))
m = trimesh.load(a.mesh, force="mesh")
c = m.centroid
_, v = np.linalg.eigh(np.cov(trimesh.sample.sample_surface(m, 20000, seed=0)[0].T))
ax = v[:, {"long": 2, "mid": 1, "short": 0}[a.axis]]
Fl = np.eye(4); Fl[:3, :3] = R.from_rotvec(np.pi * ax).as_matrix(); Fl[:3, 3] = c - Fl[:3, :3] @ c  # flip about centroid
rt = m.ray
files = sorted(Path(a.masks).glob("*.png")); n = len(files)
T = load_track(a.fp, n)
D = sorted((E / "stereo_fs/stereo_depth").glob("*.png"))
rng = np.random.default_rng(0)


def render_depth(Tco, uv):
    """Depth (camera z) of the mesh at pose Tco along the rays through pixels uv; nan where missed."""
    d = np.c_[(uv[:, 0] - K["cx"]) / K["fx"], (uv[:, 1] - K["cy"]) / K["fy"], np.ones(len(uv))]
    Ti = np.linalg.inv(Tco)
    o = np.repeat(Ti[:3, 3][None], len(uv), 0); dirs = d @ Ti[:3, :3].T
    loc, idx, _ = rt.intersects_location(o, dirs, multiple_hits=False)
    z = np.full(len(uv), np.nan)
    if len(idx):
        pc = loc @ Tco[:3, :3].T + Tco[:3, 3]
        z[idx] = pc[:, 2]
    return z


rows = []
for f in range(0, n, a.every):
    if np.isnan(T[f, 0, 0]) or f >= len(D):
        continue
    mk = cv2.erode((cv2.imread(str(files[f]), 0) > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
    ys, xs = np.nonzero(mk)
    if len(xs) < 200:
        continue
    k = rng.choice(len(xs), min(800, len(xs)), replace=False)
    uv = np.c_[xs[k], ys[k]].astype(float)
    raw = cv2.imread(str(D[f]), cv2.IMREAD_UNCHANGED).astype(np.float32)[ys[k], xs[k]]
    meas = np.where(raw < 65535, 65535.0 / np.maximum(raw, 1) - 1.0, np.nan)
    za, zb = render_depth(T[f], uv), render_depth(T[f] @ Fl, uv)
    ok = ~np.isnan(meas) & ~np.isnan(za) & ~np.isnan(zb)
    if ok.sum() < 50:
        continue
    ea, eb = np.median(np.abs(za[ok] - meas[ok])) * 1000, np.median(np.abs(zb[ok] - meas[ok])) * 1000
    rows.append((f, ea, eb))
print(f"{a.obj}: depth error (mm, median in mask) tracked pose A vs flipped B ({a.axis} axis)")
for f, ea, eb in rows:
    print(f"  {f:4d}  A {ea:5.1f}  B {eb:5.1f}  -> {'A' if ea < eb else 'B'}{' (clear)' if abs(ea - eb) > 1.5 else ''}")
na = sum(ea < eb for _, ea, eb in rows); clear = [(f, 'A' if ea < eb else 'B') for f, ea, eb in rows if abs(ea - eb) > 1.5]
print(f"frames A better {na}/{len(rows)}; clear decisions: " + " ".join(f"{f}{b}" for f, b in clear))

if a.refs:
    import pyarrow.parquet as pq

    def sim(pkg):
        p = glob.glob(f"{pkg}/sim_reference/processed/sequence_id=ep{a.epnum}/robot_name=sharpa_wave/*.parquet")[0]
        r = pq.read_table(p).to_pylist()[0]; i = list(r["object_body_names"]).index(a.obj)
        return R.from_quat(np.array(r["object_body_wxyz"])[:, i][:, [1, 2, 3, 0]])
    raw_pkg, refs = a.refs[0], a.refs[1:]
    R0 = sim(raw_pkg)
    print("reference branch per frame (angle of body-frame change vs the raw package; ~180 = flipped = B):")
    fr = [f for f, _, _ in rows]
    for spec in refs:
        lab, pkg = spec.split("=", 1)
        Rr = sim(pkg)
        ang = [float(np.degrees((R0[f].inv() * Rr[f]).magnitude())) for f in fr if f < len(Rr)]
        print(f"  {lab:12} " + " ".join(f"{f}:{'B' if x > 120 else 'A' if x < 60 else '?'}" for f, x in zip(fr, ang)))
