"""Object-only views for multi-view reconstruction services that assume a static scene (e.g. World Labs Atlas): per
frame, the object is cut out with its SAM2 mask onto a black background and cropped to a square around it, so a handheld
object turning in front of a still camera looks like a camera orbiting a static object. Frames are chosen greedily for
the most different object orientations relative to the camera (FoundationPose track), among frames whose mask is large
and whose track agrees with the mask.
  python object_views.py VIDEO MASK_DIR FP_DIR MESH INTRINSICS OUT_DIR [--n 16] [--size 1024] [--pad 0.15] [--min-iou 0.6]
Writes OUT_DIR/view_XX_fFFFFFF.png and views.json (frame, crop box, rotation angle to the first view)."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from fp_track_diag import load_track, silhouette  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("masks"); ap.add_argument("fp"); ap.add_argument("mesh"); ap.add_argument("intr")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=16); ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--pad", type=float, default=0.15); ap.add_argument("--min-iou", type=float, default=0.6)
ap.add_argument("--min-area", type=float, default=0.5, help="fraction of the median mask area")
ap.add_argument("--border", type=int, default=8, help="skip frames whose mask comes within this many px of the edge")
a = ap.parse_args()
K = json.load(open(a.intr))
files = sorted(Path(a.masks).glob("*.png"))
n = len(files)
T = load_track(a.fp, n)
m = trimesh.load(a.mesh, force="mesh")
if len(m.faces) > 6000:
    m = m.simplify_quadric_decimation(face_count=6000)
s = 0.25
hw = (int(K["height"] * s), int(K["width"] * s))
area, iou = np.zeros(n), np.zeros(n)
for i, f in enumerate(files):
    mk = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 0
    if mk[:a.border].any() or mk[-a.border:].any() or mk[:, :a.border].any() or mk[:, -a.border:].any():
        continue  # cut by the image edge: the straight cut would read as geometry
    area[i] = mk.sum()
    if np.isnan(T[i, 0, 0]) or not mk.any():
        continue
    ms = cv2.resize(mk.astype(np.uint8), hw[::-1], interpolation=cv2.INTER_NEAREST)
    sil = silhouette(m, T[i], K, hw, s)
    iou[i] = (sil & ms).sum() / max((sil | ms).sum(), 1)
ok = np.flatnonzero((iou >= a.min_iou) & (area >= a.min_area * np.median(area[area > 0])))
if len(ok) < a.n:
    raise SystemExit(f"only {len(ok)} usable frames (IoU >= {a.min_iou}); lower --min-iou")
rots = R.from_matrix(T[ok, :3, :3])
pick = [int(np.argmax(area[ok]))]  # start from the largest view, then farthest-point in orientation
dmin = np.degrees((rots * rots[pick[0]].inv()).magnitude())
while len(pick) < a.n:
    k = int(np.argmax(dmin)); pick.append(k)
    dmin = np.minimum(dmin, np.degrees((rots * rots[k].inv()).magnitude()))
pick = sorted(ok[pick])
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(a.video)
want, meta, fi = set(pick), [], 0
while want:
    okf, img = cap.read()
    if not okf:
        break
    if fi in want:
        mk = cv2.imread(str(files[fi]), cv2.IMREAD_GRAYSCALE) > 0
        ys, xs = np.nonzero(mk)
        c, half = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2]), max(np.ptp(xs), np.ptp(ys)) * (0.5 + a.pad)
        x0, y0 = int(c[0] - half), int(c[1] - half)
        side = int(2 * half)
        cut = np.where(mk[..., None], img, 0).astype(np.uint8)
        canvas = np.zeros((side, side, 3), np.uint8)  # crop may run off the image: paste into a black square
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + side, img.shape[1]), min(y0 + side, img.shape[0])
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = cut[sy0:sy1, sx0:sx1]
        k = len(meta)
        cv2.imwrite(str(out / f"view_{k:02d}_f{fi:06d}.png"), cv2.resize(canvas, (a.size, a.size), interpolation=cv2.INTER_AREA))
        meta.append({"view": k, "frame": fi, "box": [x0, y0, side], "iou": round(float(iou[fi]), 3),
                     "deg_from_view0": round(float(np.degrees((R.from_matrix(T[fi, :3, :3]) * R.from_matrix(T[pick[0], :3, :3]).inv()).magnitude())), 1)})
        want.discard(fi)
    fi += 1
json.dump(meta, open(out / "views.json", "w"), indent=1)
print(f"{len(meta)} views from {len(ok)} usable frames -> {out}")
for v in meta:
    print(f"  view {v['view']:2d} frame {v['frame']:4d} IoU {v['iou']:.2f} {v['deg_from_view0']:6.1f} deg from view 0")
