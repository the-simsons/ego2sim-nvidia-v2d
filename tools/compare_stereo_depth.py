"""Compare two ego stereo depth folders (ego_stereo_depth.py outputs) inside an object's masks.
  compare_stereo_depth.py DEPTH_A DEPTH_B MASK_DIR OUT_PNG [--frames N] [--show FRAME] [--video MP4]
Per frame: valid fraction of the mask in each, median |A-B| where both are valid, and depth spread (p90-p10) on
the mask in each. Writes a side-by-side figure for one frame (frame | A | B), coloured over a fixed near..far range."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def load(d, i):
    p = Path(d) / f"{i:06d}.png"
    if not p.exists():
        return None
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
    z = 65535.0 / np.maximum(raw, 1) - 1.0
    z[raw >= 65535] = 0
    return z


def colour(z, lo, hi):
    t = np.clip((z - lo) / (hi - lo), 0, 1)
    c = cv2.applyColorMap((255 * (1 - t)).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[z <= 0] = 235
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("masks"); ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    rows = []
    for i in range(args.frames):
        za, zb = load(args.a, i), load(args.b, i)
        mp = Path(args.masks) / f"{i:06d}.png"
        if za is None or zb is None or not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, za.shape[::-1], interpolation=cv2.INTER_NEAREST) > 127
        m = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0  # keep away from the silhouette edge
        if m.sum() < 500:
            continue
        va, vb = m & (za > 0), m & (zb > 0)
        both = va & vb
        spread = lambda z, v: float(np.subtract(*np.percentile(z[v], [90, 10]))) * 100 if v.sum() > 50 else np.nan
        rows.append((i, va.sum() / m.sum(), vb.sum() / m.sum(),
                     float(np.median(np.abs(za[both] - zb[both]))) * 100 if both.sum() > 50 else np.nan,
                     float(np.median(zb[both] - za[both])) * 100 if both.sum() > 50 else np.nan,
                     spread(za, va), spread(zb, vb)))
    r = np.array(rows)
    print("frames", len(r))
    print(f"valid fraction on mask: A {np.nanmean(r[:, 1]):.1%}  B {np.nanmean(r[:, 2]):.1%}")
    print(f"median |A-B| on mask: {np.nanmedian(r[:, 3]):.2f} cm (B-A median {np.nanmedian(r[:, 4]):+.2f} cm)")
    print(f"depth spread p90-p10 on mask: A {np.nanmedian(r[:, 5]):.1f} cm  B {np.nanmedian(r[:, 6]):.1f} cm")
    s = args.show
    za, zb = load(args.a, s), load(args.b, s)
    m = cv2.imread(str(Path(args.masks) / f"{s:06d}.png"), cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, za.shape[::-1], interpolation=cv2.INTER_NEAREST) > 127
    near = np.percentile(np.concatenate([za[m & (za > 0)], zb[m & (zb > 0)]]), [2, 98])
    lo, hi = max(near[0] - 0.3, 0.1), near[1] + 0.6
    panels = [colour(za, lo, hi), colour(zb, lo, hi)]
    if args.video:
        cap = cv2.VideoCapture(args.video); cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        ok, f = cap.read()
        if ok:
            panels.insert(0, cv2.resize(f, za.shape[::-1]))
    fig = np.concatenate([cv2.resize(p, (676, 507)) for p in panels], axis=1)
    cv2.imwrite(args.out, fig)
    print("->", args.out)


if __name__ == "__main__":
    main()
