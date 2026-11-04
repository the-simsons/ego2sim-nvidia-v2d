"""Contact sheet of SAM2 hand masks + WiLoR boxes for one episode (debugging side assignment / matching).
  python hand_sheet.py EP_DIR OUT.jpg [--frames 0,60,120,180] [--masks hand_masks]"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

COL = [(255, 64, 64), (64, 255, 64), (64, 64, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
ap = argparse.ArgumentParser()
ap.add_argument("ep")
ap.add_argument("out")
ap.add_argument("--frames", default="0,60,120,180")
ap.add_argument("--masks", default="hand_masks")
a = ap.parse_args()
E = Path(a.ep)
tiles = []
for f in map(int, a.frames.split(",")):
    im = cv2.imread(str(E / f"hands/frames/{f:06d}.jpg"))
    ov = im.copy()
    for t in sorted(p.name for p in (E / a.masks).iterdir()):
        m = cv2.imread(str(E / a.masks / t / f"{f:06d}.png"), 0)
        if m is None or not (m > 0).any():
            continue
        ov[m > 0] = COL[int(t) % 6]
        ys, xs = np.nonzero(m)
        cv2.putText(im, f"m{t}", (int(xs.mean()), int(ys.mean())), 0, 2.5, COL[int(t) % 6], 6)
    im = cv2.addWeighted(im, 0.6, ov, 0.4, 0)
    for d in json.load(open(E / f"hands/wilor_raw/{f:06d}.json")):
        b = d["bbox"]
        c = (0, 0, 255) if d["is_right"] else (255, 0, 0)
        cv2.rectangle(im, (int(b["x0"]), int(b["y0"])), (int(b["x1"]), int(b["y1"])), c, 8)
        cv2.putText(im, "R" if d["is_right"] else "L", (int(b["x0"]), int(b["y0"]) - 10), 0, 3, c, 8)
    cv2.putText(im, f"frame {f}", (30, 100), 0, 3, (255, 255, 255), 8)
    tiles.append(cv2.resize(im, (im.shape[1] // 3, im.shape[0] // 3)))
row = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
w = max(r.shape[1] for r in row)
cv2.imwrite(a.out, np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in row]))
print(a.out)
