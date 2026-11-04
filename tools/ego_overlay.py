"""Ego video panel: undistorted cam_a frames with SAM2 object masks (colour per object, same order/colours as the 3D
render) and hand masks (white) blended in.   ego_overlay.py EPDIR OUT.mp4 [--width 960]"""
import argparse
from pathlib import Path
import cv2, numpy as np
OBJ_BGR = [(77, 84, 237), (102, 191, 77), (230, 128, 77), (64, 191, 242)]  # = render_recon_gl OBJ_COLORS in BGR
ap = argparse.ArgumentParser(); ap.add_argument("epdir"); ap.add_argument("out"); ap.add_argument("--width", type=int, default=960)
a = ap.parse_args(); E = Path(a.epdir)
cap = cv2.VideoCapture(str(E / "stereo/cam_a_undist.mp4")); fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
objs = sorted((E / "masks").glob("*"), key=lambda p: int(p.name)); hands = sorted((E / "hand_masks").glob("*"))
w = None; i = 0
while True:
    ok, im = cap.read()
    if not ok:
        break
    im = im.astype(np.float32)
    hm = np.zeros(im.shape[:2], bool)
    for d in hands:
        m = cv2.imread(str(d / f"{i:06d}.png"), 0)
        if m is not None:
            hm |= m > 0
    im[hm] = 0.6 * im[hm] + 0.4 * 255
    for k, d in enumerate(objs):
        m = cv2.imread(str(d / f"{i:06d}.png"), 0)
        if m is None:
            continue
        m = m > 0
        im[m] = 0.45 * im[m] + 0.55 * np.array(OBJ_BGR[k % 4], np.float32)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, OBJ_BGR[k % 4], 4)
    im = im.clip(0, 255).astype(np.uint8)
    h = int(round(im.shape[0] * a.width / im.shape[1])); im = cv2.resize(im, (a.width, h), interpolation=cv2.INTER_AREA)
    if w is None:
        w = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (a.width, h))
    w.write(im); i += 1
w.release(); print(f"{a.out}: {i} frames")