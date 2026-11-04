"""Compose the three demo panels side by side with titles: ego video | 3D reconstruction | robot policy in sim.
All inputs are stepped frame by frame (one sim step per video frame).
  compose_demo.py EGO.mp4 RECON.mp4 SIM.mp4 OUT.mp4 --title "..." [--subtitle "..."] [--height 405]"""
import argparse
import cv2, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("ego"); ap.add_argument("recon"); ap.add_argument("sim"); ap.add_argument("out")
ap.add_argument("--title", default=""); ap.add_argument("--subtitle", default=""); ap.add_argument("--height", type=int, default=405)
a = ap.parse_args()
caps = [cv2.VideoCapture(p) for p in (a.ego, a.recon, a.sim)]
labels = ["1  Egocentric video  (SAM2 objects + hands)", "2  3D reconstruction  (meshes, 6-DoF tracks, MANO hands)",
          "3  Robot policy in simulation  (Sharpa hands, FlashSAC)"]
H = a.height; HEAD, FOOT = 64, 40
frames_last = [None] * 3; out = None; n = 0
while True:
    tiles, alive = [], False
    for k, c in enumerate(caps):
        ok, f = c.read()
        if ok:
            frames_last[k] = f; alive = True
        f = frames_last[k]
        if f is None:
            f = np.zeros((H, H, 3), np.uint8)
        f = cv2.resize(f, (int(round(f.shape[1] * H / f.shape[0])), H), interpolation=cv2.INTER_AREA)
        bar = np.full((34, f.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, labels[k], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, f]))
    if not alive:
        break
    row = np.hstack([np.hstack([t, np.full((t.shape[0], 6, 3), 18, np.uint8)]) for t in tiles])[:, :-6]
    head = np.full((HEAD, row.shape[1], 3), 18, np.uint8)
    cv2.putText(head, a.title, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    foot = np.full((FOOT, row.shape[1], 3), 18, np.uint8)
    cv2.putText(foot, a.subtitle, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(foot, f"frame {n}", (row.shape[1] - 130, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
    img = np.vstack([head, row, foot])
    if img.shape[1] % 2: img = img[:, :-1]
    if img.shape[0] % 2: img = img[:-1]
    if out is None:
        out = cv2.VideoWriter(a.out + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (img.shape[1], img.shape[0]))
    out.write(img); n += 1
out.release(); print(f"{a.out}.tmp.mp4: {n} frames {img.shape[1]}x{img.shape[0]}")