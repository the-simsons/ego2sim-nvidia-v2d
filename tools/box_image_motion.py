"""Video only: frames + real frame times, an object mask by colour (optional), and the object's image motion relative
to the background. The motion signal is what e2s.mocap.assist.sync correlates with the mocap speed; it uses nothing
but the video. (Ported from the 2026-11-08 AIMS sync_clips.py, video half.)

    python tools/box_image_motion.py --video V.MOV --out CLIP_DIR [--hsv 100,90,70,128,255,255] [--masks DIR]
Writes CLIP_DIR/frames/NNNNNN.jpg (if missing), CLIP_DIR/frame_times.json {"pts": [...]},
CLIP_DIR/box_masks/NNNNNN.png (with --hsv), CLIP_DIR/video_motion.json {"pts", "signal"} (px/s at half resolution)."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def colour_mask(bgr, lo, hi):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    return (lab == k).astype(np.uint8) * 255


def frame_times(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0",
                        str(video)], capture_output=True, text=True, check=True)
    return np.array([float(x.strip().strip(",")) for x in r.stdout.split() if x.strip(", ")])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hsv", help="lo_h,lo_s,lo_v,hi_h,hi_s,hi_v: make masks by colour into OUT/box_masks")
    ap.add_argument("--masks", help="existing mask dir (NNNNNN.png) instead of --hsv")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    if not (out / "frames" / "000000.jpg").exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.video, "-vsync", "0", "-start_number", "0", "-q:v", "2",
                        str(out / "frames" / "%06d.jpg")], check=True)
    pts = frame_times(a.video)
    files = sorted((out / "frames").glob("*.jpg"))
    assert len(files) == len(pts), (len(files), len(pts))
    json.dump({"pts": pts.tolist()}, open(out / "frame_times.json", "w"))
    if a.hsv:
        h = [int(x) for x in a.hsv.split(",")]
        lo, hi = tuple(h[:3]), tuple(h[3:])
        (out / "box_masks").mkdir(exist_ok=True)
    mdir = Path(a.masks) if a.masks else out / "box_masks"

    prev, c_prev, flow, missing = None, None, [np.nan], 0
    for i, f in enumerate(files):
        im = cv2.imread(str(f))
        if a.hsv:
            m = colour_mask(im, lo, hi)
            if m is not None:
                cv2.imwrite(str(mdir / f"{i:06d}.png"), m)
        else:
            m = cv2.imread(str(mdir / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
            m = m if m is not None and m.any() else None
        g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
        if m is None:
            missing += 1
            c = None
        else:
            ys, xs = np.nonzero(m)
            c = np.array([xs.mean(), ys.mean()]) / 2
        if prev is not None and c is not None and c_prev is not None:
            excl = cv2.resize(m, (g.shape[1], g.shape[0]))
            bgm = (cv2.dilate(excl, np.ones((41, 41), np.uint8)) == 0).astype(np.uint8) * 255
            p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 12, mask=bgm)
            bg = np.zeros(2)
            if p0 is not None and len(p0) > 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(21, 21), maxLevel=3)
                ok = st.ravel() == 1
                if ok.sum() > 8:
                    bg = np.median((p1 - p0)[ok].reshape(-1, 2), axis=0)
            flow.append(float(np.linalg.norm(c - c_prev - bg)))
        elif i > 0:
            flow.append(np.nan)
        prev, c_prev = g, c
    v = np.nan_to_num(np.array(flow)) / np.maximum(np.gradient(pts), 1e-3)
    v = np.convolve(v, np.ones(3) / 3, mode="same")
    json.dump({"pts": pts.tolist(), "signal": v.tolist(), "missing_frames": missing}, open(out / "video_motion.json", "w"))
    print(f"{out.name}: {len(files)} frames, {pts[-1]:.2f} s, object missing in {missing} -> {out / 'video_motion.json'}")


if __name__ == "__main__":
    main()
