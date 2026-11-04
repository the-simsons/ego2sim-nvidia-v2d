"""Track 3 cam_a -> pinhole input video for monocular methods (same frames handbench gave HaWoR).

cam_a is undistorted with the refined rig model (handtracking/refs/calib/rig.json) to a pinhole with the factory focal
and the principal point at the image centre, at the true frame rate (paths.fps(), ~20.254; MP4 headers say 20).
Same camera centre and orientation as the handbench benchmark frame.

  ../.venv-bench/bin/python tools/prepare_track3_video.py --episode 41 [--out data/track3_input]
Writes <out>/episode_XXXXXX.mp4 (H.264, CRF 12) and episode_XXXXXX.json (K, size, fps, n_frames, source video).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from handbench import paths  # noqa: E402
from handbench.calib import load_rig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "track3_input"))
    a = ap.parse_args()
    ep = paths.ep_name(a.episode)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A = load_rig(paths.rig_json())["ego_cam_a"]
    W, H = A.size
    f = float(A.K[0, 0])
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    mx, my = cv2.initUndistortRectifyMap(A.K, A.dist, np.eye(3), K, (W, H), cv2.CV_32FC1)
    src = str(paths.video(ep, "ego_cam_a"))
    fps = paths.fps()
    dst = out / f"{ep}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                           "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(src)
    n = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        ff.stdin.write(np.ascontiguousarray(cv2.remap(im, mx, my, cv2.INTER_LINEAR)).tobytes())
        n += 1
    ff.stdin.close()
    ff.wait()
    json.dump({"episode": ep, "K": K.tolist(), "size": [W, H], "fps": fps, "n_frames": n, "source_video": src,
               "rig": str(paths.rig_json())}, open(out / f"{ep}.json", "w"), indent=1)
    print(f"{dst}: {n} frames at {fps:.4f} fps")


if __name__ == "__main__":
    main()
