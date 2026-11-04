"""HOT3D Aria clips -> pinhole input videos for the monocular methods (the same view handbench gave HaWoR).

The RGB stream (FISHEYE624) is re-projected with handbench.datasets.hot3d to the upright pinhole benchmark camera
(focal F_PIN, principal point at the image centre, SIZE_PIN square), all 150 frames at the native 30 fps. Clips: the 7
scored in the earlier HOT3D check (handtracking/results/hot3d_check.json: per_clip_mpjpe), 1,433 hand-frames.
HOT3D Aria is in EgoFound3R's training data (and these clips are from HOT3D's train split): seen domain.

  ../.venv-bench/bin/python tools/prepare_hot3d_video.py [--out data/hot3d_input]
Writes <out>/<clip>.mp4 (H.264, CRF 12) and <clip>.json (K, size, fps, n_frames, source tar).
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
from handbench.datasets import hot3d as H  # noqa: E402

FPS = 30.0


def scored_clips() -> list[str]:
    return list(json.loads((paths.ROOT / "results" / "hot3d_check.json").read_text())["per_clip_mpjpe"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "hot3d_input"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    K = H.pinhole_K()
    S = H.SIZE_PIN
    for name in scored_clips():
        clip = H.Clip(H.ROOT / "train_aria" / f"{name}.tar")
        dst = out / f"{name}.mp4"
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{S}x{S}", "-r", f"{FPS}", "-i", "-", "-c:v", "libx264", "-crf", "12",
                               "-preset", "medium", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
        maps = None
        for i in range(clip.n):
            if maps is None:  # the RGB calibration is constant within a clip
                c = clip.cameras(i)[H.RGB]
                maps = H.pinhole_from_fisheye_maps(c["params"], K, (S, S), H.R_CAM_FROM_PIN)
            ff.stdin.write(np.ascontiguousarray(cv2.remap(clip.image(i, H.RGB), *maps, cv2.INTER_LINEAR)).tobytes())
        ff.stdin.close()
        ff.wait()
        json.dump({"clip": name, "K": K.tolist(), "size": [S, S], "fps": FPS, "n_frames": clip.n,
                   "source": str(clip.path), "R_cam_from_pin": np.asarray(H.R_CAM_FROM_PIN).tolist()},
                  open(out / f"{name}.json", "w"), indent=1)
        print(f"{dst}: {clip.n} frames")


if __name__ == "__main__":
    main()
