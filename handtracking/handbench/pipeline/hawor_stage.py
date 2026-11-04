"""Pipeline stage 1 (GPU, runs in methods/.venv-hawor): HaWoR on one raw cam_a video -> OUT/<name>/export.npz.

The video is undistorted with the rig's cam_a model to a pinhole with the factory focal and a centred principal point
(HaWoR assumes cx = W/2, cy = H/2), at its native frame rate. Same orientation as the undistorted cam_a of the
pipeline (tools/ego_stereo_depth.py cam_a_undist.mp4), so HaWoR's camera frame is the pipeline's camera frame.

Usage (called by handbench.pipeline.multiview_hands):
  python -m handbench.pipeline.hawor_stage --video RAW_CAM_A.mp4 --out DIR [--name NAME] [--rig rig.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

HB = Path(__file__).resolve().parents[2]  # handtracking/


def prepare_from_video(ep, out: Path, video: str, rig: str):
    import cv2

    sys.path.insert(0, str(HB))
    from handbench.calib import load_rig

    A = load_rig(rig)["ego_cam_a"]
    W, H = A.size
    f = float(A.K[0, 0])
    K_h = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    mx, my = cv2.initUndistortRectifyMap(A.K, A.dist, np.eye(3), K_h, (W, H), cv2.CV_32FC1)
    img = out / ep / "extracted_images"
    img.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        cv2.imwrite(str(img / f"{i:04d}.jpg"), cv2.remap(im, mx, my, cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        i += 1
    vid = out / f"{ep}.mp4"
    if not vid.exists():
        os.symlink(os.path.abspath(video), vid)  # HaWoR only derives names from the path once frames exist
    json.dump({"K": K_h.tolist(), "focal": f, "n_frames": i, "size": [W, H], "video": video},
              open(out / "input.json", "w"), indent=1)
    return vid, f, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="hawor")
    ap.add_argument("--rig", default=str(HB / "refs" / "calib" / "rig.json"))
    a = ap.parse_args()
    out = Path(a.out).resolve()  # HaWoR's runner chdirs into methods/HaWoR: every path must be absolute
    out.mkdir(parents=True, exist_ok=True)
    a.video, a.rig = str(Path(a.video).resolve()), str(Path(a.rig).resolve())
    if (out / a.name / "export.npz").exists():
        print(f"{out / a.name / 'export.npz'} exists, skipped")
        return
    import multiprocessing
    import torch.multiprocessing as tmp

    multiprocessing.set_start_method = tmp.set_start_method = lambda *x, **k: None  # HaWoR/DROID call it again
    from handbench.runners.hawor_run import run_episode

    m = run_episode(a.name, out, partial(prepare_from_video, video=a.video, rig=a.rig))
    print(json.dumps({k: m[k] for k in ("wall_time_s", "fps", "gpu")}))


if __name__ == "__main__":
    main()
