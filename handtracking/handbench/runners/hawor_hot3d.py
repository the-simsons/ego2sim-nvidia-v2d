"""HaWoR on the HOT3D Aria clips (ground-truth check of the E1best methodology). Runs in methods/.venv-hawor.

Input per clip: the RGB stream rendered as an UPRIGHT pinhole view (handbench.datasets.hot3d: focal F_PIN, principal
point at the image centre, rotated -90 deg about the optical axis), all 150 frames at the native 30 fps.
Output: data/hot3d_hawor/<clip>/export.npz (same layout as hawor_run.py).
Usage: python -m handbench.runners.hawor_hot3d [--gpus 0,1]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HB = Path(__file__).resolve().parents[2]
OUT = HB / "data" / "hot3d_hawor"


def prepare_hot3d(ep, out: Path):
    import cv2

    sys.path.insert(0, str(HB))
    from handbench.datasets import hot3d as H

    clip = H.Clip(H.ROOT / "train_aria" / f"{ep}.tar")
    K = H.pinhole_K()
    img_dir = out / ep / "extracted_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    maps = None
    for i in range(clip.n):
        c = clip.cameras(i)[H.RGB]
        if maps is None:  # the RGB calibration is constant within a clip
            maps = H.pinhole_from_fisheye_maps(c["params"], K, (H.SIZE_PIN, H.SIZE_PIN), H.R_CAM_FROM_PIN)
        cv2.imwrite(str(img_dir / f"{i:04d}.jpg"), cv2.remap(clip.image(i, H.RGB), *maps, cv2.INTER_LINEAR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    vid = out / f"{ep}.mp4"
    if not vid.exists():
        os.symlink(clip.path, vid)  # HaWoR only derives names from the video path once frames exist
    json.dump({"K": K.tolist(), "focal": H.F_PIN, "n_frames": clip.n}, open(out / "input.json", "w"))
    return vid, H.F_PIN, clip.n


def main():
    from handbench.runners.hawor_run import worker

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()
    sys.path.insert(0, str(HB))
    from handbench.datasets import hot3d as H

    eps = [p.stem for p in H.clips() if not (OUT / p.stem / "export.npz").exists()]
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    man = ctx.Manager()
    q = man.Queue()
    for e in eps:
        q.put(e)
    res = man.list()
    t0 = time.time()
    ps = [ctx.Process(target=worker, args=(int(g), q, OUT, res, prepare_hot3d)) for g in a.gpus.split(",")]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    json.dump({"clips": list(res), "wall_time_s": time.time() - t0}, open(OUT / "run.json", "w"), indent=1)
    print(f"done {len(res)}/{len(eps)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
