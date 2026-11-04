"""HaWoR (single camera) on prepared AssemblyHands / HOT3D clips, with handbench's HaWoR runner unchanged.

Runs in ../methods/.venv-hawor (handbench.runners.hawor_run.worker; cwd = methods/HaWoR is set there).
Input per clip: data/<dataset>_input/<clip>.mp4 + .json (K, size). HaWoR assumes the principal point at the image
centre, so each frame is translated by (W/2 - cx, H/2 - cy) px (exact for a pinhole camera; the prepared clips are
2-4 px off) and HaWoR gets the true focal. Its camera frame then differs from the input camera only by that pixel
shift; tools/hawor_to_groundtruth_hands.py maps its hands back (pixel and depth kept).

  ../methods/.venv-hawor/bin/python tools/run_hawor_on_clips.py --dataset assemblyhands [--gpu 0]
Writes data/hawor/<dataset>/<clip>/{export.npz, input.json, <clip>/extracted_images/} (handbench hawor_run layout);
clips that already have export.npz are skipped.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HB = ROOT.parent
sys.path.insert(0, str(HB))


def prepare_clip(dataset, clip, out: Path):
    """handbench.runners.hawor_run prepare hook: out = data/hawor/<dataset>/<clip>; HaWoR reads frames from
    <out>/<clip>/extracted_images (it derives that folder from the video path <out>/<clip>.mp4)."""
    import cv2
    import numpy as np

    meta = json.loads((ROOT / "data" / f"{dataset}_input" / f"{clip}.json").read_text())
    K = np.array(meta["K"])
    W, H = meta["size"]
    f = float((K[0, 0] + K[1, 1]) / 2)
    shift = np.array([[1, 0, W / 2 - K[0, 2]], [0, 1, H / 2 - K[1, 2]]], np.float64)
    img = out / clip / "extracted_images"
    img.mkdir(parents=True, exist_ok=True)
    src = ROOT / "data" / f"{dataset}_input" / f"{clip}.mp4"
    cap = cv2.VideoCapture(str(src))
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        cv2.imwrite(str(img / f"{i:04d}.jpg"), cv2.warpAffine(im, shift, (W, H), flags=cv2.INTER_LINEAR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        i += 1
    vid = out / f"{clip}.mp4"
    if not vid.exists():
        os.symlink(src, vid)  # HaWoR derives names from the video path; frames already exist
    K_h = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    json.dump({"K": K_h.tolist(), "K_input": K.tolist(), "focal": f, "n_frames": i, "size": [W, H],
               "pixel_shift": [W / 2 - K[0, 2], H / 2 - K[1, 2]]}, open(out / "input.json", "w"), indent=1)
    return vid, f, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["assemblyhands", "hot3d"])
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    from functools import partial

    from handbench.runners.hawor_run import worker

    out = ROOT / "data" / "hawor" / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    clips = sorted(p.stem for p in (ROOT / "data" / f"{a.dataset}_input").glob("*.mp4")
                   if not (out / p.stem / "export.npz").exists())
    ctx = mp.get_context("spawn")
    man = ctx.Manager()
    q, res = man.Queue(), man.list()
    for c in clips:
        q.put(c)
    p = ctx.Process(target=worker, args=(int(a.gpu), q, out, res, partial(prepare_clip, a.dataset)))
    p.start()
    p.join()
    done = [m["episode"] for m in res] if res and "episode" in res[0] else len(res)
    print(f"done {len(res)}/{len(clips)}: {done}")
    sys.exit(0 if len(res) == len(clips) else 1)


if __name__ == "__main__":
    main()
