"""Small I/O helpers shared by reference builders and metrics: frame reading, hand-track ids of the cached run."""
from __future__ import annotations

import json

import cv2
import numpy as np

from handbench import paths


def read_frames(ep, i: int, cams=("ego_cam_a", "ego_cam_b", "ego_cam_c")) -> dict | None:
    """Random access to frame i of several camera videos (published, distorted images, BGR)."""
    out = {}
    for c in cams:
        cap = cv2.VideoCapture(str(paths.video(ep, c)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, im = cap.read()
        if not ok:
            return None
        out[c] = im
    return out


class VideoReader:
    """Sequential reader over several synchronised camera videos."""

    def __init__(self, ep, cams=("ego_cam_a", "ego_cam_b", "ego_cam_c")):
        self.caps = {c: cv2.VideoCapture(str(paths.video(ep, c))) for c in cams}

    def __iter__(self):
        i = 0
        while True:
            fr = {}
            for c, cap in self.caps.items():
                ok, im = cap.read()
                if not ok:
                    return
                fr[c] = im
            yield i, fr
            i += 1


def hand_track_ids(ep) -> dict:
    """{sam2 track id: 'L'|'R'} from the cached run's hand_tracks.json."""
    d = json.load(open(paths.CACHE / paths.ep_name(ep) / "hand_tracks.json"))
    return {int(t["object_id"]): ("R" if t["is_right"] else "L") for t in d["tracks"] if t.get("role") == "hand"}


def episode_times(ep) -> np.ndarray:
    """Per-frame time in seconds from the parquet timestamps (falls back to frame / info fps)."""
    import pyarrow.parquet as pq

    n = paths.num_frames(ep)
    try:
        t = pq.read_table(paths.parquet(ep), columns=["timestamp"]).column("timestamp").to_numpy().astype(float)
        if len(t) == n:
            return t - t[0]
    except Exception:
        pass
    return np.arange(n) / paths.fps()
