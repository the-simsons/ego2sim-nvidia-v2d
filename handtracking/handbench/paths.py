"""Filesystem layout. Everything is relative to the handtracking/ root unless overridden by env vars:
HANDBENCH_DATA (default handtracking/data/track3 -> Track 3 public), HANDBENCH_CACHE (cached pipeline outputs)."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("HANDBENCH_DATA", ROOT / "data" / "track3"))
CACHE = Path(os.environ.get("HANDBENCH_CACHE", ROOT / "data" / "cache"))
REFS = Path(os.environ.get("HANDBENCH_REFS", ROOT / "refs"))
RESULTS = Path(os.environ.get("HANDBENCH_RESULTS", ROOT / "results"))
SETS = ROOT / "handbench" / "sets"


def ep_name(ep: int | str) -> str:
    return ep if isinstance(ep, str) and ep.startswith("episode_") else f"episode_{int(ep):06d}"


def ep_index(ep: int | str) -> int:
    return int(str(ep).replace("episode_", ""))


def calib_json() -> Path:
    return DATA / "meta" / "camera_calibration.json"


def rig_json() -> Path:
    return REFS / "calib" / "rig.json"


def video(ep, cam: str) -> Path:
    return DATA / "videos" / "chunk-000" / f"observation.images.{cam}" / f"{ep_name(ep)}.mp4"


def parquet(ep) -> Path:
    return DATA / "data" / "chunk-000" / f"{ep_name(ep)}.parquet"


@lru_cache(None)
def info() -> dict:
    return json.load(open(DATA / "meta" / "info.json"))


def fps() -> float:
    return float(info()["fps"])


@lru_cache(None)
def num_frames(ep) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video(ep, "ego_cam_a")))
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def episodes(set_name: str = "v2d_dev") -> list[str]:
    return [ep_name(l.strip()) for l in open(SETS / f"{set_name}.txt") if l.strip() and not l.startswith("#")]
