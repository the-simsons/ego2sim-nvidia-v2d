"""Clip manifests: bench/clips/<clip>.json.

  inputs        what reconstruction may use: video, frames, frame times, intrinsics, depth source, hand side
  ground_truth  scoring only: mocap file, rigid bodies and where each mesh sits in them, sync offset, table height
  outputs_dir   where the pipeline stages for this clip write (relative to the data root)
  split         train | val | test, assigned by object and recording session, never by frame

Paths are relative to the data root (--root, or $E2S_DATA)."""
import json
import os
from pathlib import Path

CLIPS = Path(__file__).parent / "clips"
REQUIRED = {"clip", "dataset", "split", "session", "inputs", "ground_truth", "outputs_dir"}


def load(name_or_path):
    p = Path(name_or_path)
    if not p.suffix:
        p = CLIPS / f"{name_or_path}.json"
    m = json.load(open(p))
    missing = REQUIRED - set(m)
    if missing:
        raise ValueError(f"{p}: missing keys {sorted(missing)}")
    if m["split"] not in ("train", "val", "test"):
        raise ValueError(f"{p}: split must be train/val/test, got {m['split']!r}")
    return m


def all_clips(split=None, dataset=None):
    out = [load(p) for p in sorted(CLIPS.glob("*.json"))]
    return [m for m in out if (split in (None, "all") or m["split"] == split) and (dataset is None or m["dataset"] == dataset)]


def inputs_only(m):
    """What a reconstruction step may see: the manifest without ground truth."""
    return {k: v for k, v in m.items() if k != "ground_truth"}


def data_root(arg=None):
    root = arg or os.environ.get("E2S_DATA")
    if not root:
        raise SystemExit("set --root or $E2S_DATA to the data root (on riddle: /usr1/e2s/rborah_v2d/aims)")
    return Path(root)
