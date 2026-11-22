"""Where a trajectory came from. Every reconstruction output (bundle manifest, reference package, policy checkpoint
metadata) carries

    {"trajectory_source": "video" | "mocap" | "mixed", "mocap_used_for": [...], "depth_source": "..."}

`mocap_used_for` lists what motion capture supplied, from: "object_trajectory", "camera_path", "table_height",
"sync", "mesh_placement". It is empty for a video-only result. Policy training reads this and refuses anything but
"video" when it is asked for a final result (plans/RECONSTRUCTION_PLAN.md, section 8).
"""
import json
from pathlib import Path

SOURCES = ("video", "mocap", "mixed")
USES = ("object_trajectory", "camera_path", "table_height", "sync", "mesh_placement")
FILENAME = "provenance.json"


def make(trajectory_source, mocap_used_for=(), depth_source="unknown", **extra):
    if trajectory_source not in SOURCES:
        raise ValueError(f"trajectory_source must be one of {SOURCES}, got {trajectory_source!r}")
    bad = [u for u in mocap_used_for if u not in USES]
    if bad:
        raise ValueError(f"unknown mocap uses {bad}; known: {USES}")
    if trajectory_source == "video" and mocap_used_for:
        raise ValueError(f"a video trajectory cannot list mocap uses {list(mocap_used_for)}")
    if trajectory_source == "mocap" and not mocap_used_for:
        raise ValueError("a mocap trajectory must say what mocap supplied (mocap_used_for)")
    return {"trajectory_source": trajectory_source, "mocap_used_for": sorted(set(mocap_used_for)),
            "depth_source": depth_source, **extra}


def write(directory, record):
    p = Path(directory) / FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(record, f, indent=1)
    return p


def read(directory_or_file):
    """The provenance record next to an output, searching the directory and up to 4 parents. None if absent."""
    p = Path(directory_or_file)
    for d in [p if p.is_dir() else p.parent] + list((p if p.is_dir() else p.parent).parents)[:4]:
        f = d / FILENAME
        if f.is_file():
            with open(f) as fh:
                return json.load(fh)
    return None


def require_video(record, final, what="this reference"):
    """Raise unless the record says video-only. With final=False a mocap or mixed source is allowed but reported."""
    src = (record or {}).get("trajectory_source")
    if src == "video":
        return "video"
    if final:
        raise RuntimeError(f"{what} has trajectory_source={src!r} (mocap_used_for={record and record.get('mocap_used_for')}); "
                           "a final result must come from video only (plans/RECONSTRUCTION_PLAN.md section 8)")
    return src or "unlabelled"
