"""Pipeline stages the benchmark knows how to read. A scored pipeline is one camera stage plus one object stage:

    world <- mesh  =  (world <- camera, from the camera stage)  @  (camera <- mesh, from the object stage)

so a single stage can be swapped and re-scored while the other is read from its cached outputs. Each stage declares
what motion capture it used; the pipeline's provenance is the union (e2s.provenance).

To add a stage: write its outputs under the clip's outputs_dir and add an entry here."""
from pathlib import Path

import numpy as np

from e2s import provenance
from e2s.geometry import load_pose_dir

# camera stages: an npz with camera_to_world (N,4,4); world = whatever the stage uses (camera 0, or mocap world)
CAMERA = {
    "video_vo": {"file": "camera_vo_video.npz", "mocap_used_for": [], "about": "RGB-D odometry on the background (tools/rgbd_vo.py)"},
    "video_pg": {"file": "camera_vo_pg.npz", "mocap_used_for": [],
                 "about": "rgbd_vo refined by tools/camera_pose_graph.py (table planes + resting object)"},
    "video_pi3": {"file": "camera_vo_pi3.npz", "mocap_used_for": [],
                  "about": "Pi3X video odometry, 16-frame chunks, overlap 6, MoGe scale (tools/pi3_camera.py)"},
    "video_pi3c24": {"file": "camera_vo_pi3c24.npz", "mocap_used_for": [],
                     "about": "Pi3X video odometry, 24-frame chunks, overlap 8, MoGe scale"},
    "video_pi3_joint": {"file": "camera_vo_pi3_joint.npz", "mocap_used_for": [],
                        "about": "Pi3X camera refined jointly with the object tracker: resting-object constraints + per-10-frame scale "
                                 "(tools/camera_pose_graph.py --no-table --scale-every 10)"},
    "mocap": {"file": "camera_vo.npz", "mocap_used_for": ["camera_path", "sync", "mesh_placement"],
              "about": "camera from the mocap object and FoundationPose (e2s/mocap/assist)"},
}
# object stages: per-frame pose JSONs, camera <- mesh
OBJECT = {
    "fp_box": {"dir": "fp_box", "mocap_used_for": [], "about": "FoundationPose on MoGe depth, video only"},
    "fp_sym": {"dir": "fp_sym", "mocap_used_for": [],
               "about": "fp_box after tools/sym_track.py (look-alike twins chosen over the clip, frame 0 from the still start)"},
    "fp_rep": {"dir": "fp_rep", "mocap_used_for": [],
               "about": "fp_box with jump failures repaired (tools/fp_repair.py, on the Pi3X joint camera)"},
    "fp_mocap": {"dir": "fp_mocap", "mocap_used_for": ["object_trajectory", "camera_path", "sync", "mesh_placement"],
                 "about": "the mocap trajectory expressed in the camera (mocap-assisted reference)"},
}
PRESETS = {
    "video_baseline": ("video_vo", "fp_box"),
    "video_sym": ("video_vo", "fp_sym"),
    "video_pg": ("video_pg", "fp_box"),
    "video_pi3": ("video_pi3", "fp_box"),
    "video_pi3c24": ("video_pi3c24", "fp_box"),
    "video_pi3_joint": ("video_pi3_joint", "fp_box"),
    "video_pi3_joint_rep": ("video_pi3_joint", "fp_rep"),
    "mocap_assisted": ("mocap", "fp_mocap"),
    "diag_fp_mocap_camera": ("mocap", "fp_box"),
}


def parse(spec):
    """'video_baseline' or 'camera=video_vo,object=fp_box' -> (camera, object)."""
    if spec in PRESETS:
        return PRESETS[spec]
    kv = dict(x.split("=", 1) for x in spec.split(","))
    cam, obj = kv.get("camera"), kv.get("object")
    if cam not in CAMERA or obj not in OBJECT:
        raise SystemExit(f"unknown stages in {spec!r}; camera: {list(CAMERA)}, object: {list(OBJECT)}, presets: {list(PRESETS)}")
    return cam, obj


def load(manifest, root, cam, obj, n_frames):
    """{"objects": {name: (N,4,4) world <- mesh}, "provenance": record}. Missing frames are NaN."""
    out_dir = Path(root) / manifest["outputs_dir"]
    c2w = np.load(out_dir / CAMERA[cam]["file"])["camera_to_world"].astype(float)
    if len(c2w) < n_frames:
        c2w = np.concatenate([c2w, np.full((n_frames - len(c2w), 4, 4), np.nan)])
    objects = {}
    for name in manifest["ground_truth"]["bodies"]:
        d = out_dir / OBJECT[obj]["dir"]
        if len(manifest["ground_truth"]["bodies"]) > 1:
            d = d / name
        T_cm = load_pose_dir(d, n_frames)
        objects[name] = c2w[:n_frames] @ T_cm
    uses = set(CAMERA[cam]["mocap_used_for"]) | set(OBJECT[obj]["mocap_used_for"])
    src = "video" if not uses else ("mocap" if "object_trajectory" in uses else "mixed")
    rec = provenance.make(src, sorted(uses), depth_source=manifest["inputs"].get("depth_source", "unknown"),
                          camera_stage=cam, object_stage=obj)
    return {"objects": objects, "provenance": rec}
