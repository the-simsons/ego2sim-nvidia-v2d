"""Per-frame HaMeR hands for the multi-view hand stage's fallback (multiview_hands.py).

Runs the upstream per-track HaMeR regression (video_to_data v2d.hamer.lib.masks_to_hands, imported from an existing
environment, never edited) on the pipeline's SAM2 hand masks, and returns its hands in the stage's camera.

  inputs      undistorted cam_a frames (written here from the raw video with the rig calibration, the camera the
              masks are in) + hand_tracks.json (SAM2 track id -> left / right, from the stage's track assignment)
  HaMeR       crop on each mask (bbox x1.7), MANO per (track, frame); its camera is a virtual pinhole (focal
              5000/256 * max(W, H), principal point at the image centre)
  output      per side and frame: joints (21,3), vertices (778,3), betas, hand pose in cam_a: the hand (its shape
              unchanged) is moved so its centroid lands on the same pixel at the depth that gives it the same image
              size under cam_a's focal (depth * f_cam / f_virtual), as upstream align_hands does before its depth
              alignment. That depth is HaMeR's size-based guess, not metric: the caller corrects it (keypoint shift /
              stereo scale) before use.

Environment: the first existing of --hamer-python / $HAMER_PYTHON / ~/v2d/envs/v2d_hamer/.venv/bin/python /
handtracking/egofound/methods/.venv-e0/bin/python; weights likewise (--hamer-weights / $HAMER_WEIGHTS / ...): a
folder with _DATA/hamer_ckpts and _DATA/data/mano. Outputs are cached under WORK/hamer/ and reused.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

HB = Path(__file__).resolve().parents[2]  # handtracking/
PYTHON_CANDIDATES = (Path.home() / "v2d" / "envs" / "v2d_hamer" / ".venv" / "bin" / "python",
                     HB / "egofound" / "methods" / ".venv-e0" / "bin" / "python")
WEIGHT_CANDIDATES = (Path.home() / "v2d" / "vtd_pinned" / "reconstruction" / "data" / "weights" / "hamer",
                     HB / "egofound" / "methods" / "weights-e0" / "hamer")
BBOX_EXPANSION = 1.7  # upstream default
MASK_MIN_PIXELS = 1  # upstream value when HaMeR is the primary hand source


def find(explicit: str, env: str, candidates) -> Path | None:
    for p in ([explicit] if explicit else []) + ([os.environ[env]] if os.environ.get(env) else []) + list(candidates):
        if p and Path(p).exists():
            return Path(p)
    return None


def write_inputs(work: Path, raw_video: Path, A, tracks: dict, n: int) -> Path:
    """Undistorted cam_a frames (<work>/hamer_frames/<frame>.png) + hand_tracks.json; returns the tracks file."""
    frames = work / "hamer_frames"
    if not (frames.exists() and len(list(frames.glob("*.png"))) >= n):
        frames.mkdir(parents=True, exist_ok=True)
        maps = cv2.initUndistortRectifyMap(A.K, A.dist, np.eye(3), A.K, A.size, cv2.CV_32FC1)
        cap = cv2.VideoCapture(str(raw_video))
        for i in range(n):
            ok, im = cap.read()
            if not ok:
                break
            cv2.imwrite(str(frames / f"{i:06d}.png"), cv2.remap(im, *maps, cv2.INTER_LINEAR))
    tracks_path = work / "hamer_hand_tracks.json"
    tracks_path.write_text(json.dumps({"tracks": [{"object_id": int(t) if str(t).isdigit() else t, "role": "hand",
                                                   "is_right": side == "right"} for t, side in tracks.items()]},
                                      indent=1))
    return tracks_path


def run(work: Path, raw_video: Path, A, tracks: dict, n: int, masks_dir: str, gpu: str,
        python: str = "", weights: str = "") -> Path | None:
    """Run masks_to_hands once (cached); returns the output folder, or None when no HaMeR environment is found."""
    out = work / "hamer"
    if (out / ".done").exists():
        return out
    py = find(python, "HAMER_PYTHON", PYTHON_CANDIDATES)
    wd = find(weights, "HAMER_WEIGHTS", WEIGHT_CANDIDATES)
    if py is None or wd is None:
        print(f"[multiview] HaMeR fallback off: no environment ({py}) or weights ({wd}) found", flush=True)
        return None
    tracks_path = write_inputs(work, raw_video, A, tracks, n)
    cmd = [str(py), "-m", "v2d.hamer.lib.masks_to_hands", "--frames_dir", str(work / "hamer_frames"),
           "--masks_dir", str(masks_dir), "--tracks_path", str(tracks_path), "--output_dir", str(out),
           "--weights_dir", str(wd), "--bbox_expansion", str(BBOX_EXPANSION), "--mask_min_pixels", str(MASK_MIN_PIXELS)]
    print("[multiview] HaMeR:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYOPENGL_PLATFORM="egl"))
    (out / ".done").write_text(json.dumps({"python": str(py), "weights": str(wd), "cmd": cmd}))
    return out


def to_camera(X: np.ndarray, f_virtual: float, size, K: np.ndarray, centroid: np.ndarray | None = None) -> np.ndarray:
    """A hand (points X, (n,3)) in HaMeR's virtual pinhole (focal f_virtual, principal point at the image centre) ->
    cam_a (K): translated (shape kept) so that `centroid` (default: X's mean) keeps its pixel and its depth becomes
    depth * f / f_virtual (same image size under cam_a's focal f)."""
    W, H = size
    c = X.mean(0) if centroid is None else centroid
    uv = c[:2] / c[2] * f_virtual + np.array([W / 2.0, H / 2.0])
    z = c[2] * float(np.mean(np.diag(K)[:2])) / f_virtual
    c_new = np.r_[(uv - K[:2, 2]) / np.diag(K)[:2] * z, z]
    return X - c + c_new


def hands(out: Path, tracks: dict, mano, K: np.ndarray, n: int, masks_dir: str) -> dict:
    """-> {side: (J (n,21,3), V (n,778,3), betas (n,10), hand_pose (n,45))}, NaN where HaMeR has no hand. When a side has
    several SAM2 tracks on a frame, the one with the larger mask is used."""
    res = {}
    for side in ("left", "right"):
        J, V = np.full((n, 21, 3), np.nan), np.full((n, 778, 3), np.nan)
        B, P = np.zeros((n, 10)), np.zeros((n, 45))
        area = np.zeros(n)
        is_right = side == "right"
        for t, s in tracks.items():
            if s != side:
                continue
            for f in sorted((out / str(t)).glob("*.json")):
                rec = json.loads(f.read_text())
                i = int(rec["frame_idx"])
                if i >= n:
                    continue
                m = cv2.imread(str(Path(masks_dir) / str(t) / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
                a = float((m > 0).sum()) if m is not None else 0.0
                if a <= area[i]:
                    continue
                mo = rec["mano"]
                go, hp, be = (np.asarray(mo[k], float).ravel() for k in ("global_orient", "hand_pose", "betas"))
                v, j = mano.forward(go, hp, be, np.asarray(rec["camera"]["cam_t"], float), is_right, 1.0)
                if not (v[0][:, 2] > 0.05).all():
                    continue
                fv, size = float(rec["camera"]["focal_length"]), rec["image_size"]
                c = v[0].mean(0)  # joints and vertices move together (same translation)
                J[i], V[i] = to_camera(j[0], fv, size, K, c), to_camera(v[0], fv, size, K, c)
                B[i], P[i], area[i] = be, hp, a
        res[side] = (J, V, B, P)
    return res
