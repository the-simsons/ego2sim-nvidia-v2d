"""Common hand-track format every method writes: results/<method>/<episode>/hands.npz

One row per (frame, side) for EVERY frame of the episode and both sides (missing hands: valid=False), sorted by
(frame_idx, side). All geometry in the benchmark frame: undistorted cam_a OpenCV camera (handbench.calib), metres.

  frame_idx (N,) int32         t (N,) float64 seconds (parquet timestamps; real fps ~20.25, never assumed)
  side (N,) '<U1' 'L'|'R'      valid (N,) bool          source (N,) '<U8' 'observed'|'infilled'|'none'
  conf (N,) float32 in [0,1]
  global_orient (N,3) hand_pose (N,45) betas (N,10) transl (N,3) scale (N,)   MANO params, handbench.mano convention
  joints_cam (N,21,3) float32  OpenPose joint order      verts_cam (N,778,3) float16 (optional; regenerate from MANO)
  joints_world (N,21,3), T_world_cam (N,4,4)             optional, if the method has a world frame
  meta: JSON string (method, version/commit, checkpoint, args, gpu, wall_time_s, fps, consumed_stereo_depth, notes)

Rows with source='infilled' are never scored for accuracy; they only enter the coverage line.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIDES = ("L", "R")
SOURCES = ("observed", "infilled", "none")
REQUIRED = ["frame_idx", "t", "side", "valid", "source", "conf", "global_orient", "hand_pose", "betas", "transl",
            "joints_cam"]
SHAPES = {"global_orient": (3,), "hand_pose": (45,), "betas": (10,), "transl": (3,), "joints_cam": (21, 3),
          "verts_cam": (778, 3), "joints_world": (21, 3), "T_world_cam": (4, 4), "scale": ()}
META_KEYS = ["method", "version", "checkpoint", "args", "gpu", "wall_time_s", "fps", "consumed_stereo_depth"]


@dataclass
class HandTracks:
    arrays: dict
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.arrays["frame_idx"])

    def __getitem__(self, k):
        return self.arrays[k]

    def get(self, k, default=None):
        return self.arrays.get(k, default)

    @property
    def n_frames(self) -> int:
        return int(self.arrays["frame_idx"].max()) + 1 if len(self) else 0

    def rows(self, side: str, scored_only: bool = False) -> np.ndarray:
        """Row indices of one side ordered by frame (one row per frame)."""
        m = self.arrays["side"] == side
        if scored_only:
            m &= self.scored_mask()
        return np.nonzero(m)[0]

    def scored_mask(self) -> np.ndarray:
        return self.arrays["valid"] & (self.arrays["source"] == "observed")

    def by_side(self, side: str, key: str) -> np.ndarray:
        """(n_frames, ...) array of one field for one side (rows are complete per frame)."""
        return self.arrays[key][self.rows(side)]


def empty(n_frames: int, t: np.ndarray, with_verts: bool = False) -> dict:
    N = 2 * n_frames
    a = {"frame_idx": np.repeat(np.arange(n_frames, dtype=np.int32), 2),
         "t": np.repeat(np.asarray(t, np.float64)[:n_frames], 2),
         "side": np.tile(np.array(SIDES, "<U1"), n_frames),
         "valid": np.zeros(N, bool), "source": np.full(N, "none", "<U8"), "conf": np.zeros(N, np.float32),
         "global_orient": np.zeros((N, 3)), "hand_pose": np.zeros((N, 45)), "betas": np.zeros((N, 10)),
         "transl": np.zeros((N, 3)), "scale": np.ones(N), "joints_cam": np.full((N, 21, 3), np.nan, np.float32)}
    if with_verts:
        a["verts_cam"] = np.full((N, 778, 3), np.nan, np.float16)
    return a


def row_index(frame: int, side: str) -> int:
    return 2 * frame + SIDES.index(side)


def save(path: str | Path, arrays: dict, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    errs = validate_arrays(arrays, meta)
    if errs:
        raise ValueError(f"{path}: invalid hand tracks:\n  " + "\n  ".join(errs))
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load(path: str | Path, validate: bool = True) -> HandTracks:
    z = np.load(path, allow_pickle=False)
    arrays = {k: z[k] for k in z.files if k != "meta"}
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    if validate:
        errs = validate_arrays(arrays, meta)
        if errs:
            raise ValueError(f"{path}: invalid hand tracks:\n  " + "\n  ".join(errs))
    return HandTracks(arrays, meta)


def validate_arrays(a: dict, meta: dict | None = None, check_mano: bool = False, mano=None) -> list[str]:
    errs = []
    for k in REQUIRED:
        if k not in a:
            errs.append(f"missing field {k}")
    if errs:
        return errs
    N = len(a["frame_idx"])
    for k, v in a.items():
        if len(v) != N:
            errs.append(f"{k}: length {len(v)} != {N}")
        if k in SHAPES and v.shape[1:] != SHAPES[k]:
            errs.append(f"{k}: shape {v.shape[1:]} != {SHAPES[k]}")
    if errs:
        return errs
    if N % 2:
        errs.append("odd number of rows: need one row per (frame, side)")
    fi = a["frame_idx"]
    exp = np.repeat(np.arange(N // 2), 2)
    if not np.array_equal(fi, exp):
        errs.append("rows must be (frame 0 L, frame 0 R, frame 1 L, ...) covering every frame")
    if not np.array_equal(a["side"], np.tile(np.array(SIDES), N // 2)):
        errs.append("side must alternate L, R")
    if not set(np.unique(a["source"])) <= set(SOURCES):
        errs.append(f"unknown source values {set(np.unique(a['source'])) - set(SOURCES)}")
    t = a["t"][::2]
    if len(t) > 1:
        dt = np.diff(t)
        if (dt <= 0).any():
            errs.append("t must be strictly increasing")
        elif abs(1 / np.median(dt) - 30.0) < 0.05 and meta and abs(meta.get("fps", 30) - 30) > 0.5:
            errs.append("t looks like an assumed 30 fps")
    v = a["valid"]
    if (a["source"][v] == "none").any():
        errs.append("valid rows need source observed|infilled")
    if (a["source"][~v] != "none").any():
        errs.append("invalid rows must have source='none'")
    J = a["joints_cam"][v]
    if not np.isfinite(J).all():
        errs.append("non-finite joints on valid rows")
    elif len(J) and (J[..., 2] <= 0).any():
        errs.append("valid joints behind the camera (z <= 0)")
    c = a["conf"]
    if ((c < 0) | (c > 1)).any():
        errs.append("conf outside [0, 1]")
    if meta is not None:
        miss = [k for k in META_KEYS if k not in meta]
        if miss:
            errs.append(f"meta missing {miss}")
    if check_mano and v.any():
        from handbench.mano import Mano

        mano = mano or Mano()
        idx = np.nonzero(v)[0][:: max(1, v.sum() // 50)]
        _, Jm = mano.forward(a["global_orient"][idx], a["hand_pose"][idx], a["betas"][idx], a["transl"][idx],
                             a["side"][idx] == "R", a["scale"][idx] if "scale" in a else 1.0)
        err = np.linalg.norm(Jm - a["joints_cam"][idx], axis=-1).max(1)
        if np.median(err) > 0.005:
            errs.append(f"MANO params do not reproduce joints_cam (median max-joint err {np.median(err) * 1000:.1f} mm)")
    return errs


def validate(path: str | Path, check_mano: bool = True) -> list[str]:
    z = np.load(path, allow_pickle=False)
    a = {k: z[k] for k in z.files if k != "meta"}
    meta = json.loads(str(z["meta"])) if "meta" in z.files else None
    return validate_arrays(a, meta, check_mano=check_mano)
