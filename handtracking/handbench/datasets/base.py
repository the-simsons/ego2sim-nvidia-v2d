"""Dataset adapter interface. A dataset yields episodes; each episode gives the references the metrics need.
Every field may be None when the dataset does not have it; metrics that need it are then skipped (NaN)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeRefs:
    name: str
    n_frames: int
    fps: float
    cams: dict  # handbench.calib.Cam per view; geometry in the benchmark (cam_a) frame
    visible: np.ndarray  # (n, 2) bool: hand visible in the input view (scored frames, rule 1.4)
    kp2d: np.ndarray | None = None  # (n, 2, V, 21, 2) published px per view
    kp2d_score: np.ndarray | None = None  # (n, 2, V, 21)
    kp_views: tuple = ()
    joints_gt: dict = field(default_factory=dict)  # name -> (n, 2, 21, 3) cam_a frame, NaN = no reference
    mask_fn: object = None  # (view, frame, side) -> bool mask (H, W) in that view's image (benchmark px for cam_a)
    stereo: object = None  # StereoCrops or None
    true_gt: bool = False  # True when joints_gt is real ground truth (e.g. HOT3D), False for pseudo-GT

    def mask(self, view, i, side):
        return self.mask_fn(view, i, side) if self.mask_fn else None


class Dataset:
    name = "base"

    def episodes(self) -> list[str]:
        raise NotImplementedError

    def refs(self, ep: str) -> EpisodeRefs:
        raise NotImplementedError
