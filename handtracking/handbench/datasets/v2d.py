"""V2D Track 3 dev set, cam_b (and cam_c) held out. References from handbench.build_refs."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from handbench import paths
from handbench.adapters.v2d_upstream import UpstreamCamera, anycalib_dir
from handbench.calib import load_rig
from handbench.datasets.base import Dataset, EpisodeRefs
from handbench.refs_io import hand_track_ids

MIN_MASK_PX = 400
SIDES = ("L", "R")


class StereoCrops:
    """refs/<ep>/stereo.npz: half-res SGBM depth crops around visible hands in the benchmark camera."""

    def __init__(self, path: Path):
        self.z = np.load(path)
        self.meta = json.loads(str(self.z["meta"]))
        self.scale = self.meta["scale"]
        self.boxes = self.z["boxes"]

    def get(self, i, side):
        key = f"d_{i}_{side}"
        if key not in self.z.files:
            return None
        b = self.boxes[i, SIDES.index(side)]
        d = self.z[key].astype(np.float32)
        m = np.unpackbits(self.z[f"m_{i}_{side}"])[: d.size].reshape(d.shape).astype(bool)
        return d, m, b


class V2DDev(Dataset):
    name = "v2d"

    def __init__(self, set_name: str = "v2d_dev"):
        self.set_name = set_name
        self.cams = load_rig(paths.rig_json())

    def episodes(self):
        return paths.episodes(self.set_name)

    def refs(self, ep) -> EpisodeRefs:
        ep = paths.ep_name(ep)
        rd = paths.REFS / ep
        kp = np.load(rd / "kp2d.npz")
        tri = np.load(rd / "tri.npz")
        n = kp["kp"].shape[0]
        area = kp["mask_area_a"]
        U = UpstreamCamera(paths.CACHE / ep, self.cams["ego_cam_a"], anycalib_dir(ep))
        tracks = hand_track_ids(ep)
        mb_dir = rd / "masks_b"
        score = kp["score"]
        mask_b_ok = None
        if (rd / "kp_gate.npz").exists():  # mask-consistency gate (build_refs --gate)
            g = np.load(rd / "kp_gate.npz")
            score = score * g["keep"][..., None]
            mask_b_ok = g["mask_b_ok"]

        def mask_fn(view, i, side):
            if view == "ego_cam_a":
                m = None
                for tid, s in tracks.items():
                    if s != side:
                        continue
                    x = cv2.imread(str(paths.CACHE / ep / "masks" / str(tid) / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
                    if x is not None:
                        x = U.warp_to_bench(x) > 0
                        m = x if m is None else (m | x)
                return m
            if view == "ego_cam_b" and mb_dir.exists():
                if mask_b_ok is not None and not mask_b_ok[i, SIDES.index(side)]:
                    return None
                x = cv2.imread(str(mb_dir / side / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
                return None if x is None else x > 0
            if view == "ego_cam_c" and (rd / "masks_c").exists():
                x = cv2.imread(str(rd / "masks_c" / side / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
                return None if x is None else x > 0
            return None

        st = rd / "stereo.npz"
        return EpisodeRefs(
            name=ep, n_frames=n, fps=paths.fps(), cams=self.cams, visible=area >= MIN_MASK_PX,
            kp2d=kp["kp"], kp2d_score=score, kp_views=tuple(str(v) for v in kp["views"]),
            joints_gt={"tri_abc": tri["tri_abc"], "tri_ab": tri["tri_ab"]}, mask_fn=mask_fn,
            stereo=StereoCrops(st) if st.exists() else None)


def get(set_name: str) -> Dataset:
    if set_name in ("v2d", "v2d_dev"):
        return V2DDev("v2d_dev")
    if set_name == "hot3d":
        from handbench.datasets.hot3d import HOT3D

        return HOT3D()
    raise KeyError(set_name)
