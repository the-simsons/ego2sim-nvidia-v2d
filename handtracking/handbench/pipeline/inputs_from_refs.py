"""Build the multi-view hand stage's inputs for a Track 3 episode from handbench data, to run and score the stage off
the pipeline machine (where tools/ego_stereo_depth.py and the SAM2 step write them under ~/v2d/ego2sim/epEP/).

  STEREO/intrinsics_undist.json   cam_a undistorted pinhole (the rig's K), as tools/ego_stereo_depth.py writes it
  STEREO/stereo_depth/<f>.png     SGBM depth (cam_c + cam_b, handbench.stereo) in that camera, v2d inverse-depth
                                  encoding (raw = 65535 / (depth_m + 1), 0 = no depth)
  MASKS/<track>/<f>.png           SAM2 hand masks of the cached upstream run (tracks 2, 3), warped to that camera
                                  (handbench's cam_a reference masks), one folder per track

  python -m handbench.pipeline.inputs_from_refs --episode 41 --out data/pipeline_inputs/ep41
Then: python -m handbench.pipeline.multiview_hands --video-dir <Track 3 videos/chunk-000> --episode 41 \\
          --stereo-dir OUT/stereo --hand-masks OUT/hand_masks --out OUT/hands/hands_by_side
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from handbench import paths
from handbench.adapters.v2d_upstream import UpstreamCamera, anycalib_dir
from handbench.calib import load_rig
from handbench.refs_io import VideoReader, hand_track_ids
from handbench.stereo import StereoDepth


def encode_depth(d: np.ndarray) -> np.ndarray:
    raw = np.zeros(d.shape, np.uint16)
    ok = d > 0
    raw[ok] = np.clip(np.round(65535.0 / (d[ok] + 1.0)), 1, 65534).astype(np.uint16)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ep = paths.ep_name(a.episode)
    out = Path(a.out)
    cams = load_rig(paths.rig_json())
    A = cams["ego_cam_a"]
    st = out / "stereo"
    (st / "stereo_depth").mkdir(parents=True, exist_ok=True)
    W, H = A.size
    json.dump({"fx": float(A.K[0, 0]), "fy": float(A.K[1, 1]), "cx": float(A.K[0, 2]), "cy": float(A.K[1, 2]),
               "width": int(W), "height": int(H)}, open(st / "intrinsics_undist.json", "w"), indent=1)
    sd = StereoDepth(cams)
    U = UpstreamCamera(paths.CACHE / ep, A, anycalib_dir(ep))
    tracks = hand_track_ids(ep)
    n = 0
    for i, fr in VideoReader(ep, ("ego_cam_c", "ego_cam_b")):
        if i >= paths.num_frames(ep):
            break
        cv2.imwrite(str(st / "stereo_depth" / f"{i:06d}.png"), encode_depth(sd.compute(fr["ego_cam_c"], fr["ego_cam_b"])))
        for tid in tracks:
            x = cv2.imread(str(paths.CACHE / ep / "masks" / str(tid) / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if x is None:
                continue
            m = U.warp_to_bench(x) > 0
            if m.any():
                d = out / "hand_masks" / str(tid)
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{i:06d}.png"), m.astype(np.uint8) * 255)
        n += 1
        if i % 50 == 0:
            print(f"frame {i}", flush=True)
    print(f"{ep}: {n} frames -> {st}, {out / 'hand_masks'} (tracks {sorted(tracks)})")


if __name__ == "__main__":
    main()
