"""Presentation video of one method on one Track 3 episode: the undistorted colour camera (cam_a) with the method's
hand meshes, and a plain text label in the top-left corner (outlined text, no box).

  ../.venv-bench/bin/python tools/render_method_video.py --episode 41 --method F3 \
      --results results/track3 --label "EgoFound3R + stereo" --out results/videos_for_team/ep41_egofound3r_stereo.mp4
--results is the folder holding <method>/<episode>/hands.npz (handbench layout). Hands on frames the method did not
predict itself (interpolated, e.g. EgoFound3R's in-between frames) are drawn too, in the same colours, so the video
does not flicker. Left hand orange, right hand red.
--stage-output DIR instead reads the pipeline hand stage's own output (DIR/{left,right}/<frame>.json, e.g.
$E/hands/hands_by_side); with it, frames whose "correction" is "fallback" (the per-frame HaMeR hand) are drawn green
and a small legend says so.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from handbench import format as hf  # noqa: E402
from handbench import paths  # noqa: E402
from handbench.calib import load_rig  # noqa: E402
from handbench.mano import Mano  # noqa: E402
from handbench.overlay import COL, draw_mesh  # noqa: E402
from handbench.refs_io import VideoReader  # noqa: E402


def label(img, text):
    """White text with a dark outline, top-left corner, sized to the frame."""
    scale = img.shape[1] / 1100
    th = max(2, int(round(2 * scale)))
    org = (int(24 * scale), int(52 * scale))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (20, 20, 20), th * 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)
    return img


FALLBACK_COL = (80, 200, 60)  # BGR green


def legend(img, active: bool):
    """Bottom-left legend for the stage output: hand colours, the fallback colour highlighted on fallback frames."""
    scale = img.shape[1] / 1800
    th = max(1, int(round(2 * scale)))
    y = img.shape[0] - int(20 * scale)
    for k, (txt, col) in enumerate((("left", COL["L"]), ("right", COL["R"]), ("HaMeR fallback", FALLBACK_COL))):
        x = int(20 * scale) + k * int(230 * scale)
        cv2.circle(img, (x + int(10 * scale), y - int(8 * scale)), int(9 * scale), col, -1, cv2.LINE_AA)
        w = th + (2 if k == 2 and active else 0)
        cv2.putText(img, txt, (x + int(26 * scale), y), cv2.FONT_HERSHEY_DUPLEX, 0.7 * scale, (20, 20, 20), w * 4,
                    cv2.LINE_AA)
        cv2.putText(img, txt, (x + int(26 * scale), y), cv2.FONT_HERSHEY_DUPLEX, 0.7 * scale,
                    FALLBACK_COL if k == 2 and active else (255, 255, 255), w, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--method", default="")
    ap.add_argument("--results", default="", help="folder with <method>/<episode>/hands.npz")
    ap.add_argument("--stage-output", default="", help="pipeline hand stage output folder ({left,right}/<frame>.json)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--crf", type=int, default=23)
    a = ap.parse_args()
    ep = paths.ep_name(a.episode)
    A = load_rig(paths.rig_json())["ego_cam_a"]
    mapa = cv2.initUndistortRectifyMap(A.K, A.dist, np.eye(3), A.K, A.size, cv2.CV_32FC1)
    mano = Mano()
    fallback_rows = set()
    if a.stage_output:
        import json

        from handbench.adapters.team_objfill import convert
        arr, _ = convert(int(ep.split("_")[-1]), mano, root=a.stage_output)
        tr = hf.HandTracks(arr, {})
        for s_, side in enumerate(("left", "right")):
            for f in Path(a.stage_output, side).glob("*.json"):
                if json.loads(f.read_text()).get("correction") == "fallback":
                    fallback_rows.add(2 * int(f.stem) + s_)
    else:
        tr = hf.load(Path(a.results) / a.method / ep / "hands.npz")
    W = a.width - a.width % 2
    H = int(round(W * A.size[1] / A.size[0]))
    H -= H % 2
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", f"{paths.fps():.4f}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                          stdin=subprocess.PIPE)
    n = paths.num_frames(ep)
    for i, fr in VideoReader(ep, ("ego_cam_a",)):
        if i >= n:
            break
        img = cv2.remap(fr["ego_cam_a"], *mapa, cv2.INTER_LINEAR)
        for s, side in enumerate("LR"):
            r = 2 * i + s
            if not tr["valid"][r]:
                continue
            V, _ = mano.forward(tr["global_orient"][r], tr["hand_pose"][r], tr["betas"][r], tr["transl"][r],
                                side == "R", tr["scale"][r])
            V = V[0]
            img = draw_mesh(img, V[:, :2] / V[:, 2:3] * np.diag(A.K)[:2] + A.K[:2, 2], mano.faces_for(side == "R"),
                            FALLBACK_COL if r in fallback_rows else COL[side])
        img = label(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), a.label)
        if a.stage_output:
            legend(img, any(2 * i + s_ in fallback_rows for s_ in (0, 1)))
        ff.stdin.write(np.ascontiguousarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
