"""Copies of prepared clips that start OFFSET frames later, so a method with a fixed output rate below the video's
(EgoFound3R: hands at 10 fps) predicts the frames it skipped on the original clip. tools/merge_offset_exports.py
merges the passes back onto the original frame numbering.

  python3 tools/offset_clips.py --dataset assemblyhands --offset 1
Writes data/<dataset>_input_offset<k>/<clip>.mp4 (frames k.. of the original, same fps, lossless-ish re-encode CRF 12),
<clip>.json (original metadata + "frame_offset": k, n_frames reduced) and <clip>_rootK.json if the original has one.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--offset", type=int, required=True)
    a = ap.parse_args()
    src = ROOT / "data" / f"{a.dataset}_input"
    dst = ROOT / "data" / f"{a.dataset}_input_offset{a.offset}"
    dst.mkdir(parents=True, exist_ok=True)
    for meta_p in sorted(p for p in src.glob("*.json") if not p.stem.endswith("_rootK")):
        clip = meta_p.stem
        m = json.loads(meta_p.read_text())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src / f"{clip}.mp4"), "-vf",
                        f"select=gte(n\\,{a.offset}),setpts=N/FRAME_RATE/TB", "-r", str(m["fps"]), "-c:v", "libx264",
                        "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p", str(dst / f"{clip}.mp4")], check=True)
        m2 = dict(m, frame_offset=a.offset, n_frames=m["n_frames"] - a.offset, source_clip=str(src / f"{clip}.mp4"))
        if "frames" in m2:
            m2["frames"] = m["frames"][a.offset:]
        (dst / f"{clip}.json").write_text(json.dumps(m2, indent=1))
        rk = src / f"{clip}_rootK.json"
        if rk.exists():
            (dst / rk.name).write_text(rk.read_text())
        print(f"{clip}: offset {a.offset}, {m2['n_frames']} frames")


if __name__ == "__main__":
    main()
