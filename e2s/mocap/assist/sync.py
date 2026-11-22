"""Video <-> mocap time offset for one clip: correlate the video motion signal (tools/box_image_motion.py) with the mocap
object speed. Writes <clip dir>/sync.json {"lag_s", "corr", "pts"} (mocap time = video time + lag), the starting lag
for e2s.mocap.assist.camera, which refines it. (Ported from the 2026-11-08 AIMS sync_clips.py, mocap half.)

    python -m e2s.mocap.assist.sync --clip BlueBox_Take1_000 [--root DATA] [--check]
--check compares with the sync.json already there instead of writing."""
import argparse
import json

import numpy as np

from bench import manifest as mf
from e2s.mocap import motive
from e2s.mocap.sync import best_lag, mocap_speed


def lag_for(m, root):
    from pathlib import Path
    root = Path(root)
    clip_dir = (root / m["inputs"]["frames"]).parent
    vm = json.load(open(clip_dir / "video_motion.json"))
    gt = m["ground_truth"]
    d = motive.load(root / gt["mocap_csv"])
    body = next(iter(gt["bodies"].values()))
    b = d["bodies"][body.get("rigid_body", next(iter(gt["bodies"])))]
    sp = mocap_speed(d["time"], b["pos"])
    lag, corr, _ = best_lag(np.asarray(vm["pts"]), np.asarray(vm["signal"]), d["time"], sp)
    return clip_dir, {"lag_s": lag, "corr": corr, "pts": vm["pts"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--root")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    m = mf.load(a.clip)
    clip_dir, rec = lag_for(m, mf.data_root(a.root))
    print(f"{a.clip}: lag {rec['lag_s']:.3f} s (mocap t = video t + lag), corr {rec['corr']:.3f}")
    if a.check:
        old = json.load(open(clip_dir / "sync.json"))
        d = abs(old["lag_s"] - rec["lag_s"])
        print(f"  existing sync.json lag {old['lag_s']:.3f} s: {'OK' if d < 0.0015 else 'DIFFERS'} ({d * 1000:.1f} ms)")
        raise SystemExit(0 if d < 0.0015 else 1)
    json.dump(rec, open(clip_dir / "sync.json", "w"))


if __name__ == "__main__":
    main()
