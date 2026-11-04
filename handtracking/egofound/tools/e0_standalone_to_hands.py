"""Standalone E0 output (tools/run_e0_standalone.py, upstream layout) -> handbench hands.npz for Track 3 episodes.

Reuses handbench's adapter for the cached upstream run (handbench.adapters.upstream_hamer.convert_episode) with the run
root pointed at data/e0_standalone/track3/; the camera mapping (AnyCalib-undistorted pinhole -> benchmark camera) is
the same because the standalone run is fed the cached run's AnyCalib-undistorted video and intrinsics.

  ../.venv-bench/bin/python tools/e0_standalone_to_hands.py --episodes 41 [--method E0-standalone]
Writes results/track3/<method>/<ep>/hands.npz. E0_STANDALONE_RUNS=<dir> reads runs from another folder than
data/e0_standalone/track3 (e.g. data/e0_standalone_cached_settings/track3).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("E0_STANDALONE_RUNS", ROOT / "data" / "e0_standalone" / "track3")).resolve()
os.environ["HANDBENCH_CACHE"] = str(RUNS)  # before handbench.paths is imported
sys.path.insert(0, str(ROOT.parent))
from handbench import format as hf  # noqa: E402
from handbench import paths  # noqa: E402
from handbench.adapters.upstream_hamer import convert_episode, git_commit  # noqa: E402
from handbench.calib import load_rig  # noqa: E402
from handbench.mano import Mano  # noqa: E402

RESULTS = ROOT / "results" / "track3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--method", default="E0-standalone")
    ap.add_argument("--dir", default="hamer_aligned_filled")
    a = ap.parse_args()
    mano, cams = Mano(), load_rig(paths.rig_json())
    for ep in [paths.ep_name(x) for x in a.episodes.split(",")]:
        t0 = time.time()
        run_meta = json.loads((RUNS / ep / "run_meta.json").read_text())
        arr = convert_episode(ep, a.dir, mano, cams)
        meta = {"method": a.method, "version": "standalone E0: " + run_meta["upstream"],
                "checkpoint": json.dumps(run_meta["weights"]), "args": dict(run_meta["args"], source_dir=a.dir),
                "gpu": run_meta["gpu"], "wall_time_s": run_meta["wall_time_s"],
                "nvidia_smi_peak_mib": run_meta.get("nvidia_smi_peak_mib"), "step_wall_time_s": run_meta["step_wall_time_s"],
                "hamer_patch": run_meta.get("hamer_patch"), "fps": paths.fps(), "consumed_stereo_depth": False,
                "depth_source": "MoGe (monocular), AnyCalib-undistorted intrinsics", "repo_commit": run_meta["repo_commit"],
                "object_detections": run_meta["object_detections"], "adapter_commit": git_commit(),
                "adapter_wall_time_s": time.time() - t0}
        hf.save(RESULTS / a.method / ep / "hands.npz", arr, meta)
        print(f"{ep}: {arr['valid'].sum()} valid rows ({(arr['source'] == 'infilled').sum()} infilled)")


if __name__ == "__main__":
    main()
