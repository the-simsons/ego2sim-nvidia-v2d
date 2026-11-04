"""EgoFound3R export.npz (tools/egofound3r_export.py) -> handbench hands.npz for one Track 3 episode.

EgoFound3R's camera = the benchmark camera (input video is undistorted cam_a, same centre and orientation), so its
camera-frame hands are used as they are, placed for EgoFound3R's own predicted focal (K_pred, recorded in meta).
Rows on frames EgoFound3R predicts (its H rate, 10 fps) are `observed`; its interpolation to the input rate is
`infilled` (never scored). MANO params are the fit to EgoFound3R's 778 vertices.

  --joints native   score EgoFound3R's own 21 joints and vertices (default; this is F0)
  --joints mano     score the fitted MANO joints / mesh instead (shows what the MANO fit changes)

  ../.venv-bench/bin/python tools/egofound3r_to_hands.py --episodes 41 --method F0
Reads <--runs, default data/egofound3r/track3>/<ep>/{export.npz,run_meta.json}; writes results/track3/<method>/<ep>/hands.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from handbench import format as hf  # noqa: E402
from handbench import paths  # noqa: E402
from handbench.mano import Mano  # noqa: E402
from handbench.refs_io import episode_times  # noqa: E402

RESULTS = ROOT / "results" / "track3"


def convert(ep: str, joints: str, mano: Mano, runs: Path):
    run = runs / ep
    e = np.load(run / "export.npz")
    run_meta = json.loads((run / "run_meta.json").read_text())
    n = paths.num_frames(ep)
    T = min(n, e["valid"].shape[0])
    a = hf.empty(n, episode_times(ep), with_verts=True)
    for s, side in enumerate(hf.SIDES):
        is_right = side == "R"
        idx = np.nonzero(e["valid"][:T, s])[0]
        if not len(idx):
            continue
        r = 2 * idx + s
        go, hp, be, tr = (e[k][idx, s] for k in ("global_orient", "hand_pose", "betas", "transl"))
        a["valid"][r] = True
        a["source"][r] = np.where(e["predicted"][idx, s], "observed", "infilled")
        a["conf"][r] = e["predicted"][idx, s].astype(np.float32)
        a["global_orient"][r], a["hand_pose"][r], a["betas"][r], a["transl"][r] = go, hp, be, tr
        if joints == "native":
            a["joints_cam"][r] = e["joints_native"][idx, s]
            a["verts_cam"][r] = e["verts778"][idx, s].astype(np.float16)
        else:
            V, J = mano.forward(go, hp, be, tr, is_right, 1.0)
            a["joints_cam"][r], a["verts_cam"][r] = J, V.astype(np.float16)
    xm = json.loads(str(e["meta"]))
    K = e["K_pred"][:T]
    meta = {"method": None, "version": f"EgoFound3R@{run_meta['egofound3r_commit']}",
            "checkpoint": run_meta["checkpoint"], "config": run_meta["config"],
            "args": {"joints": joints, "egofound3r": run_meta["args"]}, "gpu": run_meta["gpu"],
            "wall_time_s": run_meta["wall_time_s"], "nvidia_smi_peak_mib": run_meta["nvidia_smi_peak_mib"],
            "summary_performance": run_meta["summary_performance"], "fps": paths.fps(), "h_fps": xm["h_fps"],
            "consumed_stereo_depth": False, "repo_commit": run_meta["repo_commit"],
            "mano_fit": {k: xm[k] for k in xm if k.startswith("fit")},
            "K_pred_median": np.median(K, 0).tolist(), "K_pred_fx_range": [float(K[:, 0, 0].min()),
                                                                           float(K[:, 0, 0].max())],
            "notes": "observed = EgoFound3R H-rate frames; input-rate interpolation marked infilled"}
    return a, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--method", default="F0")
    ap.add_argument("--joints", choices=["native", "mano"], default="native")
    ap.add_argument("--runs", default=str(ROOT / "data" / "egofound3r" / "track3"),
                    help="exports root (e.g. data/egofound3r_f2/track3 for F2)")
    a = ap.parse_args()
    mano = Mano()
    for ep in [paths.ep_name(x) for x in a.episodes.split(",")]:
        arr, meta = convert(ep, a.joints, mano, Path(a.runs))
        meta["method"] = a.method
        meta["exports"] = str(Path(a.runs).resolve())
        hf.save(RESULTS / a.method / ep / "hands.npz", arr, meta)
        print(f"{ep}: {(arr['source'] == 'observed').sum()} observed, {(arr['source'] == 'infilled').sum()} infilled")


if __name__ == "__main__":
    main()
