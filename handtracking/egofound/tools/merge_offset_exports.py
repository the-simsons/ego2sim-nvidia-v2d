"""Merge EgoFound3R exports of offset passes (tools/offset_clips.py) into one export where EVERY frame is a frame
EgoFound3R predicted itself (its own 10 fps schedule hits different frames in each pass).

Per frame and hand the pass with a real prediction is used (the original pass first, then offsets in order); the
export's per-frame fields (joints, vertices, MANO fit, K_pred, ...) are copied from that pass at frame t - offset.
Fails if any frame is not a prediction frame of some pass. `pass_offset` (T,2) records the source (-1 = no hand).

  ../.venv-bench/bin/python tools/merge_offset_exports.py --dataset assemblyhands \
      --passes data/egofound3r:0,data/egofound3r_offset1:1 --out data/egofound3r_fullrate
Writes <out>/<dataset>/<clip>/{export.npz, run_meta.json}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PER_FRAME = ("joints_native", "verts778", "valid", "predicted", "global_orient", "hand_pose", "betas", "transl",
             "fit_err_mean_mm", "fit_err_max_mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--passes", required=True, help="root:offset,... (first = the original clip)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    passes = [(Path(p.split(":")[0]), int(p.split(":")[1])) for p in a.passes.split(",")]
    base_root = passes[0][0] / a.dataset
    for run in sorted(p for p in base_root.iterdir() if (p / "export.npz").exists()):
        clip = run.name
        ex = [(dict(np.load(r / a.dataset / clip / "export.npz")), k) for r, k in passes]
        T = ex[0][0]["valid"].shape[0]
        out = {k: v.copy() for k, v in ex[0][0].items()}
        src = np.where(out["predicted"], 0, -1).astype(np.int8)
        h_frame = np.zeros(T, bool)
        Kp = out["K_pred"].copy()
        for e, k in ex:
            hi = e["high_frame_indices"] + k
            h_frame[hi[hi < T]] = True
        for e, k in ex[1:]:
            n = min(T - k, e["valid"].shape[0])
            take = e["predicted"][:n] & (src[k:k + n] < 0)
            for f in PER_FRAME:
                out[f][k:k + n][take] = e[f][:n][take]
            src[k:k + n][take] = k
            rows = take.any(1)
            Kp[k:k + n][rows] = e["K_pred"][:n][rows]
        if not h_frame.all():
            raise SystemExit(f"{clip}: frames {np.nonzero(~h_frame)[0][:10]} are not prediction frames of any pass")
        out["K_pred"], out["pass_offset"] = Kp, src
        out["high_frame_indices"] = np.arange(T)
        od = Path(a.out) / a.dataset / clip
        od.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(od / "export.npz", **out)
        metas = [json.loads((r / a.dataset / clip / "run_meta.json").read_text()) for r, _ in passes]
        meta = dict(metas[0], wall_time_s=sum(m["wall_time_s"] for m in metas),
                    passes=[{"root": str(r), "offset": k, "wall_time_s": m["wall_time_s"]}
                            for (r, k), m in zip(passes, metas)])
        (od / "run_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{clip}: all {T} frames are prediction frames; hands predicted {int(out['predicted'].sum())} "
              f"(original pass {int((src == 0).sum())}, offsets {int((src > 0).sum())})")


if __name__ == "__main__":
    main()
