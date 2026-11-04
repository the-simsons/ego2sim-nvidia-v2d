"""Known camera intrinsics for EgoFound3R: one external Root K sidecar per input clip.

EgoFound3R normally predicts its own focal; `--root-k-source external --root-k-file F` makes its root solver and hand
projection use F's pinhole K instead (egohandmetric_prompt/inference_intrinsics.py, schema
egofound3r.root_intrinsics_input.v1). K is given in the input video's full-frame pixels and rescaled by EgoFound3R to
its model input; that is exact here because the input resize is a pure scale (keep_aspect, no padding or crop).

  python3 tools/write_root_k_sidecars.py data/track3_input data/hot3d_input data/assemblyhands_input
Writes <dir>/<clip>_rootK.json for every <clip>.json with K and size (as written by the prepare_* tools).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    for d in map(Path, sys.argv[1:]):
        n = 0
        for meta_p in sorted(d.glob("*.json")):
            if meta_p.stem.endswith("_rootK"):
                continue
            m = json.loads(meta_p.read_text())
            W, H = m["size"]
            side = {"schema_version": "egofound3r.root_intrinsics_input.v1", "camera_model": "pinhole",
                    "coordinate_frame": "rgb_camera", "image_size_hw": [int(H), int(W)], "K": m["K"], "valid": True,
                    "source": str(meta_p)}
            (d / f"{meta_p.stem}_rootK.json").write_text(json.dumps(side, indent=1))
            n += 1
        print(f"{d}: {n} sidecars")


if __name__ == "__main__":
    main()
