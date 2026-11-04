"""Per-episode diagnostic Track 3 packages: one package per evaluation episode in which only that episode carries its policy
rollout, plus an all-masked control. Pose metrics average over the 20 episodes, so a package's AUC is
(AUC of its episode + leakage of the 19 masked ones) / 20.

Masked episode: the rollout's frame 0 is kept (the host aligns each episode from its frame-0 scene, so that alignment is
unchanged) and every later frame is moved +1 m along x, far beyond the 9 cm top ADD threshold: only frame 0 can score.
Its checkpoint_sha256 is replaced by "masked-probe-not-a-rollout" (metadata and report), so no masked row claims to be
the output of a checkpoint.

    AUC_episode ~= 20 * (AUC_package - AUC_control) + leak_episode,    leak_episode ~= 1 / frames of that episode
(the control holds the 20 frame-0 leaks; probe_summary.csv lists each episode's frame count and leak).

  make_probe_packages.py --source <results dir> --meshes <mesh dir> --out <dir> --commit <github commit URL> [--kit <kit root>]
Writes <out>/probe_ep<N>.parquet (20) and <out>/probe_control.parquet; the uploader writes the CD-O files itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MASK = "masked-probe-not-a-rollout"


def mask_episode(src_parquet: Path, src_json: Path, dst_dir: Path, offset: float) -> int:
    table = pq.read_table(src_parquet)
    step = table.column("step").to_numpy()
    x = table.column("achieved_position_x").to_numpy().astype(np.float32)
    x = np.where(step >= 1, x + offset, x).astype(np.float32)
    table = table.set_column(table.schema.get_field_index("achieved_position_x"), "achieved_position_x",
                             pa.array(x, type=pa.float32()))
    meta = dict(table.schema.metadata or {})
    meta[b"flash_chord.checkpoint_sha256"] = MASK.encode()
    meta[b"flash_chord.probe"] = f"masked: frames >= 1 moved +{offset} m in x".encode()
    pq.write_table(table.replace_schema_metadata(meta), dst_dir / src_parquet.name)
    report = json.loads(src_json.read_text())
    report["checkpoint_sha256"] = MASK
    report["checkpoint"] = MASK
    (dst_dir / src_json.name).write_text(json.dumps(report, indent=2))
    return int(step.max()) + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True); ap.add_argument("--meshes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--commit", required=True)
    ap.add_argument("--kit", type=Path, default=Path("/usr1/e2s/v2d_submission_kit"))
    ap.add_argument("--offset", type=float, default=1.0)
    args = ap.parse_args()
    episodes = sorted(int(p.stem.split("_")[1]) for p in args.source.glob("episode_*.parquet"))
    if len(episodes) != 20:
        raise SystemExit(f"expected 20 evaluation episodes in {args.source}, found {len(episodes)}")
    work = args.out / "work"
    frames = {}
    for target in episodes + ["control"]:
        d = work / (f"ep{target}" if target != "control" else "control")
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        for ep in episodes:
            stem = f"episode_{ep:06d}"
            if ep == target:
                shutil.copy2(args.source / f"{stem}.parquet", d); shutil.copy2(args.source / f"{stem}.json", d)
            else:
                frames[ep] = mask_episode(args.source / f"{stem}.parquet", args.source / f"{stem}.json", d, args.offset)
        name = f"probe_ep{target}" if target != "control" else "probe_control"
        out = args.out / f"{name}.parquet"
        run = subprocess.run([sys.executable, "tools/pack_submission.py", "--track", "track_3", "--results", str(d.resolve()),
                              "--meshes", str(args.meshes.resolve()), "--code-commit", args.commit, "-o", str(out.resolve())],
                             cwd=args.kit, capture_output=True, text=True)
        (args.out / f"{name}.pack.log").write_text(run.stdout + run.stderr)
        if run.returncode != 0:
            raise SystemExit(f"packing {name} failed, see {args.out / f'{name}.pack.log'}")
        print(f"{name}: {out}", flush=True)
    with open(args.out / "probe_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "frames", "leak_if_masked", "package"])
        for ep in episodes:
            w.writerow([ep, frames[ep], f"{1 / frames[ep]:.4f}", f"probe_ep{ep}.parquet"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
