"""Score a pipeline on the benchmark clips against motion capture, and append one row to bench/history.csv.

    python -m bench.run --pipeline video_baseline --split val [--root DATA_ROOT] [--note "what changed"]
    python -m bench.run --pipeline camera=mocap,object=fp_box          # swap one stage, the other from cache

Writes bench/results/<time>_<pipeline>/{report.md, per_clip.json, per_frame/<clip>_<object>.npz}.
Run from the repository root (needs numpy, scipy). Data root on riddle: /usr1/e2s/rborah_v2d/aims."""
import argparse
import csv
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from bench import manifest as mf
from bench import metrics, stages
from e2s.mocap import ground_truth

HERE = Path(__file__).parent


def frame_times(m, root):
    p = Path(root) / m["inputs"]["frame_times"]
    return np.asarray(json.load(open(p))["pts"], float)


def git_state():
    """Short commit (+dirty) of this checkout; on a copy without .git (riddle), $E2S_COMMIT names the commit it was
    copied from, else "unknown"."""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
        if rev:
            dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                        text=True, cwd=HERE).stdout.strip())
            return rev + ("+dirty" if dirty else "")
    except OSError:
        pass
    return os.environ.get("E2S_COMMIT", "unknown") + ("+copy" if "E2S_COMMIT" in os.environ else "")


def fmt(x, k):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if "frac" in k or "coverage" in k or "auc" in k:
        return f"{x:.3f}"
    return f"{x:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--root")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-history", action="store_true")
    a = ap.parse_args(argv)
    root = mf.data_root(a.root)
    cam, obj = stages.parse(a.pipeline)
    clips = mf.all_clips(a.split)
    if not clips:
        raise SystemExit(f"no clips in split {a.split!r} (bench/clips/*.json)")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = HERE / "results" / f"{stamp}_{cam}+{obj}"
    (out / "per_frame").mkdir(parents=True)
    rows, per_frame_all, fps_all, prov = [], [], [], None
    for m in clips:
        t = frame_times(m, root)
        fps = (len(t) - 1) / (t[-1] - t[0])
        rec = stages.load(m, root, cam, obj, len(t))
        prov = rec["provenance"]
        gt = ground_truth.object_trajectories(m, root, t)
        for name, T_rec in rec["objects"].items():
            spec = m["ground_truth"]["bodies"][name]
            verts = metrics.load_obj_vertices(Path(root) / spec["mesh"])
            s, pf = metrics.object_metrics(T_rec, gt[name], verts, fps, spec.get("symmetry"))
            rows.append({"clip": m["clip"], "object": name, **s})
            per_frame_all.append(pf)
            fps_all.append(fps)
            np.savez_compressed(out / "per_frame" / f"{m['clip']}_{name}.npz", **pf)
    summary = metrics.pool(per_frame_all, fps_all)
    json.dump({"pipeline": a.pipeline, "camera": cam, "object": obj, "split": a.split, "provenance": prov,
               "summary": summary, "clips": rows}, open(out / "per_clip.json", "w"), indent=1)
    keys = ["coverage", "pos_cm_median", "pos_cm_p95", "rot_deg_median", "rot_deg_p95", "add_cm_median", "adds_cm_median",
            "add_auc", "adds_auc", "flip_frac", "flip_longest_s", "end_still_pos_cm"]
    lines = [f"# Benchmark: {a.pipeline} (camera={cam}, object={obj}), split {a.split}", "",
             f"commit {git_state()}, {stamp}; trajectory_source **{prov['trajectory_source']}**, mocap used for "
             f"{prov['mocap_used_for'] or 'nothing'}, depth {prov['depth_source']}", "",
             "| clip | object | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]
    for r in rows:
        lines.append(f"| {r['clip']} | {r['object']} | " + " | ".join(fmt(r.get(k), k) for k in keys) + " |")
    lines.append("| **pooled** | | " + " | ".join(fmt(summary.get(k), k) for k in keys) + " |")
    lines += ["", f"Alignment: one rigid transform per clip from the first still stretch of the ground truth "
              f"({rows[0]['alignment']}). Position in cm, rotation in degrees, AUC of ADD / ADD-S over 0-{metrics.AUC_MAX_CM:g} cm."]
    if len(clips) < 5:
        lines.append(f"Only {len(clips)} clip(s): treat differences as indicative.")
    report = "\n".join(lines) + "\n"
    open(out / "report.md", "w").write(report)
    print(report)
    if not a.no_history:
        hist = HERE / "history.csv"
        fields = ["time", "commit", "pipeline", "camera", "object", "split", "clips", "trajectory_source", "mocap_used_for",
                  "depth_source"] + metrics.SUMMARY_KEYS[:-2] + ["note"]
        new = not hist.exists()
        with open(hist, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({"time": stamp, "commit": git_state(), "pipeline": a.pipeline, "camera": cam, "object": obj,
                        "split": a.split, "clips": len(clips), "trajectory_source": prov["trajectory_source"],
                        "mocap_used_for": "+".join(prov["mocap_used_for"]), "depth_source": prov["depth_source"],
                        **{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in summary.items()}, "note": a.note})
        print(f"appended to {hist}")
    print(f"results: {out}")


if __name__ == "__main__":
    main()
