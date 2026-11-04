"""Tables for AssemblyHands / HOT3D from score_groundtruth.py outputs (results/<dataset>/<method>/).

Two tables: every method on its own scored hand-frames (metrics.json), and every method on the COMMON hand-frames
(clip, frame, side) that all listed methods scored (per_frame.csv). Delta vs the baseline in brackets; best in bold
(lowest for errors / jitter / bone std, highest for PCK / coverage; signed bias: closest to 0).

  ../.venv-bench/bin/python tools/compare_groundtruth.py --dataset hot3d --methods E0,F0,F2,F3 [--baseline E0]
Writes results/<dataset>/REPORT_TABLES.md.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLS = [("mpjpe", "3D err (mm)", "low", "{:.1f}"), ("mpjpe_median", "median (mm)", "low", "{:.1f}"),
        ("depth", "depth (mm)", "low", "{:.1f}"), ("lateral", "lateral (mm)", "low", "{:.1f}"),
        ("bias", "depth bias (mm)", "zero", "{:+.1f}"), ("ca_mpjpe", "pose (mm)", "low", "{:.1f}"),
        ("pa_mpjpe", "PA (mm)", "low", "{:.1f}"), ("kp_px", "2D (px)", "low", "{:.1f}"),
        ("pck10", "PCK@10px", "high", "{:.3f}")]
EXTRA = [("jitter_ms2", "jitter (m/s²)", "low", "{:.2f}"), ("bone_std_mm", "bone std (mm)", "low", "{:.2f}"),
         ("coverage", "coverage", "high", "{:.1%}"), ("n_scored", "hand-frames", None, "{:d}")]


def best_idx(vals, mode):
    v = np.array(vals, float)
    if mode == "low":
        return int(np.nanargmin(v))
    if mode == "high":
        return int(np.nanargmax(v))
    if mode == "zero":
        return int(np.nanargmin(np.abs(v)))
    return None


def table(rows, cols, methods, base):
    head = "| method | " + " | ".join(c[1] for c in cols) + " |"
    out = [head, "|" + "---|" * (len(cols) + 1)]
    bests = {k: best_idx([r[k] for r in rows], mode) for k, _, mode, _ in cols}
    b = methods.index(base) if base in methods else None
    for i, (m, r) in enumerate(zip(methods, rows)):
        cells = []
        for k, _, mode, fmt in cols:
            s = fmt.format(r[k])
            if b is not None and i != b and mode is not None and mode != "zero":
                d = r[k] - rows[b][k]
                s += f" ({d:+.1%})" if fmt.endswith("%}") else f" ({d:+.{1 if 'f}' in fmt and '.1' in fmt else 2}f})"
            if bests[k] == i and mode is not None:
                s = f"**{s}**"
            cells.append(s)
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--baseline", default="E0")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    methods = a.methods.split(",")
    res = ROOT / "results" / a.dataset
    mets = [json.loads((res / m / "metrics.json").read_text()) for m in methods]
    rows = [{**x["raw"], **{k: x[k] for k, *_ in EXTRA}} for x in mets]
    md = [f"# {a.dataset}: {', '.join(methods)}", "", a.note, "",
          "## Each method on its own scored hand-frames", "", table(rows, COLS + EXTRA, methods, a.baseline), ""]
    if all("+offset" in x for x in mets):
        md += ["## After a per-joint offset in the hand frame (fitted on the other sequence)", "",
               table([x["+offset"] for x in mets], COLS, methods, a.baseline), ""]
    pf = {}
    for m in methods:
        with open(res / m / "per_frame.csv") as f:
            pf[m] = {(r["clip"], r["frame"], r["side"]): r for r in csv.DictReader(f)}
    common = set.intersection(*(set(v) for v in pf.values()))
    crow = [{k: float(np.mean([float(pf[m][key][f"raw_{k}"]) for key in common])) for k, *_ in COLS if k != "mpjpe_median"}
            | {"mpjpe_median": float(np.median([float(pf[m][key]["raw_mpjpe"]) for key in common]))} for m in methods]
    md += [f"## Common hand-frames ({len(common)} scored by every method)", "",
           table(crow, COLS, methods, a.baseline), ""]
    (res / "REPORT_TABLES.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
