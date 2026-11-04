"""Comparison table across methods -> results/REPORT_TABLES.md (included by the hand-written REPORT.md).

  python -m handbench.compare [--methods E0,E1b-huber,...] [--set v2d] [--baseline E0]

Two tables: (1) each method on all of its own scored hand-frames; (2) all methods on the COMMON hand-frames that
every listed method scores (fair when coverage differs). Best per column in bold, delta vs the baseline in brackets.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from handbench import paths

# (column, label, lower_is_better, fmt)
PRIMARY = [("mpjpe_tri_abc", "Tri MPJPE abc (mm)", True, "{:.1f}"),
           ("depth_tri_abc", "depth (mm)", True, "{:.1f}"),
           ("lateral_tri_abc", "lateral (mm)", True, "{:.1f}"),
           ("bias_tri_abc", "depth bias (mm)", None, "{:+.1f}"),
           ("mpjpe_tri_ab", "Tri MPJPE ab (mm)", True, "{:.1f}"),
           ("kp_b", "cam_b kp (px)", True, "{:.2f}"),
           ("pck10_b", "cam_b PCK@10", False, "{:.3f}"),
           ("kp_c", "cam_c kp (px)", True, "{:.2f}"),
           ("iou_b", "cam_b IoU", False, "{:.3f}")]
SECONDARY = [("iou_a", "cam_a IoU", False, "{:.3f}"),
             ("kp_a", "cam_a kp (px)", True, "{:.2f}"),
             ("ca_mpjpe_tri_abc", "centred MPJPE (mm)", True, "{:.1f}"),
             ("stereo_abs", "stereo resid (mm)*", True, "{:.1f}"),
             ("stereo_signed", "stereo bias (mm)*", None, "{:+.1f}"),
             ("jitter_accel_ms2", "jitter (m/s²)", True, "{:.2f}"),
             ("wrist_fast_frac", "% wrist >2 m/s", True, "{:.2%}"),
             ("bone_len_std_mm", "bone std (mm)", True, "{:.2f}"),
             ("betas_std", "betas std", True, "{:.3f}"),
             ("swaps_per_min", "swaps/min", True, "{:.2f}"),
             ("coverage_observed", "coverage", False, "{:.1%}"),
             ("infilled_frac_of_valid", "% infilled", None, "{:.1%}")]


def load_metrics(m, set_name):
    p = paths.RESULTS / m / f"metrics_{set_name}.json"
    return json.load(open(p)) if p.exists() else None


def load_metrics_subset(m, set_name, episodes):
    """Own-frames metrics restricted to some episodes, recomputed from the per-frame / per-episode CSVs."""
    pf, pe = paths.RESULTS / m / f"per_frame_{set_name}.csv", paths.RESULTS / m / f"per_episode_{set_name}.csv"
    if not (pf.exists() and pe.exists()):
        return None
    d = pd.read_csv(pf)
    d = d[d.episode.isin(episodes)]
    s = d[d.scored]
    out = {c: float(s[c].mean()) for c, *_ in PRIMARY + SECONDARY if c in s and s[c].notna().any()}
    e = pd.read_csv(pe)
    e = e[e.episode.isin(episodes)]
    out.update({c: float(e[c].mean()) for c, *_ in SECONDARY if c in e and c not in out})
    out["n_scored_hand_frames"] = int(len(s))
    m_json = load_metrics(m, set_name) or {}
    out["method_meta"] = m_json.get("method_meta", {})
    return out


def common_frame_metrics(methods, set_name, episodes=None):
    dfs = {}
    for m in methods:
        p = paths.RESULTS / m / f"per_frame_{set_name}.csv"
        if p.exists():
            d = pd.read_csv(p)
            if episodes:
                d = d[d.episode.isin(episodes)]
            dfs[m] = d[d.scored].set_index(["episode", "frame", "side"])
    if len(dfs) < 2:
        return None, 0
    keys = None
    for d in dfs.values():
        k = set(d.index[d["mpjpe_tri_abc"].notna()]) if "mpjpe_tri_abc" in d else set(d.index)
        keys = k if keys is None else keys & k
    keys = sorted(keys)
    out = {}
    for m, d in dfs.items():
        dd = d.loc[keys]
        out[m] = {c: float(dd[c].mean()) for c, *_ in PRIMARY + SECONDARY if c in dd}
    return out, len(keys)


def table(rows: dict, cols, baseline):
    methods = list(rows)
    head = "| method | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines = [head, sep]
    best = {}
    for c, _, lower, _ in cols:
        vals = [rows[m].get(c) for m in methods]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals and lower is not None:
            best[c] = min(vals) if lower else max(vals)
    for m in methods:
        cells = []
        for c, _, lower, fmt in cols:
            v = rows[m].get(c)
            if v is None or not np.isfinite(v):
                cells.append("–")
                continue
            s = fmt.format(v)
            if c in best and np.isclose(v, best[c]) and len(methods) > 1:
                s = f"**{s}**"
            b = rows.get(baseline, {}).get(c)
            if m != baseline and b is not None and np.isfinite(b) and lower is not None:
                d = v - b
                s += f" ({'+' if d >= 0 else ''}{fmt.format(d).replace('+', '')})" if "%" not in fmt else \
                     f" ({d * 100:+.1f}pp)"
            cells.append(s)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="")
    ap.add_argument("--set", default="v2d")
    ap.add_argument("--baseline", default="E0")
    ap.add_argument("--out", default=str(paths.RESULTS / "REPORT_TABLES.md"))
    ap.add_argument("--episodes", default="", help="restrict every table to these episodes (recomputed from CSVs)")
    a = ap.parse_args()
    episodes = [paths.ep_name(e) for e in a.episodes.split(",")] if a.episodes else None
    methods = a.methods.split(",") if a.methods else sorted(
        [p.name for p in paths.RESULTS.iterdir() if (p / f"metrics_{a.set}.json").exists()],
        key=lambda m: (m != a.baseline, m))
    rows = {m: (load_metrics_subset(m, a.set, episodes) if episodes else load_metrics(m, a.set)) for m in methods}
    rows = {m: r for m, r in rows.items() if r}
    meta_lines = []
    for m, r in rows.items():
        mm = r.get("method_meta", {})
        meta_lines.append(f"| {m} | {mm.get('version', '')} | {mm.get('checkpoint', '')} | "
                          f"{'yes' if mm.get('consumed_stereo_depth') else 'no'} | {mm.get('gpu', '')} | "
                          f"{mm.get('wall_time_s') if mm.get('wall_time_s') is not None else '–'} | "
                          f"{mm.get('fps_processing', '–')} | {r.get('n_scored_hand_frames', 0)} |")
    common, n_common = common_frame_metrics(list(rows), a.set, episodes)
    sub = f"; episodes {','.join(str(paths.ep_index(e)) for e in episodes)} only" if episodes else ""
    md = [f"## Primary metrics (set {a.set}{sub}, cam_b/cam_c held out; each method on its own scored hand-frames)", "",
          table(rows, PRIMARY, a.baseline), "",
          "## Secondary metrics", "", table(rows, SECONDARY, a.baseline), "",
          "\\* stereo residual uses SGBM depth from cam_b+cam_c: not a held-out metric, and circular for methods that "
          "consumed stereo depth.", ""]
    if common:
        md += [f"## Common hand-frames ({n_common} hand-frames scored by every method with a tri-abc reference)", "",
               table(common, PRIMARY[:7] + SECONDARY[:3], a.baseline), ""]
    md += ["## Runs", "", "| method | version | checkpoint | consumed stereo depth | GPU | wall time (s) | fps | "
           "scored hand-frames |", "|---|---|---|---|---|---|---|---|"] + meta_lines
    open(a.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
