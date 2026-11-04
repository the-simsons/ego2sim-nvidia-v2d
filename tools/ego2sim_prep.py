"""Ego2Sim harness inputs (riddle side), from stereo depth that is pixel-aligned with the undistorted cam_a frames.

  occluders  Rebuild each target's occluder mask with depth ordering. Hands always occlude; another object's
             pixels occlude only where they are NEARER than the target (stereo depth < target's near surface
             + margin) or have no valid depth. Fixes AgentSTAR's wish.md item 1 (a rack/tray behind the cup was
             don't-care, so the render could spill into it for free).
  size       Metric size target for the object from stereo: per frame, back-project the eroded, unoccluded object
             mask, drop depth outliers, and take the longest PCA extent (--trim percentile at each end); the target is
             the p90 over the frames where no hand touches the object mask (all frames if fewer than 5). Each view only
             sees part of the object, so the per-frame extent is a lower bound: the old median over all frames was
             12% low on average (worst -33%, nested cup) on 10 public objects with FoundationStereo depth; erode 3 +
             hand-free + p90 is 0.0% on average, 3.2% mean / 8.3% worst error (size_scale_check.py, NOTES Phase 7c).
             Fixes wish.md item 2 (measure_depth's box was dominated by hand/background points).

Usage:
  python ego2sim_prep.py occluders INPUT_OBJ_DIR OTHER_MASKS_DIR[,DIR...] HAND_MASKS_DIR[,DIR...] STEREO_DIR OUT_DIR
  python ego2sim_prep.py size INPUT_OBJ_DIR STEREO_DIR OUT_JSON [--frames 000000,000010,...]
  python ego2sim_prep.py stereo_depth CAPTURE_DIR STEREO_DIR OUT_DEPTH_DIR   (replace Pi3X pointmaps with stereo)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 0


def stereo_depth(stereo_dir, stem):
    """v2d inverse-depth uint16 png -> metres (0 = invalid)."""
    raw = cv2.imread(str(Path(stereo_dir) / "stereo_depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    raw = raw.astype(np.float32)
    d = 65535.0 / np.maximum(raw, 1.0) - 1.0
    d[(raw >= 65535) | (raw <= 0)] = 0
    return d


def cmd_occluders(a):
    obj = Path(a.input_obj_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    others = [Path(p) for p in a.other_masks.split(",") if p]
    hands = [Path(p) for p in a.hand_masks.split(",") if p]
    n_before = n_after = 0
    for f in sorted((obj / "mask_object").glob("*.png")):
        stem = f.stem
        tgt = read_mask(f)
        d = stereo_depth(a.stereo_dir, stem)
        occ = np.zeros(tgt.shape, bool)
        for h in hands:
            hm = read_mask(h / f"{stem}.png")
            if hm is not None:
                occ |= hm
        other = np.zeros(tgt.shape, bool)
        for o in others:
            om = read_mask(o / f"{stem}.png")
            if om is not None:
                other |= om
        other &= ~tgt
        if d is not None:
            td = d[tgt & (d > 0)]
            near = np.percentile(td, 10) if td.size > 50 else np.inf
            nearer = ((d > 0) & (d < near + a.margin)).astype(np.uint8)
            nearer = cv2.morphologyEx(nearer, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # drop depth speckle
            # pixels without stereo depth follow their neighbourhood: occluder only next to a nearer region
            near_zone = cv2.dilate(nearer, np.ones((21, 21), np.uint8)) > 0
            occ |= other & ((nearer > 0) | ((d <= 0) & near_zone))
        else:
            occ |= other
        occ &= ~tgt
        n_before += int((other | occ).sum())
        n_after += int(occ.sum())
        cv2.imwrite(str(out / f"{stem}.png"), occ.astype(np.uint8) * 255)
    print(f"occluders -> {out}: kept {n_after / max(n_before, 1):.0%} of the old hand+other-object don't-care area")


def cmd_size(a):
    obj = Path(a.input_obj_dir)
    K = json.load(open(Path(a.stereo_dir) / "intrinsics_undist.json"))
    stems = a.frames.split(",") if a.frames else [p.stem for p in sorted((obj / "mask_object").glob("*.png"))]
    kernel = np.ones((7, 7), np.uint8)
    edge = np.ones((a.erode, a.erode), np.uint8) if a.erode else None
    rows = []
    for stem in stems:
        m = read_mask(obj / "mask_object" / f"{stem}.png")
        occ = read_mask(obj / "mask_hand" / f"{stem}.png")
        d = stereo_depth(a.stereo_dir, stem)
        if m is None or d is None:
            continue
        hand = cv2.dilate(occ.astype(np.uint8), kernel) > 0 if occ is not None else np.zeros_like(m)
        handfree = bool(hand[m].mean() < 0.01) if m.any() else False
        if edge is not None:
            m = cv2.erode(m.astype(np.uint8), edge) > 0  # drop depth-edge flying pixels
        m &= ~hand
        v, u = np.nonzero(m & (d > 0))
        if len(u) < 200:
            continue
        z = d[v, u]
        med = np.median(z)
        mad = 1.4826 * np.median(np.abs(z - med)) + 1e-4
        k = np.abs(z - med) < 3 * mad
        u, v, z = u[k], v[k], z[k]
        p = np.stack([(u - K["cx"]) * z / K["fx"], (v - K["cy"]) * z / K["fy"], z], 1)
        p -= p.mean(0)
        _, _, vt = np.linalg.svd(p[:: max(1, len(p) // 5000)], full_matrices=False)
        q = p @ vt.T
        ext = np.percentile(q, 100 - a.trim, 0) - np.percentile(q, a.trim, 0)
        area = float(np.sum((z / K["fx"]) * (z / K["fy"])))  # metric visible area (m^2)
        rows.append({"frame": stem, "n": int(len(u)), "area_m2": area, "longest_m": float(ext.max()), "handfree": handfree,
                     "extents_m": [float(x) for x in sorted(ext, reverse=True)]})
    if a.select == "full":  # frames showing (nearly) the whole object: partial views shrink the extent
        A = np.array([r["area_m2"] for r in rows])
        keep = A >= 0.85 * np.percentile(A, 90)
        rows = [r for r, k in zip(rows, keep) if k]
    select = a.select
    if select == "handfree":
        free = [r for r in rows if r["handfree"]]
        rows, select = (free, "handfree") if len(free) >= 5 else (rows, f"all ({len(free)} hand-free < 5)")
    L = np.array([r["longest_m"] for r in rows])
    target = float(np.median(L)) if a.agg == "median" else float(np.percentile(L, int(a.agg[1:])))
    res = {"method": f"stereo PCA extent of mask eroded {a.erode} px, unoccluded, {a.trim}-{100 - a.trim} pct, per frame, "
                     f"frames: {select}, target {a.agg}",
           "frames_used": len(rows),
           "longest_m_median": float(np.median(L)), "longest_m_p80": float(np.percentile(L, 80)),
           "longest_m_p90": float(np.percentile(L, 90)), "longest_m_max": float(L.max()),
           "target_longest_m": target, "tolerance": a.tolerance,
           "per_frame": rows}
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(f"size: {len(rows)} frames ({select}); longest extent median {res['longest_m_median']*100:.1f} cm, "
          f"p90 {res['longest_m_p90']*100:.1f} cm, max {res['longest_m_max']*100:.1f} cm -> target ({a.agg}) "
          f"{target*100:.1f} cm +/- {a.tolerance:.0%}  ({a.out_json})")


def cmd_stereo_depth(a):
    """Replace an AgentSTAR capture's Pi3X pointmaps with stereo depth on the same grid. Pi3X matches stereo over
    the whole frame (ratio ~1.00 after `agentstar_prep.py metric`) but puts a hand-held object 25-41% too close
    (ep12 cup, frames 60-180), so no global rescale can fix it. Points are back-projected with the capture's own
    intrinsics (the ones the harness renders with); conf = 1 where stereo is valid, else 0 (keep = False)."""
    cap = Path(a.capture_dir)
    cams = np.load(cap / "tracking" / "cameras.npz")
    W, H = (int(v) for v in cams["size"])
    kfs = json.load(open(cap / "tracking" / "keyframes.json"))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    for i, kf in enumerate(kfs):
        stem = Path(kf["frame_name"]).stem
        K = cams["intrinsics"][i]
        d = stereo_depth(a.stereo_dir, stem)
        z = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        keep = (z > 0.1) & (z < a.max_depth)
        z = np.where(keep, z, 0).astype(np.float32)
        local = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1).astype(np.float32)
        np.savez_compressed(out / f"{stem}.npz", local=local, conf=keep.astype(np.float32), keep=keep)
        print(f"{stem}: stereo valid {keep.mean():.0%}")
    print(f"stereo pointmaps -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("occluders")
    o.add_argument("input_obj_dir")
    o.add_argument("other_masks")
    o.add_argument("hand_masks")
    o.add_argument("stereo_dir")
    o.add_argument("out_dir")
    o.add_argument("--margin", type=float, default=0.01)
    s = sub.add_parser("size")
    s.add_argument("input_obj_dir")
    s.add_argument("stereo_dir")
    s.add_argument("out_json")
    s.add_argument("--frames", default="")
    s.add_argument("--tolerance", type=float, default=0.08)
    s.add_argument("--select", choices=["handfree", "all", "full"], default="handfree",
                   help="handfree: frames where hands cover < 1%% of the object mask (all frames if fewer than 5); "
                        "full: only frames whose visible metric area >= 85%% of its p90 (drops partial / stacked views)")
    s.add_argument("--agg", choices=["p90", "p80", "median"], default="p90",
                   help="target over the selected frames (old default: median over all frames, 12%% low on average)")
    s.add_argument("--erode", type=int, default=3, help="mask erosion in px before back-projection (old default 7)")
    s.add_argument("--trim", type=float, default=0.5, help="percentile trimmed at each end of the PCA extent (0.5 + median target: worst 4.1%% error on cup/dust pan/lid scans)")
    sd = sub.add_parser("stereo_depth")
    sd.add_argument("capture_dir")
    sd.add_argument("stereo_dir")
    sd.add_argument("out_dir")
    sd.add_argument("--max-depth", type=float, default=3.0)
    a = ap.parse_args()
    {"occluders": cmd_occluders, "size": cmd_size, "stereo_depth": cmd_stereo_depth}[a.cmd](a)


if __name__ == "__main__":
    main()
