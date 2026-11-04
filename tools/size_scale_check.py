"""Where does the ego size bias come from? Public episodes only (mocap + scans).

  python size_scale_check.py EP NAME=MASK_ID [NAME=MASK_ID ...] [--depth stereo_fs] [--fp-suffix _fs] [--every 2]
                             [--json OUT]

1. Depth scale: render the scan at the FoundationPose pose (camera frame, z-buffered surface samples) and compare with the
   stereo depth on the object's eroded, hand-free mask: median stereo/render ratio (1.00 = depth agrees with a true-size
   object at the tracked pose) and median stereo-render offset.
2. Track scale, no odometry: for pairs of objects, distance between their centres in the camera frame (FoundationPose)
   vs mocap, per frame; median ratio.
3. Size estimator variants vs the scan's longest box edge (what contract_check compares the built mesh with): erosion,
   outlier filter (3 MAD as now / local depth-edge filter), measure (trimmed PCA extent as now / long side of the
   minimum-area rectangle), frame set (all / hands off the object), aggregate over frames (median as now / p80 / p90).
   Prints est/scan per variant.
"""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

E2 = Path.home() / "v2d/ego2sim"
DATA = Path.home() / "v2d/v2d_data/track_3/public"


def scan_mesh(name):
    for c in (DATA / "mesh" / name / f"{name}.glb", DATA / "mesh" / name / f"{name}_visual.glb"):
        if c.exists():
            return trimesh.load(str(c), force="mesh")
    raise FileNotFoundError(name)


def pca_longest(p, trim=0.5):
    p = p - p.mean(0)
    _, _, vt = np.linalg.svd(p[:: max(1, len(p) // 5000)], full_matrices=False)
    q = p @ vt.T
    return float((np.percentile(q, 100 - trim, 0) - np.percentile(q, trim, 0)).max())


def rect_longest(p, trim=0.5):
    """Long side of the minimum-area rectangle in the two main PCA axes (trimmed extents along its sides)."""
    p = p - p.mean(0)
    _, _, vt = np.linalg.svd(p[:: max(1, len(p) // 5000)], full_matrices=False)
    q = p @ vt.T
    a = np.radians(cv2.minAreaRect(q[:: max(1, len(q) // 20000), :2].astype(np.float32))[2])
    r = q[:, :2] @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    r = np.column_stack([r, q[:, 2]])
    return float((np.percentile(r, 100 - trim, 0) - np.percentile(r, trim, 0)).max())


def read_depth(d, i):
    raw = cv2.imread(str(d / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    raw = raw.astype(np.float32)
    z = 65535.0 / np.maximum(raw, 1.0) - 1.0
    z[(raw >= 65535) | (raw <= 0)] = 0
    return z


def read_mask(p, shape):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return cv2.resize(m, shape[::-1], interpolation=cv2.INTER_NEAREST) > 127


def fp_pose(d, i):
    p = Path(d) / f"{i:06d}.json"
    if not p.exists():
        return None
    j = json.load(open(p))
    w, x, y, z = j["rotation"]
    T = np.eye(4)
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = j["translation"]
    return T


def gt_centres(parquet, names, centroids):
    rows = pq.read_table(parquet, columns=["observation.objects"]).column(0).to_pylist()
    out = {n: np.full((len(rows), 3), np.nan) for n in names}
    for t, objs in enumerate(rows):
        for o in objs:
            if o["name"] in out:
                x, y, z, qw, qx, qy, qz = o["pose"]
                out[o["name"]][t] = R.from_quat([qx, qy, qz, qw]).apply(centroids[o["name"]]) + [x, y, z]
    return out


def render_depth(pts, T, K, shape):
    c = pts @ T[:3, :3].T + T[:3, 3]
    c = c[c[:, 2] > 0.05]
    u = np.round(K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2]).astype(int)
    k = (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
    z = np.full(shape, np.inf, np.float32)
    np.minimum.at(z, (v[k], u[k]), c[k, 2].astype(np.float32))
    z[~np.isfinite(z)] = 0
    return z


def frame_points(z, m, K, erode, filt):
    if erode:
        m = cv2.erode(m.astype(np.uint8), np.ones((erode, erode), np.uint8)) > 0
    m = m & (z > 0)
    if filt == "local":  # drop flying pixels at depth edges, keep the object's real depth range
        med = cv2.medianBlur(np.where(z > 0, z, 0).astype(np.float32), 5)
        m &= np.abs(z - med) < 0.015
    v, u = np.nonzero(m)
    if len(u) < 200:
        return None
    zz = z[v, u]
    if filt == "mad3":
        md = np.median(zz)
        mad = 1.4826 * np.median(np.abs(zz - md)) + 1e-4
        k = np.abs(zz - md) < 3 * mad
    else:
        k = np.abs(zz - np.median(zz)) < 0.5
    u, v, zz = u[k], v[k], zz[k]
    return np.stack([(u - K[0, 2]) * zz / K[0, 0], (v - K[1, 2]) * zz / K[1, 1], zz], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ep", type=int)
    ap.add_argument("objs", nargs="+")
    ap.add_argument("--depth", default="stereo_fs")
    ap.add_argument("--fp-suffix", default="_fs")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    E = E2 / f"ep{a.ep}"
    D = E / a.depth / "stereo_depth"
    Kj = json.load(open(E / "stereo" / "intrinsics_undist.json"))
    names = [s.split("=")[0] for s in a.objs]
    ids = {s.split("=")[0]: s.split("=")[1] for s in a.objs}
    z0 = read_depth(D, 0)
    sx, sy = z0.shape[1] / Kj["width"], z0.shape[0] / Kj["height"]
    K = np.array([[Kj["fx"] * sx, 0, Kj["cx"] * sx], [0, Kj["fy"] * sy, Kj["cy"] * sy], [0, 0, 1]])
    meshes = {n: scan_mesh(n) for n in names}
    samples = {n: np.asarray(trimesh.sample.sample_surface(meshes[n], 300000, seed=0)[0]) for n in names}
    cents = {n: samples[n].mean(0) for n in names}
    scan_long = {n: float(meshes[n].extents.max()) for n in names}  # what contract_check compares: longest box edge
    scan_pca = {n: pca_longest(samples[n], 0.5) for n in names}  # the estimator's own definition on the full surface
    hand_dirs = sorted((E / "hand_masks").iterdir())
    n_frames = len(list((E / "masks" / ids[names[0]]).glob("*.png")))
    frames = range(0, n_frames, a.every)
    variants = list(itertools.product([7, 3, 0], ["mad3", "local"], ["pca", "rect"]))
    rec = {n: {"ratio": [], "off": [], "ext": {v: [] for v in variants}, "handfree": []} for n in names}
    poses = {n: {} for n in names}
    for i in frames:
        z = read_depth(D, i)
        if z is None:
            continue
        hand = np.zeros(z.shape, bool)
        for hd in hand_dirs:
            h = read_mask(hd / f"{i:06d}.png", z.shape)
            if h is not None:
                hand |= h
        hand_d = cv2.dilate(hand.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        for n in names:
            m = read_mask(E / "masks" / ids[n] / f"{i:06d}.png", z.shape)
            if m is None or m.sum() < 500:
                continue
            T = fp_pose(E / f"fp_{n}{a.fp_suffix}", i)
            if T is not None:
                poses[n][i] = T
                zr = render_depth(samples[n], T, K, z.shape)
                me = (cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~hand_d
                both = me & (z > 0) & (zr > 0)
                if both.sum() > 300:
                    rec[n]["ratio"].append(float(np.median(z[both] / zr[both])))
                    rec[n]["off"].append(float(np.median(z[both] - zr[both])) * 100)
            mf = m & ~hand_d
            rec[n]["handfree"].append(bool(hand_d[m].mean() < 0.01) if m.any() else False)
            pts = {}
            for v in variants:
                if v[:2] not in pts:
                    pts[v[:2]] = frame_points(z, mf, K, *v[:2])
                p = pts[v[:2]]
                rec[n]["ext"][v].append(np.nan if p is None else (pca_longest(p) if v[2] == "pca" else rect_longest(p)))
    print(f"ep{a.ep} depth={a.depth} fp=fp_*{a.fp_suffix} frames every {a.every}; K scaled to depth {z0.shape[::-1]}")
    out = {"ep": a.ep, "depth": a.depth, "objects": {}}
    for n in names:
        r = rec[n]
        print(f"== {n} (mask {ids[n]}): scan longest box edge {scan_long[n]*100:.1f} cm, full-surface PCA extent "
              f"{scan_pca[n]*100:.1f} cm ({scan_pca[n] / scan_long[n]:.3f} of the box edge)")
        if r["ratio"]:
            q = np.percentile(r["ratio"], [25, 50, 75])
            print(f"  depth vs true-size scan at FP pose: stereo/render median {q[1]:.3f} (IQR {q[0]:.3f}-{q[2]:.3f}), "
                  f"offset {np.median(r['off']):+.2f} cm, {len(r['ratio'])} frames")
        hf = np.array(r["handfree"])
        o = {"scan_longest_m": scan_long[n], "scan_pca_m": scan_pca[n], "depth_ratio": float(np.median(r["ratio"])) if r["ratio"] else None,
             "variants": {}}
        for v in variants:
            L = np.array(r["ext"][v])
            for fs_name, sel in (("all", np.isfinite(L)), ("handfree", np.isfinite(L) & hf)):
                if sel.sum() < 5:
                    continue
                for agg, f in (("median", np.median), ("p80", lambda x: np.percentile(x, 80)),
                               ("p90", lambda x: np.percentile(x, 90))):
                    key = f"erode{v[0]}_{v[1]}{'_rect' if v[2] == 'rect' else ''}_{fs_name}_{agg}"
                    o["variants"][key] = float(f(L[sel])) / scan_long[n]
        cur = o["variants"].get("erode7_mad3_all_median")
        best = sorted(o["variants"].items(), key=lambda kv: abs(kv[1] - 1))[:3]
        print(f"  size est/scan: current (erode7 mad3 all median) {cur:.3f}; hand-free frames {hf.sum()}/{len(hf)}; "
              f"closest: " + ", ".join(f"{k} {x:.3f}" for k, x in best))
        out["objects"][n] = o
    pairs = [(p, q) for p, q in itertools.combinations(names, 2)]
    if pairs:
        pqf = next(DATA.glob(f"data/*/episode_{a.ep:06d}.parquet"))
        gt = gt_centres(pqf, names, cents)
        for p, q in pairs:
            ts = sorted(set(poses[p]) & set(poses[q]))
            de = np.array([np.linalg.norm((poses[p][t][:3, :3] @ cents[p] + poses[p][t][:3, 3]) -
                                          (poses[q][t][:3, :3] @ cents[q] + poses[q][t][:3, 3])) for t in ts])
            dg = np.array([np.linalg.norm(gt[p][t] - gt[q][t]) for t in ts])
            k = dg > 0.08  # skip nested / touching objects
            if k.sum() > 5:
                rr = de[k] / dg[k]
                print(f"== pair {p}-{q}: camera-frame centre distance FP/mocap median {np.median(rr):.3f} "
                      f"(IQR {np.percentile(rr, 25):.3f}-{np.percentile(rr, 75):.3f}), mocap distance median "
                      f"{np.median(dg[k])*100:.1f} cm, {k.sum()} frames")
                out.setdefault("pairs", {})[f"{p}-{q}"] = float(np.median(rr))
            else:
                print(f"== pair {p}-{q}: objects within 8 cm in all frames, skipped")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
