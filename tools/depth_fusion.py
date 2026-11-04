"""Object mesh by multi-view depth fusion: TSDF from stereo depth inside the object's mask + silhouette carving.

  python depth_fusion.py EP_DIR MASK_ID OUT_MESH (--vo [--still 0.015] | --poses FP_POSE_DIR) [--depth stereo_fs]
                         [--voxel 0.003] [--trunc 0.01] [--every 2] [--margin 0.03]

Grid frame: world (--vo, camera_vo.npz camera_to_world; only for an object that does not move) or the object frame of
a FoundationPose track (--poses; any mesh, e.g. our own reconstruction, so no scan is used). Per frame, each voxel
projects to a pixel:
  - inside the (unoccluded) object mask with valid depth: TSDF update, sdf = depth - z clamped to [-trunc, trunc]
    (skipped behind the surface, sdf < -trunc);
  - outside the dilated object mask and not under a hand / another object: carved (empty, +1);
  - elsewhere (occluded, no depth, off-image): no update.
Voxels never updated stay inside (-1), so unseen parts are closed by the visual hull; everything below the support
plane (RANSAC on the depth in a ring around the mask; --no-table to skip) is empty. Marching cubes at 0, largest
connected component, OBJ out (grid frame, metres).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from skimage import measure


def read_depth(d, i):
    raw = cv2.imread(str(Path(d) / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    raw = raw.astype(np.float32)
    z = 65535.0 / np.maximum(raw, 1.0) - 1.0
    z[(raw >= 65535) | (raw <= 0)] = 0
    return z


def read_mask(p, shape):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape, bool)
    return cv2.resize(m, shape[::-1], interpolation=cv2.INTER_NEAREST) > 127


def load_pose(p):
    j = json.load(open(p))
    w, x, y, z = j["rotation"]
    T = np.eye(4)
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = j["translation"]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ep_dir"); ap.add_argument("mask_id"); ap.add_argument("out_mesh")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vo", action="store_true")
    g.add_argument("--poses")
    ap.add_argument("--depth", default="stereo_fs")
    ap.add_argument("--voxel", type=float, default=0.003)
    ap.add_argument("--trunc", type=float, default=0.01)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--carve-dilate", type=int, default=7, help="px the mask is dilated before carving")
    ap.add_argument("--no-table", dest="table", action="store_false",
                    help="do not carve below the support plane (fitted to the depth around the object mask)")
    ap.add_argument("--table-offset", type=float, default=0.002, help="m above the plane where carving stops")
    ap.add_argument("--still", type=float, default=None,
                    help="with --vo: fuse only the longest run of frames whose object depth centroid stays within this "
                         "many m of the run's median (the object sits still); needs >= 20 frames")
    a = ap.parse_args()
    E = Path(a.ep_dir)
    D = E / a.depth / "stereo_depth"
    Kj = json.load(open(E / "stereo" / "intrinsics_undist.json"))
    n = len(list((E / "masks" / a.mask_id).glob("*.png")))
    if a.vo:
        c2w = np.load(E / "camera_vo.npz")["camera_to_world"]
        cam_T_grid = {i: np.linalg.inv(c2w[i]) for i in range(n)}
    else:
        cam_T_grid = {int(p.stem): load_pose(p) for p in sorted(Path(a.poses).glob("[0-9]*.json"))}
    frames = [i for i in range(0, n, a.every) if i in cam_T_grid]
    dc_dirs = sorted((E / "hand_masks").iterdir()) + [p for p in sorted((E / "masks").iterdir()) if p.name != a.mask_id]
    z0 = read_depth(D, 0)
    H, W = z0.shape
    sx, sy = W / Kj["width"], H / Kj["height"]
    fx, fy, cx, cy = Kj["fx"] * sx, Kj["fy"] * sy, Kj["cx"] * sx, Kj["cy"] * sy

    def load(i):
        z = read_depth(D, i)
        m = read_mask(E / "masks" / a.mask_id / f"{i:06d}.png", z.shape)
        dc = np.zeros(z.shape, bool)
        for d in dc_dirs:
            dc |= read_mask(d / f"{i:06d}.png", z.shape)
        dc = cv2.dilate(dc.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        return z, m, dc

    def obj_points(i, z, m, dc):
        mm = (cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~dc & (z > 0)
        mm &= np.abs(z - cv2.medianBlur(z, 5)) < 0.015
        v, u = np.nonzero(mm)
        zz = z[v, u]
        P = np.stack([(u - cx) * zz / fx, (v - cy) * zz / fy, zz, np.ones_like(zz)], 1)
        return (P @ np.linalg.inv(cam_T_grid[i]).T)[:, :3]

    if a.still is not None:  # longest run of frames whose object depth centroid (grid frame) stays put
        cen = {}
        for i in sorted(cam_T_grid):
            z, m, dc = load(i)
            p = obj_points(i, z, m, dc)
            if len(p) >= 300:
                cen[i] = np.median(p, 0)
        ids = sorted(cen)
        best = (0, 0)
        for s in range(len(ids)):
            for e in range(s + best[1] - best[0], len(ids)):
                c = np.array([cen[j] for j in ids[s:e + 1]])
                if np.linalg.norm(c - np.median(c, 0), axis=1).max() > a.still:
                    break
                if ids[e] - ids[s] > 1.5 * (e - s) + 5:  # too many frames without a centroid
                    break
                if e - s > best[1] - best[0]:
                    best = (s, e)
        run = ids[best[0]:best[1] + 1]
        if len(run) < 20:
            raise SystemExit(f"no still run of >= 20 frames within {a.still * 100:.1f} cm (longest {len(run)})")
        frames = [i for i in run[:: a.every]]
        print(f"still run: frames {run[0]}-{run[-1]} ({len(run)} with a centroid, of {n}); using {len(frames)}")

    # grid bounds from the object's own depth points in the grid frame
    pts = []
    for i in frames[:: max(1, len(frames) // 30)]:
        z, m, dc = load(i)
        p = obj_points(i, z, m, dc)
        pts.append(p[:: max(1, len(p) // 3000)])
    pts = np.vstack(pts)
    plane = None
    if a.table:  # support plane from the depth in a ring around the object mask
        ring = []
        for i in frames[:: max(1, len(frames) // 30)]:
            z, m, dc = load(i)
            mm = m.astype(np.uint8)
            rr = (cv2.dilate(mm, np.ones((61, 61), np.uint8)) > 0) & ~(cv2.dilate(mm, np.ones((15, 15), np.uint8)) > 0)
            rr &= ~dc & (z > 0)
            v, u = np.nonzero(rr)
            zz = z[v, u]
            P = np.stack([(u - cx) * zz / fx, (v - cy) * zz / fy, zz, np.ones_like(zz)], 1)
            ring.append((P @ np.linalg.inv(cam_T_grid[i]).T)[:, :3][:: max(1, len(P) // 2000)])
        ring = np.vstack(ring)
        rng = np.random.default_rng(0)
        best = (0, None)
        for _ in range(500):
            s = ring[rng.choice(len(ring), 3, replace=False)]
            nv = np.cross(s[1] - s[0], s[2] - s[0])
            if np.linalg.norm(nv) < 1e-9:
                continue
            nv /= np.linalg.norm(nv)
            inl = np.abs((ring - s[0]) @ nv) < 0.004
            if inl.sum() > best[0]:
                best = (inl.sum(), inl)
        q = ring[best[1]]
        c0 = q.mean(0)
        nv = np.linalg.svd(q - c0, full_matrices=False)[2][2]
        if np.median((pts - c0) @ nv) < 0:
            nv = -nv  # normal toward the object
        plane = (nv, c0)
        print(f"table plane: {best[0]}/{len(ring)} ring points within 4 mm; object points above it "
              f"{np.mean((pts - c0) @ nv > 0):.1%}")
    lo = np.percentile(pts, 1, 0) - a.margin
    hi = np.percentile(pts, 99, 0) + a.margin
    dims = np.ceil((hi - lo) / a.voxel).astype(int)
    print(f"grid {dims.tolist()} ({np.prod(dims)/1e6:.1f} M voxels), extent cm {np.round((hi - lo) * 100, 1).tolist()}, "
          f"{len(frames)} frames")
    ax = [lo[k] + a.voxel * (np.arange(dims[k]) + 0.5) for k in range(3)]
    X = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    Xh = np.column_stack([X, np.ones(len(X))])
    tsdf = np.full(len(X), -1.0, np.float32)
    wsum = np.zeros(len(X), np.float32)
    carved = np.zeros(len(X), np.int32)
    seen = np.zeros(len(X), np.int32)
    k_carve = np.ones((a.carve_dilate, a.carve_dilate), np.uint8)
    for i in frames:
        z, m, dc = load(i)
        mc = cv2.dilate(m.astype(np.uint8), k_carve) > 0
        Pc = Xh @ cam_T_grid[i].T
        zc = Pc[:, 2]
        ok = zc > 0.05
        u = np.full(len(X), -1, np.int64); v = np.full(len(X), -1, np.int64)
        u[ok] = np.round(fx * Pc[ok, 0] / zc[ok] + cx).astype(np.int64)
        v[ok] = np.round(fy * Pc[ok, 1] / zc[ok] + cy).astype(np.int64)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.nonzero(ok)[0]
        uu, vv = u[idx], v[idx]
        d = z[vv, uu]
        inm = m[vv, uu] & ~dc[vv, uu] & (d > 0)
        sdf = d - zc[idx]
        upd = inm & (sdf > -a.trunc)
        j = idx[upd]
        s = np.clip(sdf[upd] / a.trunc, -1, 1)
        tsdf[j] = np.where(wsum[j] > 0, (tsdf[j] * wsum[j] + s) / (wsum[j] + 1), s)
        wsum[j] += 1
        seen[idx[inm]] += 1
        cv_ = ~mc[vv, uu] & ~dc[vv, uu]
        carved[idx[cv_]] += 1
    # a voxel is empty if carved in any view where it was never part of the depth surface band
    empty = (carved > 0) & (wsum == 0)
    vol = tsdf.copy()
    vol[empty] = 1.0
    if plane is not None:
        vol[(X - plane[1]) @ plane[0] < a.table_offset] = 1.0  # nothing below the support plane
    vol = vol.reshape(dims)
    print(f"voxels: depth-updated {np.mean(wsum > 0):.1%}, carved {np.mean(empty):.1%}, unseen-inside "
          f"{np.mean((wsum == 0) & ~empty):.1%}")
    verts, faces, _, _ = measure.marching_cubes(np.pad(vol, 1, constant_values=1.0), 0.0, spacing=(a.voxel,) * 3)
    verts += lo - a.voxel * 0.5
    mesh = trimesh.Trimesh(verts, faces, process=True)
    parts = mesh.split(only_watertight=False)
    mesh = max(parts, key=lambda p: p.area)
    mesh.export(a.out_mesh)
    print(f"{a.out_mesh}: {len(mesh.vertices)} verts {len(mesh.faces)} faces, extents cm "
          f"{np.round(np.sort(mesh.extents)[::-1] * 100, 1).tolist()}, {len(parts)} components (kept largest)")


if __name__ == "__main__":
    main()
