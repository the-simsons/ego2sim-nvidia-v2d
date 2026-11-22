"""Camera path from one camera's video plus per-frame depth: RGB-D odometry on the static background.

Features on the background (object and hand masks excluded, dilated) are tracked with pyramidal Lucas-Kanade from a
keyframe; their 3-D positions come from the keyframe's depth; each frame's pose is solved with PnP + RANSAC against
the keyframe and refined on the inliers. A new keyframe starts when too few tracks survive, too few are inliers, or
every --kf-every frames. Video-only (no motion capture): world = camera frame 0, metric scale from the depth.

    python tools/rgbd_vo.py --frames DIR --depth DIR --intrinsics JSON --exclude MASKDIR[,MASKDIR...] --out camera_vo_video.npz
Mask dirs hold NNNNNN.png (non-zero = excluded); a dir of per-track subdirs (SAM 2 layout) is merged.
Depth PNGs use the v2d encoding (pixel = 65535 / (depth_m + 1)). Writes camera_to_world (N,4,4) and per-frame stats."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def decode_depth(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = 65535.0 / np.maximum(raw, 1.0) - 1.0
    d[(raw >= 65535) | (raw <= 0)] = 0
    return d


def mask_dirs(spec):
    out = []
    for d in [Path(x) for x in spec.split(",") if x]:
        subs = [s for s in sorted(d.iterdir()) if s.is_dir()] if d.is_dir() else []
        out += subs if subs else [d]
    return out


def exclusion(dirs, stem, shape, dilate):
    m = np.zeros(shape, np.uint8)
    for d in dirs:
        p = d / f"{stem}.png"
        if p.is_file():
            x = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if x is not None:
                m |= (cv2.resize(x, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    return cv2.dilate(m, np.ones((dilate, dilate), np.uint8)) > 0 if dilate > 1 else m > 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--intrinsics", required=True, help="JSON with fx, fy, cx, cy (full-resolution pixels)")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.5, help="image scale for tracking")
    ap.add_argument("--max-depth", type=float, default=6.0)
    ap.add_argument("--features", type=int, default=600)
    ap.add_argument("--kf-every", type=int, default=12)
    ap.add_argument("--min-tracks", type=int, default=120)
    ap.add_argument("--min-inliers", type=int, default=60)
    ap.add_argument("--dilate", type=int, default=31)
    a = ap.parse_args(argv)

    K0 = json.load(open(a.intrinsics))
    frames = sorted(Path(a.frames).glob("*.jpg"))
    excl = mask_dirs(a.exclude)
    s = a.scale
    K = np.array([[K0["fx"] * s, 0, K0["cx"] * s], [0, K0["fy"] * s, K0["cy"] * s], [0, 0, 1]], float)
    n = len(frames)
    c2w = np.tile(np.eye(4), (n, 1, 1))
    stats = []

    def gray(i):
        im = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def keyframe(i, img):
        d = cv2.resize(decode_depth(Path(a.depth) / f"{frames[i].stem}.png"), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        ok = (d > 0.1) & (d < a.max_depth) & ~exclusion(excl, frames[i].stem, img.shape, a.dilate)
        p = cv2.goodFeaturesToTrack(img, a.features, 0.01, 10, mask=ok.astype(np.uint8) * 255)
        if p is None:
            return None, None
        p = p.reshape(-1, 2)
        z = d[p[:, 1].astype(int), p[:, 0].astype(int)]
        X = np.c_[(p[:, 0] - K[0, 2]) * z / K[0, 0], (p[:, 1] - K[1, 2]) * z / K[1, 1], z]
        return p.astype(np.float32), X

    prev = gray(0)
    kf_i, kf_pts, kf_X = 0, *keyframe(0, prev)
    cur_pts = kf_pts.copy()
    for i in range(1, n):
        img = gray(i)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, img, cur_pts, None, winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(img, prev, nxt, None, winSize=(21, 21), maxLevel=3)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - cur_pts, axis=1) < 1.0)
        # tracks that wandered onto the object or a hand in this frame are dropped
        ex = exclusion(excl, frames[i].stem, img.shape, a.dilate)
        inb = (nxt[:, 0] >= 0) & (nxt[:, 0] < img.shape[1] - 1) & (nxt[:, 1] >= 0) & (nxt[:, 1] < img.shape[0] - 1)
        good &= inb
        good[good] &= ~ex[nxt[good, 1].astype(int), nxt[good, 0].astype(int)]
        kf_pts, kf_X, cur_pts = kf_pts[good], kf_X[good], nxt[good]
        n_in, rmse = 0, np.nan
        T_kf_from_cur = None
        if len(cur_pts) >= 8:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(kf_X, cur_pts, K, None, iterationsCount=200, reprojectionError=2.0,
                                                     confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok and inl is not None and len(inl) >= 8:
                inl = inl.ravel()
                rvec, tvec = cv2.solvePnPRefineLM(kf_X[inl], cur_pts[inl], K, None, rvec, tvec)
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4); T[:3, :3], T[:3, 3] = R, tvec.ravel()   # cur <- kf
                T_kf_from_cur = np.linalg.inv(T)
                proj, _ = cv2.projectPoints(kf_X[inl], rvec, tvec, K, None)
                rmse = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - cur_pts[inl]) ** 2, 1))))
                n_in = len(inl)
        c2w[i] = c2w[kf_i] @ T_kf_from_cur if T_kf_from_cur is not None else c2w[i - 1]
        stats.append({"frame": i, "tracks": int(len(cur_pts)), "inliers": int(n_in), "rmse_px": rmse,
                      "keyframe": int(kf_i), "lost": T_kf_from_cur is None})
        if len(cur_pts) < a.min_tracks or n_in < a.min_inliers or i - kf_i >= a.kf_every:
            kf_i = i
            kp, kx = keyframe(i, img)
            if kp is not None:
                kf_pts, kf_X, cur_pts = kp, kx, kp.copy()
        prev = img
    np.savez(a.out, camera_to_world=c2w, method="rgbd_vo", depth=str(a.depth))
    json.dump({"frames": n, "lost_frames": int(sum(x["lost"] for x in stats)),
               "median_inliers": float(np.median([x["inliers"] for x in stats])) if stats else 0,
               "median_rmse_px": float(np.nanmedian([x["rmse_px"] for x in stats])) if stats else np.nan,
               "per_frame": stats}, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    lost = sum(x["lost"] for x in stats)
    print(f"{n} frames, lost {lost}, median inliers {np.median([x['inliers'] for x in stats]):.0f}, "
          f"median reprojection {np.nanmedian([x['rmse_px'] for x in stats]):.2f} px, path length "
          f"{np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1).sum():.2f} m -> {a.out}")


if __name__ == "__main__":
    main()
