"""RTMPose 2D hand keypoints in the input camera + one extra camera, triangulated, for F2 on AssemblyHands and HOT3D.

Same detector, crops and triangulation as handbench's Track 3 references (build_refs: RTMPose-m hand, two-pass
perspective crops, triangulate_kp with score >= 0.5 and reprojection < 3 px in every used view). Crops are placed from
the method's OWN hands (no ground truth input), as hot3d_floor.detect_clip(source="hawor") did for HaWoR: the hand
silhouette in the input image places the input crop; its pixel rays over build_refs.DEPTH_BAND place the crop in the
second camera. EgoFound3R's hands are first re-placed along their own pixel rays so they project with the TRUE K where
they project with EgoFound3R's own K (its 2D placement is right even when its focal is not, e.g. AssemblyHands).

Second camera:
  assemblyhands  the second headset camera (prepare_assemblyhands.py: rigid, 11.3 cm), rectified pinhole, native
                 (sideways) orientation; input camera = the upright-rotated input view
  hot3d          the right mono SLAM camera as a virtual pinhole (hot3d_check.rig "ego_cam_b", 12.6 cm baseline);
                 the left one is 1.5 cm from the RGB camera, too short to triangulate
Only frames where the method predicted a hand are detected (F2 shifts only those). AssemblyHands' second camera misses
a few released frames; those hand-frames get input-camera keypoints only (not triangulated).

  ../.venv-bench/bin/python tools/detect_two_view_keypoints.py --dataset assemblyhands [--jobs 8]
Writes data/two_view_kp/<dataset>/<clip>.npz: kp (T,2,2,21,2) px [input, second], score (T,2,2,21),
  tri (T,2,21,3) input camera, m (NaN = not triangulated), tri_err_px (T,2,21), wall_time_s.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from handbench import build_refs as B  # noqa: E402
from handbench.calib import Cam  # noqa: E402
from handbench.crops import PerspectiveCrop  # noqa: E402
from handbench.mano import Mano  # noqa: E402

OUT = ROOT / "data" / "two_view_kp"


def image_consistent(V, K_pred, K):
    """Move points along their pixel rays (depth kept) so projecting with K gives the pixels K_pred gave."""
    z = V[..., 2:3]
    uv = V[..., :2] / z * np.diag(K_pred)[:2] + K_pred[:2, 2]
    return np.concatenate([(uv - K[:2, 2]) / np.diag(K)[:2] * z, z], -1)


def silhouette(cam, V, faces):
    from handbench.experiments.hot3d_floor import silhouette as sil
    return sil(cam, V, faces)


def detect(rtm, cams, frame_a, frame_b, V, faces, grey_a):
    """One hand on one frame -> kp (2,21,2), score (2,21) for [input, second]; None if the hand is too small."""
    A = cams[0]
    ma = silhouette(A, V, faces)
    if ma.sum() < B.MIN_MASK_PX:
        return None
    ys, xs = np.nonzero(ma)
    pts = np.c_[xs, ys][:: max(1, len(xs) // 300)].astype(float)
    rays_a = np.c_[(pts - A.K[:2, 2]) / np.diag(A.K)[:2], np.ones(len(pts))]
    kp, sc = np.full((2, 21, 2), np.nan), np.zeros((2, 21))
    for v, (c, fr, grey) in enumerate(((A, frame_a, grey_a), (cams[1], frame_b, True))):
        if fr is None:  # second camera has no image on this frame
            continue
        rays = rays_a if v == 0 else c.to_cam(np.concatenate([rays_a * z for z in B.DEPTH_BAND]))
        if (rays[:, 2] <= 0.05).any():
            continue
        crop = PerspectiveCrop.around(c, rays, 256, 1.3)
        got = B._detect(rtm, crop, fr, grey)
        if got is None:
            continue
        k, s_, _ = got
        good = s_ >= 0.3
        if good.sum() >= 5:
            n2 = crop.crop_px_to_norm(k[good])
            crop2 = PerspectiveCrop.around(c, np.c_[n2, np.ones(len(n2))], 256, 1.6)
            got2 = B._detect(rtm, crop2, fr, grey)
            if got2 is not None and np.median(got2[1]) >= np.median(s_) - 0.05:
                crop, (k, s_, _) = crop2, got2
        kp[v], sc[v] = crop.crop_px_to_img(k), s_
    return kp, sc


def clip_sources(dataset, clip):
    """-> (cams [input, second], n, frame_a(i), frame_b(i), grey_a)."""
    meta = json.loads((ROOT / "data" / f"{dataset}_input" / f"{clip}.json").read_text())
    K = np.array(meta["K"])
    W, H = meta["size"]
    cap = cv2.VideoCapture(str(ROOT / "data" / f"{dataset}_input" / f"{clip}.mp4"))
    frames_a = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames_a.append(im)
    A = Cam("input", K, np.zeros(5), (W, H))
    if dataset == "assemblyhands":
        g = np.load(ROOT / "data" / "assemblyhands_input" / f"{clip}_gt.npz")
        T21 = np.median(g["T_cam2_from_cam"], 0)
        src = Path(meta["source"]).parent / f"{meta['second_camera']}_mono10bit"
        Bc = Cam("second", g["K2"], np.zeros(5), (H, W), T21)  # sideways native: W, H swapped back
        frame_b = lambda i: (cv2.imread(str(f)) if (f := src / f"{meta['frames'][i]:06d}.jpg").exists() else None)  # noqa: E731
        return [A, Bc], len(frames_a), frames_a.__getitem__, frame_b, True
    from handbench.datasets import hot3d as H3
    from handbench.experiments import hot3d_check as C
    hc = H3.Clip(H3.ROOT / "train_aria" / f"{clip}.tar")
    cams, maps, (_, right), _ = C.rig(hc)
    Bc = cams["ego_cam_b"]
    frame_b = lambda i: cv2.remap(hc.image(i, right), *maps[right], cv2.INTER_LINEAR)  # noqa: E731
    return [A, Bc], len(frames_a), frames_a.__getitem__, frame_b, False


def run_clip(args):
    dataset, clip, pred_dir = args
    os.environ["OMP_NUM_THREADS"] = "1"
    t0 = time.time()
    rtm, _, _ = B._models()
    mano = Mano()
    cams, n, frame_a, frame_b, grey_a = clip_sources(dataset, clip)
    e = np.load(Path(pred_dir) / clip / "export.npz")
    T = min(n, e["verts778"].shape[0])
    kp = np.full((T, 2, 2, 21, 2), np.nan, np.float32)
    sc = np.zeros((T, 2, 2, 21), np.float32)
    for i in range(T):
        fb = None
        for s in range(2):
            if not e["predicted"][i, s]:
                continue
            V = image_consistent(e["verts778"][i, s].astype(np.float64), e["K_pred"][i], cams[0].K)
            if fb is None:
                fb = frame_b(i)
            got = detect(rtm, cams, frame_a(i), fb, V, mano.faces_for(s == 1), grey_a)
            if got is not None:
                kp[i, s], sc[i, s] = got
    # handbench triangulation (3-view API): third view = second camera with zero scores, i.e. a+b only
    kp3 = np.concatenate([kp, kp[:, :, 1:]], 2)
    sc3 = np.concatenate([sc, np.zeros_like(sc[:, :, 1:])], 2)
    tri = B.triangulate_kp([cams[0], cams[1], cams[1]], kp3, sc3)
    out = OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{clip}.npz", kp=kp, score=sc, tri=tri["tri_ab"], tri_err_px=tri["tri_ab_err_px"],
                        tri_views=tri["tri_ab_views"], wall_time_s=time.time() - t0)
    n_hand = int(e["predicted"][:T].sum())
    n_tri = int((np.isfinite(tri["tri_ab"][..., 0]).sum(-1) >= 4).sum())
    return f"{clip}: {n_hand} predicted hand-frames, {n_tri} with >= 4 triangulated joints, {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["assemblyhands", "hot3d"])
    ap.add_argument("--pred-dir", default=None, help="EgoFound3R exports (default data/egofound3r/<dataset>)")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    pred = a.pred_dir or str(ROOT / "data" / "egofound3r" / a.dataset)
    clips = sorted(p.name for p in Path(pred).iterdir() if (p / "export.npz").exists())
    with mp.get_context("spawn").Pool(min(a.jobs, len(clips))) as pool:
        for line in pool.imap_unordered(run_clip, [(a.dataset, c, pred) for c in clips]):
            print(line, flush=True)


if __name__ == "__main__":
    main()
