"""cam_b hand masks (set A reference A.1): SAM2.1 (Hiera-L) video segmentation of the cam_b video, prompted from cam_a.

Prompt transfer (+ cam_b's own detector keypoints, see transfer_prompts): at keyframes (every --every frames where the cam_a SAM2 hand mask is visible, plus the first visible
frame of every run) the cam_a mask pixels are lifted to 3D with SGBM stereo depth (refs/<ep>/stereo.npz) and
projected into the published cam_b image -> a box + 5 positive points per hand, with the other hand's transferred
points as negatives. SAM2 then propagates both hands through the whole cam_b video. Masks are written in published
(distorted) cam_b pixels: refs/<ep>/masks_b/<L|R>/<frame>.png (only frames with a non-empty mask).
Like the cam_a masks, these include the forearm; the metric clips both to the hand region.

Runs in methods/.venv-sam2. Uses every visible GPU: one worker per GPU, episodes from a shared queue.
Usage: python -m handbench.build_masks_b [--episodes ...] [--every 15]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from handbench import paths

SAM2_CKPT = "/usr1/e2s/video_to_data/reconstruction/data/weights/sam2/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SIDES = ("L", "R")
OBJ = {"L": 1, "R": 2}


def transfer_prompts(ep, every: int):
    """{frame: {side: (box[4], pos_pts (k,2), neg_pts (m,2))}} in cam_b published pixels."""
    from handbench.adapters.v2d_upstream import UpstreamCamera, anycalib_dir
    from handbench.build_refs import side_masks
    from handbench.calib import load_rig
    from handbench.datasets.v2d import StereoCrops
    from handbench.refs_io import hand_track_ids

    cams = load_rig(paths.rig_json())
    A, B = cams["ego_cam_a"], cams["ego_cam_b"]
    U = UpstreamCamera(paths.CACHE / ep, A, anycalib_dir(ep))
    st = StereoCrops(paths.REFS / ep / "stereo.npz")
    tracks = hand_track_ids(ep)
    kp = np.load(paths.REFS / ep / "kp2d.npz")
    vis = kp["mask_area_a"] >= 400
    n = vis.shape[0]
    keys = set()
    for s in range(2):
        for i in range(n):
            if vis[i, s] and (i % every == 0 or i == 0 or not vis[i - 1, s]):
                keys.add(i)
    rng = np.random.default_rng(0)
    out = {}
    for i in sorted(keys):
        ms = side_masks(ep, U, tracks, i)
        pts_b = {}
        for side, mb in ms.items():
            if mb.sum() < 400:
                continue
            got = st.get(i, side)
            ys, xs = np.nonzero(mb)
            sel = rng.choice(len(xs), min(len(xs), 400), replace=False)
            xs, ys = xs[sel], ys[sel]
            z = np.full(len(xs), np.nan)
            if got is not None:
                D, _, box = got
                sc = st.scale
                u = np.clip(np.round((xs + 0.5) * sc - 0.5).astype(int) - box[0], 0, D.shape[1] - 1)
                v = np.clip(np.round((ys + 0.5) * sc - 0.5).astype(int) - box[1], 0, D.shape[0] - 1)
                z = D[v, u].astype(float)
                z[z <= 0] = np.nan
            if np.isfinite(z).sum() < 20:  # no stereo: median hand depth of the episode set (0.40 m)
                z = np.full(len(xs), 0.40)
            else:
                z = np.where(np.isfinite(z), z, np.nanmedian(z))
            rays = np.c_[(np.c_[xs, ys] - A.K[:2, 2]) / np.diag(A.K)[:2], np.ones(len(xs))]
            uv = B.project(rays * z[:, None])
            W, H = B.size
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if ok.sum() < 20:
                continue
            pts_b[side] = uv[ok]
        for side, p in pts_b.items():
            lo, hi = np.percentile(p, 2, 0), np.percentile(p, 98, 0)
            # positives: 5 points spread over the transferred mask (k-means-like: farthest point sampling)
            pos = [p[np.argmin(np.linalg.norm(p - np.median(p, 0), axis=1))]]
            for _ in range(4):
                d = np.min(np.linalg.norm(p[:, None] - np.array(pos)[None], axis=2), 1)
                pos.append(p[np.argmax(d)])
            # cam_b sees more of the hand than cam_a near the cam_a image border (127 vs 108 deg FOV): add cam_b's own
            # confident detector keypoints (wrist, MCPs, fingertips) so the mask covers the whole hand as cam_b sees it
            s_i = SIDES.index(side)
            kb, sb = kp["kp"][i, s_i, 1], kp["score"][i, s_i, 1]
            good = (sb >= 0.5) & np.isfinite(kb[:, 0])
            if good.sum() >= 5:
                q = kb[good]
                pad = 0.15 * (q.max(0) - q.min(0))
                lo, hi = np.minimum(lo, q.min(0) - pad), np.maximum(hi, q.max(0) + pad)
                for j in (0, 5, 9, 13, 17, 4, 8, 12):
                    if good[j]:
                        pos.append(kb[j])
                lo, hi = np.maximum(lo, 0), np.minimum(hi, np.array(B.size) - 1)
            other = [q for s2, q in pts_b.items() if s2 != side]
            neg = np.array([np.median(other[0], 0)]) if other else np.zeros((0, 2))
            out.setdefault(i, {})[side] = (np.r_[lo, hi], np.array(pos), neg)
    return out, n


def run_episode(ep, every, tmp_root: Path):
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    t0 = time.time()
    prompts, n = transfer_prompts(ep, every)
    frames = tmp_root / ep
    frames.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(paths.video(ep, "ego_cam_b")))
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        cv2.imwrite(str(frames / f"{i:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        i += 1
    pred = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")
    out = paths.REFS / ep / "masks_b"
    if out.exists():
        shutil.rmtree(out)
    for s in SIDES:
        (out / s).mkdir(parents=True)
    counts = {s: 0 for s in SIDES}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = pred.init_state(video_path=str(frames), offload_video_to_cpu=True, offload_state_to_cpu=False)
        for fi, per in prompts.items():
            for side, (box, pos, neg) in per.items():
                pts = np.r_[pos, neg].astype(np.float32)
                lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
                pred.add_new_points_or_box(state, frame_idx=fi, obj_id=OBJ[side], points=pts, labels=lab,
                                           box=box.astype(np.float32))
        first = min(prompts) if prompts else 0
        for fi, obj_ids, logits in pred.propagate_in_video(state, start_frame_idx=first):
            for k, oid in enumerate(obj_ids):
                side = "L" if oid == OBJ["L"] else "R"
                m = (logits[k, 0] > 0).cpu().numpy()
                if m.sum() >= 50:
                    cv2.imwrite(str(out / side / f"{fi:06d}.png"), m.astype(np.uint8) * 255)
                    counts[side] += 1
        if first > 0:  # hands present before the first prompt: propagate backwards too
            for fi, obj_ids, logits in pred.propagate_in_video(state, start_frame_idx=first, reverse=True):
                for k, oid in enumerate(obj_ids):
                    side = "L" if oid == OBJ["L"] else "R"
                    p = out / side / f"{fi:06d}.png"
                    m = (logits[k, 0] > 0).cpu().numpy()
                    if m.sum() >= 50 and not p.exists():
                        cv2.imwrite(str(p), m.astype(np.uint8) * 255)
                        counts[side] += 1
    shutil.rmtree(frames)
    info = {"episode": ep, "n_frames": n, "keyframes": len(prompts), "masks": counts, "wall_time_s": time.time() - t0,
            "fps": n / (time.time() - t0), "gpu": torch.cuda.get_device_name(0), "checkpoint": Path(SAM2_CKPT).name,
            "every": every}
    json.dump(info, open(out / "info.json", "w"), indent=1)
    return info


def worker(gpu: int, queue, every, tmp_root, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    while True:
        try:
            ep = queue.get_nowait()
        except Exception:
            return
        try:
            info = run_episode(ep, every, tmp_root)
            print(f"[gpu{gpu}] {ep}: {info['masks']} masks, {info['keyframes']} keyframes, "
                  f"{info['wall_time_s']:.0f}s ({info['fps']:.1f} fps)", flush=True)
            results.append(info)
        except Exception as e:  # keep the other episodes going
            print(f"[gpu{gpu}] {ep}: FAILED {type(e).__name__}: {e}", flush=True)


def contact_sheet(eps, per_ep=4, out=None):
    tiles = []
    for ep in eps:
        md = paths.REFS / ep / "masks_b"
        frames = sorted({int(p.stem) for s in SIDES for p in (md / s).glob("*.png")})
        if not frames:
            continue
        cap = cv2.VideoCapture(str(paths.video(ep, "ego_cam_b")))
        for fi in np.linspace(0, len(frames) - 1, per_ep).astype(int):
            f = frames[fi]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if not ok:
                continue
            for s, col in (("L", (40, 140, 255)), ("R", (60, 60, 230))):
                p = md / s / f"{f:06d}.png"
                if p.exists():
                    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0
                    im[m] = (0.5 * im[m] + 0.5 * np.array(col)).astype(np.uint8)
                    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(im, cnt, -1, col, 3)
            im = cv2.resize(im, (480, 300))
            cv2.putText(im, f"{ep[-6:]} f{f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(im)
    rows = [np.hstack(tiles[k:k + per_ep] + [np.zeros_like(tiles[0])] * (per_ep - len(tiles[k:k + per_ep])))
            for k in range(0, len(tiles), per_ep)]
    sheet = np.vstack(rows)
    out = out or paths.REFS / "masks_b_contact_sheet.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def main():
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--sheet-only", action="store_true")
    a = ap.parse_args()
    eps = [paths.ep_name(e) for e in a.episodes.split(",")] if a.episodes else paths.episodes()
    if not a.sheet_only:
        tmp_root = paths.ROOT / "data" / "tmp_camb"
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        q = manager.Queue()
        for e in sorted(eps, key=lambda e: -paths.num_frames(e)):  # longest first
            q.put(e)
        results = manager.list()
        procs = [ctx.Process(target=worker, args=(int(g), q, a.every, tmp_root, results)) for g in a.gpus.split(",")]
        t0 = time.time()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        res = list(results)
        json.dump({"episodes": res, "wall_time_s": time.time() - t0, "gpus": a.gpus},
                  open(paths.REFS / "masks_b_run.json", "w"), indent=1)
        print(f"done {len(res)}/{len(eps)} episodes in {time.time() - t0:.0f}s")
    print("contact sheet:", contact_sheet(eps))


if __name__ == "__main__":
    main()
