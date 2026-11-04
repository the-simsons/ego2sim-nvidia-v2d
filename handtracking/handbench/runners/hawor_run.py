"""Run HaWoR (CVPR 2025, methods/HaWoR @ git commit in meta) on cam_a of each episode and export raw outputs.

Runs inside methods/.venv-hawor with cwd = methods/HaWoR. One worker per GPU, episodes from a shared queue.

Input: cam_a undistorted to a pinhole with the FACTORY focal and the principal point at the image centre (HaWoR
assumes cx = W/2, cy = H/2). Same camera centre and orientation as the benchmark camera, so HaWoR's camera frame IS
the benchmark frame. Frames are written by us at the native 20.25 fps (HaWoR's own extractor resamples to 30 fps).

Export (data/hawor/<ep>/export.npz), per hand (0 = left, 1 = right, HaWoR's order) and frame:
  world_*      infilled world-space MANO (HaWoR end to end, its SLAM + Metric3D scale) and its verts
  T_w2c        per-frame SLAM camera (world -> camera, metric)
  observed     frames covered by HaWoR's per-chunk motion estimation (detections), i.e. NOT produced by the infiller
  cam_*        camera-space per-chunk outputs (before SLAM / infilling), verts
Usage: python -m handbench.runners.hawor_run [--episodes ...] [--gpus 0,1]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HB = Path(__file__).resolve().parents[2]  # handtracking/
HAWOR = HB / "methods" / "HaWoR"


def prepare_input(ep, out: Path):
    import cv2

    sys.path.insert(0, str(HB))
    from handbench import paths
    from handbench.calib import load_rig

    cams = load_rig(paths.rig_json())
    A = cams["ego_cam_a"]
    W, H = A.size
    f = float(A.K[0, 0])
    K_h = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    mx, my = cv2.initUndistortRectifyMap(A.K, A.dist, np.eye(3), K_h, (W, H), cv2.CV_32FC1)
    seq = out / ep
    img = seq / "extracted_images"
    img.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(paths.video(ep, "ego_cam_a")))
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        cv2.imwrite(str(img / f"{i:04d}.jpg"), cv2.remap(im, mx, my, cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        i += 1
    # HaWoR derives names from the video path; it never re-reads the video once frames exist
    vid = out / f"{ep}.mp4"
    if not vid.exists():
        os.symlink(paths.video(ep, "ego_cam_a"), vid)
    json.dump({"K": K_h.tolist(), "focal": f, "n_frames": i, "size": [W, H]}, open(out / "input.json", "w"), indent=1)
    return vid, f, i


def run_episode(ep, out_root: Path, prepare=None):
    import joblib
    import torch

    os.chdir(HAWOR)
    sys.path.insert(0, str(HAWOR))
    from hawor.utils.process import run_mano, run_mano_left
    from lib.eval_utils.custom_utils import load_slam_cam
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    from scripts.scripts_test_video.hawor_video import hawor_infiller, hawor_motion_estimation

    t0 = time.time()
    out = out_root / ep
    out.mkdir(parents=True, exist_ok=True)
    vid, focal, n = (prepare or prepare_input)(ep, out)
    t_in = time.time()
    args = argparse.Namespace(img_focal=focal, video_path=str(vid), input_type="file",
                              checkpoint="./weights/hawor/checkpoints/hawor.ckpt",
                              infiller_weight="./weights/hawor/checkpoints/infiller.pt", vis_mode="cam")
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    assert len(imgfiles) == n, (len(imgfiles), n)
    frame_chunks_all, _ = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)
    R_w2c, t_w2c, _, _ = load_slam_cam(slam_path)
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid_filled = hawor_infiller(args, start_idx, end_idx,
                                                                                        frame_chunks_all)
    t_run = time.time()

    T = pred_trans.shape[1]
    exp = {"n_frames": n, "T": T}
    # observed = frames inside HaWoR's detection chunks (per track; tracks merged by handedness)
    observed = np.zeros((2, T), bool)
    cam_verts = np.full((2, T, 778, 3), np.nan, np.float32)
    cam_pose = np.full((2, T, 45), np.nan, np.float32)
    cam_rot = np.full((2, T, 3), np.nan, np.float32)
    cam_betas = np.full((2, T, 10), np.nan, np.float32)
    cam_trans = np.full((2, T, 3), np.nan, np.float32)
    from hawor.utils.rotation import rotation_matrix_to_angle_axis  # noqa: E402  (HaWoR helper)

    cs = Path(seq_folder) / "cam_space"
    for tdir in sorted(cs.glob("*")) if cs.exists() else []:
        for jf in sorted(tdir.glob("*.json")):
            d = json.load(open(jf))
            a, b = (int(x) for x in jf.stem.split("_"))
            frames = np.arange(a, b + 1)
            ro = rotation_matrix_to_angle_axis(torch.tensor(d["init_root_orient"])[0]).numpy()
            hp = rotation_matrix_to_angle_axis(torch.tensor(d["init_hand_pose"])[0].reshape(-1, 3, 3)).numpy()
            hp = hp.reshape(len(frames), 45)
            tr = np.array(d["init_trans"])[0]
            be = np.array(d["init_betas"])[0]
            # handedness of this chunk: the same majority vote hawor_motion_estimation used for the flip
            side = _track_side(seq_folder, start_idx, end_idx, int(tdir.name))
            fn = run_mano if side == 1 else run_mano_left
            out_m = fn(torch.tensor(tr)[None].float().cuda(), torch.tensor(ro)[None].float().cuda(),
                       torch.tensor(hp)[None].float().cuda(), betas=torch.tensor(be)[None].float().cuda())
            v = out_m["vertices"][0].cpu().numpy()
            keep = frames < T
            observed[side, frames[keep]] = True
            cam_verts[side, frames[keep]] = v[keep]
            cam_pose[side, frames[keep]] = hp[keep]
            cam_rot[side, frames[keep]] = ro[keep]
            cam_betas[side, frames[keep]] = be[keep]
            cam_trans[side, frames[keep]] = tr[keep]
    world_verts = np.zeros((2, T, 778, 3), np.float32)
    for side, fn in ((0, run_mano_left), (1, run_mano)):
        o = fn(pred_trans[side:side + 1].cuda(), pred_rot[side:side + 1].cuda(), pred_hand_pose[side:side + 1].cuda(),
               betas=pred_betas[side:side + 1].cuda())
        world_verts[side] = o["vertices"][0].cpu().numpy()
    exp.update(world_trans=pred_trans.numpy(), world_rot=pred_rot.numpy(), world_pose=pred_hand_pose.numpy(),
               world_betas=pred_betas.numpy(), world_verts=world_verts, world_valid_filled=np.asarray(pred_valid_filled),
               R_w2c=R_w2c.numpy(), t_w2c=t_w2c.numpy(), observed=observed, cam_verts=cam_verts, cam_pose=cam_pose,
               cam_rot=cam_rot, cam_betas=cam_betas, cam_trans=cam_trans)
    meta = {"episode": ep, "hawor_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HAWOR,
                                                                  text=True).strip(),
            "checkpoint": "hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large",
            "mano_left": "stand-in (mirrored MANO_RIGHT)", "gpu": torch.cuda.get_device_name(0),
            "wall_time_s": t_run - t_in, "input_prep_s": t_in - t0, "fps": n / (t_run - t_in), "img_focal": focal}
    np.savez_compressed(out / "export.npz", meta=np.array(json.dumps(meta)), **exp)
    return meta


def _track_side(seq_folder, start_idx, end_idx, idx):
    """HaWoR merges its tracklets by handedness vote into final tracks {0: left, 1: right}; cam_space/<idx> uses
    those keys (hawor_motion_estimation), so the side is the index itself."""
    assert idx in (0, 1), idx
    return idx


def worker(gpu, q, out_root, results, prepare=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # this worker is already a spawned process; HaWoR/DROID call set_start_method("spawn") again -> make it a no-op
    import multiprocessing
    import torch.multiprocessing as tmp
    multiprocessing.set_start_method = tmp.set_start_method = lambda *a, **k: None
    while True:
        try:
            ep = q.get_nowait()
        except Exception:
            return
        try:
            m = run_episode(ep, out_root, prepare)
            print(f"[gpu{gpu}] {ep}: {m['wall_time_s']:.0f}s ({m['fps']:.2f} fps)", flush=True)
            results.append(m)
        except Exception as e:
            import traceback

            print(f"[gpu{gpu}] {ep}: FAILED {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}", flush=True)


def main():
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="")
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()
    sys.path.insert(0, str(HB))
    from handbench import paths

    eps = [paths.ep_name(e) for e in a.episodes.split(",")] if a.episodes else paths.episodes()
    out_root = HB / "data" / "hawor"
    out_root.mkdir(parents=True, exist_ok=True)
    done = [e for e in eps if (out_root / e / "export.npz").exists()]
    if done:
        print(f"skipping {len(done)} episodes with an export already: {done}")
        eps = [e for e in eps if e not in done]
    ctx = mp.get_context("spawn")
    man = ctx.Manager()
    q = man.Queue()
    for e in sorted(eps, key=lambda e: -paths.num_frames(e)):
        q.put(e)
    res = man.list()
    t0 = time.time()
    ps = [ctx.Process(target=worker, args=(int(g), q, out_root, res)) for g in a.gpus.split(",")]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    json.dump({"episodes": list(res), "wall_time_s": time.time() - t0, "gpus": a.gpus},
              open(out_root / "run.json", "w"), indent=1)
    print(f"done {len(res)}/{len(eps)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
