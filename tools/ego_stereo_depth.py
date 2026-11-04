"""Metric depth for the undistorted ego_cam_a view from the ego mono stereo pair (OpenCV SGBM, or
FoundationStereo with --method fs).

Conventions (verified on episode 12, see NOTES.md): the factory intrinsics/distortion fit the published
videos as-is ("raw"); physically cam_c is LEFT, cam_b RIGHT, cam_a in between. Extrinsics come from
estimate_ego_extrinsics.py (R, t direction from the essential matrix; |t| from the factory baselines).

Writes into OUT:
  cam_a_undist.mp4            cam_a undistorted to a pinhole camera with the factory K (same size)
  intrinsics_undist.json      that pinhole K (v2d CameraIntrinsics JSON)
  stereo_depth/%06d.png       stereo depth in the cam_a pinhole view (v2d inverse-depth uint16; 0 m = invalid)
  stereo_stats.json           per-frame valid fraction / median depth

Usage: python ego_stereo_depth.py CALIB_JSON EXTR_JSON VIDEO_DIR EPISODE OUT [--max-frames N]
       [--method sgbm|fs] [--fs-ckpt PTH] [--fs-scale S] [--fs-iters N]
--method fs needs the FoundationStereo env (~/v2d/envs/foundation_stereo) and a GPU; the matcher is the only change,
rectification, reprojection into cam_a and the output format are identical to SGBM.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ego_calib import load_cams  # noqa: E402


def encode_depth(depth_m: np.ndarray) -> np.ndarray:
    """v2d DepthImage encoding: pixel = 65535 / (depth + 1); invalid (0 m) -> 65535 decodes to 0 m."""
    return np.clip(65535.0 / (depth_m + 1.0), 0, 65535).astype(np.uint16)


def camera_center_in_b(T_from_b: np.ndarray) -> np.ndarray:
    return -T_from_b[:3, :3].T @ T_from_b[:3, 3]


FS_ROOT = Path.home() / "v2d/FoundationStereo"


SAFE_MODULES = ("numpy", "omegaconf", "builtins", "collections", "typing", "enum")


def safe_load(path):
    """torch.load with the weights-only unpickler. The checkpoint's metadata also holds numpy scalars and its OmegaConf
    training config; data classes from SAFE_MODULES are allowlisted one at a time as the loader reports them, and any
    other global is refused (needs numpy < 2, whose scalar path matches the pickle)."""
    import importlib
    import re
    import torch
    for _ in range(64):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001
            m = (re.search(r"Unsupported global: GLOBAL ([\w.]+)", str(e))
                 or re.search(r"allowlisted via `add_safe_globals`, but got <class '([\w.]+)'>", str(e)))
            if not m or not m.group(1).startswith(SAFE_MODULES):
                raise
            mod, name = m.group(1).rsplit(".", 1)
            torch.serialization.add_safe_globals([getattr(importlib.import_module(mod), name)])
    raise RuntimeError(f"{path}: too many globals to allowlist")


def foundation_stereo_matcher(ckpt, scale, iters):
    """Left-view disparity (rectified px, full size) from FoundationStereo, as a function of two grey images."""
    import types
    import torch
    from omegaconf import OmegaConf
    sys.modules.setdefault("open3d", types.ModuleType("open3d"))  # only the demo's point-cloud helpers use it
    sys.path.insert(0, str(FS_ROOT))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    cfg = OmegaConf.load(Path(ckpt).parent / "cfg.yaml")
    cfg.setdefault("vit_size", "vitl")
    cfg["valid_iters"] = iters
    model = FoundationStereo(cfg)
    model.load_state_dict(safe_load(ckpt)["model"])
    model.cuda().eval()
    torch.autograd.set_grad_enabled(False)

    def match(L, R):
        H, W = L.shape
        to_t = lambda g: torch.as_tensor(np.repeat(cv2.resize(g, None, fx=scale, fy=scale)[..., None], 3, 2)
                                         ).cuda().float()[None].permute(0, 3, 1, 2)
        l, r = to_t(L), to_t(R)
        padder = InputPadder(l.shape, divis_by=32, force_square=False)
        l, r = padder.pad(l, r)
        with torch.autocast("cuda", enabled=True):
            disp = model.forward(l, r, iters=iters, test_mode=True)
        disp = padder.unpad(disp.float())[0, 0].cpu().numpy()
        disp = cv2.resize(disp, (W, H), interpolation=cv2.INTER_LINEAR) / scale
        disp[np.arange(W)[None, :] - disp < 0] = 0  # not visible in the right image
        return disp

    return match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("extr")
    ap.add_argument("video_dir")
    ap.add_argument("episode", type=int)
    ap.add_argument("out")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--num-disp", type=int, default=192)
    ap.add_argument("--method", choices=["sgbm", "fs"], default="sgbm")
    ap.add_argument("--fs-ckpt", default=str(FS_ROOT / "pretrained_models/23-51-11/model_best_bp2.pth"))
    ap.add_argument("--fs-scale", type=float, default=1.0)
    ap.add_argument("--fs-iters", type=int, default=32)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "stereo_depth").mkdir(parents=True, exist_ok=True)
    cams = load_cams(args.calib, rotated=False)
    A, B, C = cams["ego_cam_a"], cams["ego_cam_b"], cams["ego_cam_c"]
    ex = json.load(open(args.extr))

    # b -> c from the essential matrix, scaled with the factory |b-c| baseline.
    e_bc = ex["raw:ego_cam_b->ego_cam_c:K(ego_cam_b,ego_cam_c)"]
    R_bc = np.array(e_bc["R"])
    t_bc = np.array(e_bc["t_dir"]) * np.linalg.norm(camera_center_in_b(C.T_from_b))
    R_cb, t_cb = R_bc.T, -R_bc.T @ t_bc  # left = c, right = b
    # c -> a, scaled with the factory |c-a| distance.
    e_ca = ex["raw:ego_cam_c->ego_cam_a:K(ego_cam_c,ego_cam_a)"]
    dist_ca = np.linalg.norm(camera_center_in_b(C.T_from_b) - camera_center_in_b(A.T_from_b))
    R_ca, t_ca = np.array(e_ca["R"]), np.array(e_ca["t_dir"]) * dist_ca

    size = C.size
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(C.K, C.dist, B.K, B.dist, size, R_cb, t_cb.reshape(3, 1),
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    mapl = cv2.initUndistortRectifyMap(C.K, C.dist, R1, P1, size, cv2.CV_32FC1)
    mapr = cv2.initUndistortRectifyMap(B.K, B.dist, R2, P2, size, cv2.CV_32FC1)
    f_rect, baseline = P1[0, 0], -P2[0, 3] / P2[0, 0]
    K_rect_inv = np.linalg.inv(P1[:3, :3])
    W, H = size
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = (K_rect_inv @ np.stack([uu.ravel(), vv.ravel(), np.ones(W * H)])).T  # rectified-left rays

    # cam_a pinhole (factory K, no distortion), same size as the video.
    Wa, Ha = A.size
    Ka = A.K.copy()
    mapa = cv2.initUndistortRectifyMap(A.K, A.dist, np.eye(3), Ka, (Wa, Ha), cv2.CV_32FC1)
    json.dump({"fx": Ka[0, 0], "fy": Ka[1, 1], "cx": Ka[0, 2], "cy": Ka[1, 2], "width": Wa, "height": Ha},
              open(out / "intrinsics_undist.json", "w"), indent=1)

    if args.method == "fs":
        match = foundation_stereo_matcher(args.fs_ckpt, args.fs_scale, args.fs_iters)
    else:
        sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=args.num_disp, blockSize=5,
                                     P1=8 * 5 * 5, P2=32 * 5 * 5, disp12MaxDiff=1, uniquenessRatio=10,
                                     speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        match = lambda L, R: sgbm.compute(L, R).astype(np.float32) / 16.0

    vid = lambda cam: cv2.VideoCapture(f"{args.video_dir}/observation.images.{cam}/episode_{args.episode:06d}.mp4")
    ca, cb, cc = vid("ego_cam_a"), vid("ego_cam_b"), vid("ego_cam_c")
    fps = ca.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(str(out / "cam_a_undist.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wa, Ha))
    stats = []
    i = 0
    while True:
        oka, fa = ca.read()
        okb, fb = cb.read()
        okc, fc = cc.read()
        if not (oka and okb and okc) or (args.max_frames and i >= args.max_frames):
            break
        writer.write(cv2.remap(fa, *mapa, cv2.INTER_LINEAR))
        L = cv2.remap(cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY), *mapl, cv2.INTER_LINEAR)
        R = cv2.remap(cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY), *mapr, cv2.INTER_LINEAR)
        disp = match(L, R)
        valid = (disp > 1.0).ravel()
        z = np.zeros(W * H, np.float32)
        z[valid] = f_rect * baseline / disp.ravel()[valid]
        valid &= (z > 0.15) & (z < 5.0)
        X_rect = rays[valid] * z[valid, None]
        X_c = X_rect @ R1  # R1: c -> rect, so rect -> c is R1^T (row-vector form: X @ R1)
        X_a = X_c @ R_ca.T + t_ca
        front = X_a[:, 2] > 0.1
        X_a = X_a[front]
        uv = X_a @ Ka.T
        u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
        v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
        inside = (u >= 0) & (u < Wa) & (v >= 0) & (v < Ha)
        u, v, d = u[inside], v[inside], X_a[inside, 2]
        depth = np.full((Ha, Wa), np.inf, np.float32)
        np.minimum.at(depth, (v, u), d)  # z-buffer
        depth[~np.isfinite(depth)] = 0
        # fill the sparse splat (mono 1280 -> colour 2028 px) with a small min-filter over valid pixels
        filled = depth.copy()
        big = np.where(depth > 0, depth, 1e6).astype(np.float32)
        eroded = cv2.erode(big, np.ones((3, 3), np.uint8))
        hole = (depth == 0) & (eroded < 1e5)
        filled[hole] = eroded[hole]
        cv2.imwrite(str(out / "stereo_depth" / f"{i:06d}.png"), encode_depth(filled))
        vf = filled > 0
        stats.append({"frame": i, "valid_frac": float(vf.mean()),
                      "median_depth": float(np.median(filled[vf])) if vf.any() else None})
        i += 1
    writer.release()
    json.dump({"method": args.method, "rectified_f": f_rect, "baseline_m": baseline, "dist_ca_m": dist_ca,
               "frames": stats},
              open(out / "stereo_stats.json", "w"), indent=1)
    vfs = [s["valid_frac"] for s in stats]
    print(f"{i} frames; rectified f {f_rect:.1f}px baseline {baseline*100:.2f}cm |c-a| {dist_ca*100:.2f}cm; "
          f"cam_a valid depth fraction mean {np.mean(vfs):.1%}")


if __name__ == "__main__":
    main()
