"""Video-only camera path from Pi3X (feed-forward visual geometry, yyfz/Pi3; BSD-style licence), run as chunked
video odometry (pi3.pipe.pi3x_vo: chunks aligned by Sim(3) on their overlap, metric scale from the model's head).

    python tools/pi3_camera.py --frames DIR --out camera_vo_pi3.npz [--intrinsics JSON] [--chunk 16 --overlap 6]
        [--scale-depth MOGE_DIR --exclude MASKDIR,...]
--intrinsics feeds the known camera to every chunk (rays condition) instead of letting the model guess it.
--scale-depth reports the median ratio of that depth (v2d PNG encoding) to Pi3X's own depth on the static background
(object/hand masks excluded) and applies it (unless --no-rescale), so the path is in the same metric as the object tracker.
Writes camera_to_world (N,4,4), world = camera 0, and OUT.json. Needs the env at /usr1/e2s/rborah_v2d/envs/pi3 and the
Pi3 checkout on PI3_DIR (default /usr1/e2s/rborah_v2d/third_party/Pi3)."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("PI3_DIR", "/usr1/e2s/rborah_v2d/third_party/Pi3"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--intrinsics", help="JSON file or inline JSON with fx, fy, cx, cy, width, height (full resolution)")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=6)
    ap.add_argument("--pixels", type=int, default=255000, help="model input size (pixels per frame)")
    ap.add_argument("--scale-depth")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--no-rescale", action="store_true", help="keep the model's own metric scale (1.4-2x too large on our clips)")
    a = ap.parse_args(argv)

    import cv2
    import torch
    from pi3.models.pi3x import Pi3X
    from pi3.pipe.pi3x_vo import Pi3XVO
    from pi3.utils.basic import load_multimodal_data

    dev = torch.device("cuda")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    model = Pi3X().to(dev).eval()        # weights file only, as the Pi3 examples' --ckpt path does
    missing, unexpected = model.load_state_dict(load_file(hf_hub_download("yyfz233/Pi3X", "model.safetensors")), strict=False)
    if missing or unexpected:
        print(f"weights: {len(missing)} missing, {len(unexpected)} unexpected keys")
    imgs, _ = load_multimodal_data(a.frames, conditions=None, interval=1, PIXEL_LIMIT=a.pixels, device=dev)
    imgs = imgs["images"] if isinstance(imgs, dict) else imgs
    if imgs.ndim == 4:
        imgs = imgs[None]
    B, N, _, H, W = imgs.shape
    print(f"{N} frames at {W}x{H}")
    if a.intrinsics:
        K0 = json.loads(Path(a.intrinsics).read_text()) if Path(a.intrinsics).is_file() else json.loads(a.intrinsics)
        sx, sy = W / K0["width"], H / K0["height"]
        K = torch.tensor([[K0["fx"] * sx, 0, K0["cx"] * sx], [0, K0["fy"] * sy, K0["cy"] * sy], [0, 0, 1]], device=dev).float()
        fwd = model.forward

        def forward_with_K(chunk, **kw):
            n = chunk.shape[1]
            kw["intrinsics"] = K[None, None].repeat(chunk.shape[0], n, 1, 1)
            kw["with_prior"] = True
            kw["mask_add_ray"] = torch.ones((chunk.shape[0], n), dtype=torch.bool, device=dev)
            return fwd(chunk, **kw)
        model.forward = forward_with_K
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        res = Pi3XVO(model)(imgs=imgs, chunk_size=a.chunk, overlap=a.overlap, dtype=dtype)
    P = res["camera_poses"][0].float().cpu().numpy()
    pts = res["points"][0].float().cpu().numpy()            # N,H,W,3 world
    conf = res["conf"][0].float().cpu().numpy()
    # Pi3XVO chains chunks with Sim(3) and leaves the chunk scale inside the pose's 3x3 block; project it back to the
    # nearest rotation (the translation already carries the scale)
    U, sv, Vt = np.linalg.svd(P[:, :3, :3])
    P = P.copy()
    P[:, :3, :3] = U @ Vt
    P[np.linalg.det(P[:, :3, :3]) < 0, :3, 2] *= -1
    c2w = np.linalg.inv(P[0]) @ P
    info = {"frames": int(N), "input_size": [int(W), int(H)], "chunk": a.chunk, "overlap": a.overlap,
            "chunk_scale_range": [float(sv.mean(1).min()), float(sv.mean(1).max())],
            "intrinsics_condition": bool(a.intrinsics), "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
            "trajectory_source": "video"}
    scale = 1.0
    if a.scale_depth:
        from e2s.masks import mask_dirs, union
        excl = mask_dirs(a.exclude) if a.exclude else []
        frames = sorted(Path(a.frames).glob("*.jpg"))
        ratios = []
        for i in range(0, N, 5):
            Xc = np.einsum("ij,hwj->hwi", np.linalg.inv(P[i])[:3, :3], pts[i]) + np.linalg.inv(P[i])[:3, 3]
            z = Xc[..., 2]
            raw = cv2.imread(str(Path(a.scale_depth) / f"{frames[i].stem}.png"), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            d = 65535.0 / np.maximum(cv2.resize(raw, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32), 1.0) - 1.0
            ok = (conf[i] > 0.3) & (z > 0.05) & (d > 0.05) & (d < 5)
            if excl:
                ok &= cv2.dilate(union(excl, int(frames[i].stem), (H, W)), np.ones((9, 9), np.uint8)) == 0
            if ok.sum() > 500:
                ratios.append(float(np.median(d[ok] / z[ok])))
        r = np.array(ratios)
        info["depth_ratio_median"] = float(np.median(r)) if len(r) else None
        info["depth_ratio_p10_p90"] = [float(np.percentile(r, 10)), float(np.percentile(r, 90))] if len(r) else None
        if not a.no_rescale and len(r):
            scale = float(np.median(r))
    c2w[:, :3, 3] *= scale
    info["applied_scale"] = scale
    np.savez(a.out, camera_to_world=c2w, method="pi3x_vo", scale=scale)
    json.dump(info, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    L = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1).sum()
    print(f"{a.out}: path {L:.2f} m, {info}")


if __name__ == "__main__":
    main()
