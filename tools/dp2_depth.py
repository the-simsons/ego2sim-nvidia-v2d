"""Per-frame metric depth for one clip with Depth Pro 2 (Apple; arXiv 2611.26603; inference code under Apple's licence,
weights for research use only), in video mode: frames attend to each other in 16-frame windows and share one focal length.

    python tools/dp2_depth.py --frames DIR|VIDEO --out DEPTH_DIR [--model depthpro2-l] [--scale 1.0] [--f-px F | --intrinsics JSON]
Writes DEPTH_DIR/NNNNNN.png in the v2d encoding (pixel = 65535 / (depth_m + 1), as MoGe's output here) at the frames'
full resolution, and DEPTH_DIR/dp2.json (model, focal length used or estimated, timing, peak GPU memory).
--frames is a dir of *.jpg (outputs keep their names) or a video file (outputs NNNNNN.png by frame index).
With a calibrated camera pass --intrinsics (its fx is used); without --f-px / --intrinsics the model's field-of-view
estimator picks the focal length (on our uncalibrated phone clips it gave 818-873 px vs the assumed 713).
--scale < 1 runs the model on downscaled frames (memory); the focal length is scaled with them and depth is upsampled.
Env: /usr1/e2s/rborah_v2d/envs/dp2 (HF_HOME=/usr1/e2s/rborah_v2d/hf)."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="depthpro2-l")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--f-px", type=float, help="known focal length in full-resolution pixels")
    ap.add_argument("--intrinsics", help="JSON with fx (full-resolution pixels); overrides --f-px")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=4)
    a = ap.parse_args(argv)

    import torch
    from depth_pro_2 import create_model_and_transforms

    if a.intrinsics:
        a.f_px = float(json.load(open(a.intrinsics))["fx"])
    src = Path(a.frames)
    stems = []

    def read_bgr():
        if src.is_dir():
            for f in sorted(src.glob("*.jpg")):
                stems.append(f.stem)
                yield cv2.imread(str(f))
        else:
            cap = cv2.VideoCapture(str(src))
            i = 0
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                stems.append(f"{i:06d}")
                i += 1
                yield im
    shape = {}
    model, transform = create_model_and_transforms(a.model)

    def frames():
        for im in read_bgr():
            shape.setdefault("hw", im.shape[:2])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if a.scale != 1.0:
                im = cv2.resize(im, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
            yield transform(im)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f_px = a.f_px * a.scale if a.f_px else None
    t0 = time.time()
    focal, n = None, 0
    with torch.no_grad():
        for i, pred in enumerate(model.infer_video(frames(), window=a.window, overlap=a.overlap, f_px=f_px)):
            h0, w0 = shape["hw"]
            d = pred["depth"].float().cpu().numpy()
            if focal is None and pred.get("focallength_px") is not None:
                focal = float(np.asarray(pred["focallength_px"].float().cpu() if hasattr(pred["focallength_px"], "cpu")
                                         else pred["focallength_px"]).ravel()[0]) / a.scale
            if d.shape != (h0, w0):
                d = cv2.resize(d, (w0, h0), interpolation=cv2.INTER_LINEAR)
            d = np.where(np.isfinite(d) & (d > 0), d, 0.0)
            png = np.where(d > 0, np.clip(65535.0 / (d + 1.0), 1, 65534), 65535).astype(np.uint16)
            cv2.imwrite(str(out / f"{stems[i]}.png"), png)
            n += 1
    info = {"model": a.model, "frames": n, "input_scale": a.scale, "f_px_given": a.f_px,
            "focal_px_used_fullres": focal, "seconds": time.time() - t0,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
            "window": a.window, "overlap": a.overlap, "licence": "weights: research use only (Apple LICENSE_MODEL)"}
    json.dump(info, open(out / "dp2.json", "w"), indent=1)
    print(f"{out}: {info}")


if __name__ == "__main__":
    main()
