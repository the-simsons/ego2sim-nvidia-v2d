"""Figures for the Ego2Sim harness breakdown doc, from the ep41 white pot run.
  doc_harness_figs.py OUT_DIR
Run with any env that has Pillow + numpy (e.g. envs/recon_host)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

A = Path.home() / "v2d/agenticSTAR"
CAP = A / "captures/white_pot_e2s"
RUN = A / "runs/ep41_white_pot_20261016-200728"
IT = RUN / "iterations"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 22)
except OSError:
    FONT = ImageFont.load_default()


def r(i):
    return IT / f"{i:06d}/renders/0001"


def fit_h(im, h):
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)


def row(panels, h=360, pad=12, label_h=34):
    ims = [fit_h(p.convert("RGB"), h) for p, _ in panels]
    W = sum(i.width for i in ims) + pad * (len(ims) + 1)
    canvas = Image.new("RGB", (W, h + label_h + 2 * pad), "white")
    d = ImageDraw.Draw(canvas); x = pad
    for im, (_, lab) in zip(ims, panels):
        canvas.paste(im, (x, pad + label_h)); d.text((x, pad + 4), lab, fill=(30, 30, 30), font=FONT); x += im.width + pad
    return canvas


def tint(rgb, mask, color, a=0.5):
    out = np.asarray(rgb.convert("RGB"), float).copy(); m = np.asarray(mask.convert("L").resize(rgb.size)) > 127
    out[m] = (1 - a) * out[m] + a * np.array(color, float)
    return Image.fromarray(out.astype(np.uint8))


def depth_img(npz, size):
    z = np.load(npz); d = z["local"][..., 2]; keep = z["keep"] & np.isfinite(d) & (d > 0)
    lo, hi = np.percentile(d[keep], [2, 98]); t = np.clip((d - lo) / (hi - lo), 0, 1)
    rgb = np.stack([255 * t, 255 * (1 - np.abs(t - 0.5) * 2), 255 * (1 - t)], -1)  # near blue -> far red
    rgb[~keep] = 235
    return Image.fromarray(rgb.astype(np.uint8)).resize(size, Image.NEAREST)


def save(im, name, max_w=1600):
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(OUT / name, optimize=True); print(name, im.size)


f = "000080"
frame = Image.open(CAP / f"frames/{f}.jpg")
save(row([(frame, "frame 080"),
          (tint(frame, Image.open(CAP / f"mask_object/{f}.png"), (40, 170, 90)), "object mask (SAM2)"),
          (tint(frame, Image.open(CAP / f"mask_hand/{f}.png"), (220, 60, 60)), "hand mask = don't care"),
          (depth_img(CAP / f"depth/{f}.npz", frame.size), "stereo depth (near blue, far red)")]), "s1_capture.png")
save(Image.open(RUN / "frame_sheets/frame_sheet_001.png").convert("RGB"), "s2_keyframes.png")
save(row([(Image.open(r(1) / "turntable_sheets/turntable_sheet_rest.png"), "iteration 1: turntable of the first build"),
          (Image.open(r(1) / "side_by_side_000120_preview.png"), "iteration 1, frame 120: IoU 0.473")], h=420),
     "s4_first_build.png")
save(row([(Image.open(r(12) / f"side_by_side_{f}_preview.png"), "side by side"),
          (Image.open(r(12) / f"overlap_{f}.png"), "overlap"),
          (Image.open(r(12) / f"depth_residual_{f}.png"), "depth residual")]), "s5_pass_views.png")
save(row([(Image.open(r(2) / f"overlap_{f}.png"), "iteration 2, before the sweep: IoU 0.542"),
          (Image.open(r(4) / f"overlap_{f}.png"), "iteration 4, after the sweep: IoU 0.895")]), "s6_sweep.png")


def render_panel(i):  # the RENDER third of a side-by-side preview
    im = Image.open(r(i) / f"side_by_side_{f}_preview.png"); w = im.width // 3
    return im.crop((w, 0, 2 * w, im.height))


sbs = Image.open(r(12) / f"side_by_side_{f}_preview.png"); w = sbs.width // 3
save(row([(sbs.crop((2 * w, 0, 3 * w, sbs.height)), "frame 080, masked source"),
          (render_panel(8), "iteration 8: IoU 0.925, wrong tilt"),
          (render_panel(12), "iteration 12: IoU 0.895, re-seeded by hand")]), "s6_local_minimum.png")
save(row([(Image.open(r(12) / "turntable_sheets/turntable_sheet_rest.png"), "final shape (iteration 12)")], h=520),
     "s8_final_mesh.png")
