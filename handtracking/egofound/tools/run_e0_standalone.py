"""Standalone E0 (no Docker): the upstream V2D hand stage, run step by step with the upstream library code.

Same steps and arguments as video_to_data@8c778f9 `run_ego_reconstruction --hand_tracking hamer` (run_ego_wilor):
  1. frames (PNG per frame, as v2d_pipelines.extract_images)
  2. MoGe-2 depth with the given intrinsics          v2d.moge.lib.video_to_depth
     + stable intrinsics = temporal median of MoGe's per-frame intrinsics (v2d_depth stabilize_intrinsics)
  3. WiLoR detection + MANO on every frame          v2d.wilor.lib.video_to_hands
  4. WiLoR MANO silhouettes on the reference frame as SAM2 hand prompts   v2d.wilor.lib.render_prompt_masks
  5. SAM2 mask propagation                          v2d.sam2.lib.video_to_masks
  6. HaMeR per (track, frame) on the SAM2 masks     v2d.hamer.lib.masks_to_hands (bbox x1.7, min mask 1 px)
  7. align to MoGe depth (upstream align_hands)     v2d.hamer.lib.align_hands
  8. fill gaps <= 15 frames (marked interpolated)   v2d.wilor.lib.tracks_interpolate (betas fixed)
Each step runs as `python -m <module>` in this env (methods/.venv-e0), exactly what the Docker wrappers run inside their
images, and is skipped when its output exists (re-running resumes). Upstream files are never edited.

HaMeR crop / alignment options (not in 8c778f9; the cached Track 3 E0 run used them from uncommitted changes in the
shared video_to_data tree, kept here as tools/upstream_patches/v2d_hamer_wilor_crop_valid_depth_scale_clamp.diff and
rebuilt into methods/v2d_hamer_lib_patched by setup_e0_env.sh). Steps 6-7 run the patched files when any is given:
  --hamer-crop-source wilor   centre the HaMeR crop on the WiLoR hand box inside each SAM2 mask (mask: arm-inclusive)
  --valid-depth-only          drop invalid depth pixels before aligning (always on in the patched align_hands)
  --hand-scale-mode shared    one hand scale for both hands;  --hand-scale-min / --hand-scale-max clamp it
--cached-track3-settings sets the values the cached Track 3 run used (wilor, shared, max 1.25).

Input: an undistorted (pinhole) video and its intrinsics JSON {fx, fy, cx, cy, width, height}. Optional object box
(--object-detections: a Grounding DINO detections JSON, first box used), which upstream adds as SAM2 object track 1 so
align_hands ignores object pixels; hand tracks are then 2, 3, ... as upstream.

  methods/.venv-e0/bin/python tools/run_e0_standalone.py --video V.mp4 --intrinsics K.json --out DIR [--gpu 0]
      [--reference-frame 0|auto] [--object-detections dino_detections.json]
  --reference-frame auto: after WiLoR, the first frame where WiLoR detects one left and one right hand (else the frame
  with the most detections); the SAM2 hand prompts are rendered there, as upstream does at its reference frame.
Then hand_tracks.json handedness is reconciled from the interpolation vote, as upstream.
Writes DIR/{frames,depth,intrinsics,intrinsics_stable.json,wilor_raw,sam2_hand_prompt_masks,sam2_prompts.json,
hand_tracks.json,masks,hamer,hamer_aligned,hamer_aligned_filled} (upstream layout) and DIR/run_meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path("/usr1/e2s/video_to_data/reconstruction/data/weights")
HAMER_WEIGHTS = ROOT / "methods" / "weights-e0" / "hamer"  # symlinks; MANO from WiLoR's copy (see README)
UPSTREAM = ROOT / "methods" / "v2d_upstream_8c778f9"
HAMER_BBOX_EXPANSION = 1.7
HAMER_MASK_MIN_PIXELS = 1  # upstream default when HaMeR is the primary hand source
INTERP_BETAS, INTERP_MAX_GAP = "fixed", 15
PATCHED_HAMER = ROOT / "methods" / "v2d_hamer_lib_patched"
PATCH = ROOT / "tools" / "upstream_patches" / "v2d_hamer_wilor_crop_valid_depth_scale_clamp.diff"
BACKED_OFF = 75


def has_files(d: Path) -> bool:
    return d.is_dir() and any(d.iterdir())


def foreign_on_gpu(gpu: str) -> bool:
    """True if a process of another user runs on GPU `gpu`."""
    q = lambda *a: subprocess.run(["nvidia-smi", *a], capture_output=True, text=True).stdout  # noqa: E731
    uuid = [l.split(", ")[1] for l in q("--query-gpu=index,uuid", "--format=csv,noheader").splitlines()
            if l.split(", ")[0] == gpu]
    for line in q("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader").splitlines():
        u, pid = line.split(", ")
        if uuid and u == uuid[0]:
            user = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip()
            if user and user != os.environ.get("USER"):
                return True
    return False


class Runner:
    def __init__(self, out: Path, gpu: str | None):
        self.out, self.gpu, self.times, self.peaks = out, gpu, {}, {}
        self.env = dict(os.environ, PYOPENGL_PLATFORM="egl", PYTHONHASHSEED="0")
        if gpu is not None:
            self.env["CUDA_VISIBLE_DEVICES"] = gpu

    def step(self, label: str, output: Path, module: str, script: Path | None = None, **kw):
        """Run one upstream module unless its completion marker exists; a partial output (killed run) is removed."""
        marker = self.out / ".done" / label.replace(" ", "_")
        if marker.exists():
            print(f"[skip] {label}", flush=True)
            return
        if output.is_dir():
            shutil.rmtree(output)
        if self.gpu is not None and foreign_on_gpu(self.gpu):
            print(f"[back off] another user's process is on GPU {self.gpu} before: {label}", flush=True)
            sys.exit(BACKED_OFF)
        peak_f = self.out / ".done" / (label.replace(" ", "_") + ".peak.json")
        peak_f.parent.mkdir(exist_ok=True)
        # run_module_with_peak: the module runs unchanged (runpy) and its torch peak CUDA memory is recorded
        target = [str(script)] if script is not None else ["-m", module]
        cmd = [sys.executable, str(ROOT / "tools" / "run_module_with_peak.py"), str(peak_f)] + target
        for k, v in kw.items():
            if v is None or v is False:
                continue
            cmd += [f"--{k}"] if v is True else [f"--{k}", str(v)]
        print(f"[run] {label}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        subprocess.run(cmd, check=True, env=self.env)
        self.times[label] = round(time.time() - t0, 1)
        self.peaks[label] = json.loads(peak_f.read_text()) if peak_f.exists() else None
        marker.parent.mkdir(exist_ok=True)
        marker.write_text(json.dumps({"cmd": cmd, "wall_time_s": self.times[label], "torch_peak": self.peaks[label]}))


def extract_frames(video: Path, out: Path) -> int:
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    cap, i = cv2.VideoCapture(str(video)), 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out / f"{i:06d}.png"), im)
        i += 1
    return i


def stabilize_intrinsics(folder: Path, out: Path):
    """Temporal median of per-frame intrinsics (as v2d_depth.lib.stabilize_intrinsics, principal point not fixed)."""
    import numpy as np
    d = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    med = lambda k: float(np.median([x[k] for x in d]))  # noqa: E731
    out.write_text(json.dumps({"fx": med("fx"), "fy": med("fy"), "cx": med("cx"), "cy": med("cy"),
                               "width": int(med("width")), "height": int(med("height"))}, indent=4))


def write_prompts(out: Path, detections: list, ref: int, object_box: dict | None):
    """SAM2 prompts + hand_tracks.json as run_ego_wilor step 5 (hand prompts = rendered WiLoR masks)."""
    from v2d.common.datatypes import BoundingBox
    from v2d.sam2.lib.datatypes import Sam2Prompt, Sam2Prompts
    prompts, meta = [], []
    offset = 1
    if object_box is not None:
        prompts.append(Sam2Prompt(frame_index=ref, object_id=1, box=BoundingBox.from_dict(object_box)))
        offset = 2
    for i, det in enumerate(detections):
        oid = i + offset
        rel = os.path.relpath(out / "sam2_hand_prompt_masks" / str(oid) / f"{ref:06d}.png", out)
        if not (out / rel).exists():
            raise RuntimeError(f"missing rendered WiLoR prompt mask {rel}")
        prompts.append(Sam2Prompt(frame_index=ref, object_id=oid, mask_path=rel))
        meta.append({"object_id": oid, "role": "hand", "is_right": bool(det["is_right"]), "score": float(det["score"]),
                     "sam2_prompt_mask_path": rel})
    Sam2Prompts(prompts=prompts).save(str(out / "sam2_prompts.json"))
    (out / "hand_tracks.json").write_text(json.dumps({"reference_frame": ref, "sam2_hand_prompt_source": "wilor_mask",
                                                      "tracks": meta}, indent=2))


def auto_reference_frame(wilor_dir: Path) -> int:
    """First frame with one left and one right WiLoR detection, else the frame with the most detections."""
    best, best_n = 0, -1
    for p in sorted(wilor_dir.glob("*.json")):
        d = json.loads(p.read_text())
        sides = {bool(x["is_right"]) for x in d}
        if sides == {True, False}:
            return int(p.stem)
        if len(d) > best_n:
            best, best_n = int(p.stem), len(d)
    return best


def reconcile_handedness(hand_tracks: Path, handedness: Path):
    """hand_tracks.json is_right from tracks_interpolate's per-track vote (run_ego_wilor
    _reconcile_hand_tracks_from_handedness; that module imports every Docker wrapper, so the logic is repeated here)."""
    if not handedness.exists():
        return
    canon, ht = json.loads(handedness.read_text()), json.loads(hand_tracks.read_text())
    changed = False
    for t in ht.get("tracks", []):
        e = canon.get(str(t["object_id"]))
        if e is not None and bool(t["is_right"]) != bool(e["is_right"]):
            print(f"  hand_tracks.json: track {t['object_id']} is_right {t['is_right']} -> {bool(e['is_right'])}")
            t["is_right"], changed = bool(e["is_right"]), True
    if changed:
        hand_tracks.write_text(json.dumps(ht, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--intrinsics", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--reference-frame", default="0", help="frame index, or auto (see above)")
    ap.add_argument("--object-detections", type=Path, default=None)
    ap.add_argument("--hamer-crop-source", choices=("mask", "wilor"), default="mask")
    ap.add_argument("--valid-depth-only", action="store_true")
    ap.add_argument("--hand-scale-mode", choices=("per_track", "shared"), default="per_track")
    ap.add_argument("--hand-scale-min", type=float, default=None)
    ap.add_argument("--hand-scale-max", type=float, default=None)
    ap.add_argument("--cached-track3-settings", action="store_true",
                    help="= --hamer-crop-source wilor --valid-depth-only --hand-scale-mode shared --hand-scale-max 1.25")
    a = ap.parse_args()
    if a.cached_track3_settings:
        a.hamer_crop_source, a.valid_depth_only, a.hand_scale_mode, a.hand_scale_max = "wilor", True, "shared", 1.25
    patched = (a.hamer_crop_source != "mask" or a.valid_depth_only or a.hand_scale_mode != "per_track"
               or a.hand_scale_min is not None or a.hand_scale_max is not None)
    if patched and not (PATCHED_HAMER / "align_hands.py").exists():
        raise SystemExit(f"{PATCHED_HAMER} missing: run tools/setup_e0_env.sh")
    hamer_script = lambda name: PATCHED_HAMER / f"{name}.py" if patched else None  # noqa: E731
    patched_kw = lambda **kw: kw if patched else {}  # noqa: E731
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = Runner(out, a.gpu)
    smi = None
    if a.gpu is not None:  # GPU memory sampler (whole GPU, 200 ms), as the instructions ask
        smi = subprocess.Popen(["nvidia-smi", "-i", a.gpu, "--query-gpu=timestamp,memory.used", "--format=csv",
                                "-lms", "200"], stdout=open(out / f"vram_{int(t0)}.csv", "w"))
    shutil.copy(a.intrinsics, out / "input_intrinsics.json")

    if not (out / ".done" / "frames").exists():
        shutil.rmtree(out / "frames", ignore_errors=True)
        n = extract_frames(a.video, out / "frames")
        (out / ".done").mkdir(exist_ok=True)
        (out / ".done" / "frames").write_text(str(n))
        print(f"[run] frames: {n}", flush=True)
    r.step("MoGe depth + intrinsics", out / "depth", "v2d.moge.lib.video_to_depth",
           video_path=a.video.resolve(), depth_folder=out / "depth", intrinsics_folder=out / "intrinsics",
           weights_path=WEIGHTS / "moge" / "model.pt", input_intrinsics_path=out / "input_intrinsics.json")
    if not (out / ".done" / "MoGe_depth_+_intrinsics").exists() or not (out / "intrinsics_stable.json").exists():
        stabilize_intrinsics(out / "intrinsics", out / "intrinsics_stable.json")
    r.step("WiLoR per-frame detection + MANO", out / "wilor_raw",
           "v2d.wilor.lib.video_to_hands", video_path=a.video.resolve(), output_dir=out / "wilor_raw",
           weights_dir=WEIGHTS / "wilor")
    ref = auto_reference_frame(out / "wilor_raw") if a.reference_frame == "auto" else int(a.reference_frame)
    print(f"[ref] reference frame {ref}", flush=True)
    ref_wilor = out / "wilor_raw" / f"{ref:06d}.json"
    detections = json.loads(ref_wilor.read_text())
    if not detections:
        raise RuntimeError(f"WiLoR found no hands in reference frame {ref}")
    object_box = None
    if a.object_detections is not None:
        object_box = json.loads(a.object_detections.read_text())[0]["box"]
    offset = 2 if object_box is not None else 1
    r.step("Render WiLoR hand-mask prompts", out / "sam2_hand_prompt_masks", "v2d.wilor.lib.render_prompt_masks",
           wilor_json=ref_wilor, image_path=out / "frames" / f"{ref:06d}.png", output_dir=out / "sam2_hand_prompt_masks",
           mano_assets_root=WEIGHTS / "wilor" / "pretrained_models", frame_index=ref, first_object_id=offset)
    if not (out / ".done" / "SAM2_mask_propagation").exists():
        write_prompts(out, detections, ref, object_box)
    r.step("SAM2 mask propagation", out / "masks", "v2d.sam2.lib.video_to_masks",
           video_path=a.video.resolve(), prompts_path=out / "sam2_prompts.json", masks_dir=out / "masks",
           weights_dir=WEIGHTS / "sam2")
    object_masks = out / "masks" / "1" if object_box is not None else None
    hamer_mano = HAMER_WEIGHTS / "_DATA" / "data"
    r.step("HaMeR per-frame regression", out / "hamer", "v2d.hamer.lib.masks_to_hands",
           frames_dir=out / "frames", masks_dir=out / "masks", tracks_path=out / "hand_tracks.json",
           output_dir=out / "hamer", weights_dir=HAMER_WEIGHTS, bbox_expansion=HAMER_BBOX_EXPANSION,
           mask_min_pixels=HAMER_MASK_MIN_PIXELS, script=hamer_script("masks_to_hands"),
           **patched_kw(crop_source=a.hamer_crop_source if a.hamer_crop_source != "mask" else None,
                        wilor_raw_dir=out / "wilor_raw" if a.hamer_crop_source == "wilor" else None))
    r.step("Align HaMeR hands to depth", out / "hamer_aligned", "v2d.hamer.lib.align_hands",
           hamer_dir=out / "hamer", depth_dir=out / "depth", intrinsics_path=out / "intrinsics_stable.json",
           mano_assets_root=hamer_mano, output_dir=out / "hamer_aligned", hand_masks_dir=out / "masks",
           object_masks_dir=object_masks, mask_min_pixels=HAMER_MASK_MIN_PIXELS, script=hamer_script("align_hands"),
           **patched_kw(hand_scale_mode=a.hand_scale_mode, hand_scale_min=a.hand_scale_min,
                        hand_scale_max=a.hand_scale_max))
    r.step("Interpolate HaMeR aligned frames", out / "hamer_aligned_filled", "v2d.wilor.lib.tracks_interpolate",
           aligned_dir=out / "hamer_aligned", masks_dir=out / "masks", output_dir=out / "hamer_aligned_filled",
           betas=INTERP_BETAS, max_gap_frames=INTERP_MAX_GAP)
    reconcile_handedness(out / "hand_tracks.json", out / "hamer_aligned_filled" / "handedness.json")

    peak = None
    if smi is not None:
        smi.terminate()
        vals = []
        for f in out.glob("vram_*.csv"):
            for line in f.read_text().splitlines()[1:]:
                try:
                    vals.append(int(line.split(",")[1].split()[0]))
                except (IndexError, ValueError):
                    pass
        peak = max(vals) if vals else None
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: E731
    gpu_name = None
    if a.gpu is not None:
        gpu_name = subprocess.run(["nvidia-smi", "-i", a.gpu, "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip()
    meta = {"method": "E0 standalone", "upstream": "video_to_data@8c778f9 (read-only export, " + str(UPSTREAM) + ")",
            "repo_commit": git("rev-parse", "HEAD"), "video": str(a.video.resolve()),
            "intrinsics": json.loads(a.intrinsics.read_text()), "reference_frame": ref,
            "object_detections": str(a.object_detections) if a.object_detections else None,
            "weights": {"moge": str(WEIGHTS / "moge" / "model.pt"), "wilor": str(WEIGHTS / "wilor"),
                        "sam2": str(WEIGHTS / "sam2"), "hamer": str(HAMER_WEIGHTS) + " (symlinks)"},
            "args": {"hamer_bbox_expansion": HAMER_BBOX_EXPANSION, "hamer_mask_min_pixels": HAMER_MASK_MIN_PIXELS,
                     "interp_betas": INTERP_BETAS, "interp_max_gap_frames": INTERP_MAX_GAP,
                     "sam2_hand_prompt_source": "wilor_mask", "hamer_crop_source": a.hamer_crop_source,
                     "valid_depth_only": patched, "hand_scale_mode": a.hand_scale_mode,
                     "hand_scale_min": a.hand_scale_min, "hand_scale_max": a.hand_scale_max},
            "hamer_patch": ({"diff": str(PATCH), "sha256": __import__("hashlib").sha256(PATCH.read_bytes()).hexdigest()}
                            if patched else None),
            "gpu": gpu_name, "nvidia_smi_peak_mib": peak, "step_wall_time_s": r.times, "step_torch_peak": r.peaks, "wall_time_s": round(time.time() - t0, 1)}
    prev = out / "run_meta.json"
    if prev.exists() and not r.times:  # nothing re-ran: keep the timings of the run that produced the outputs
        return
    prev.write_text(json.dumps(meta, indent=1))
    print(json.dumps({"wall_time_s": meta["wall_time_s"], "steps": r.times}))


if __name__ == "__main__":
    main()
