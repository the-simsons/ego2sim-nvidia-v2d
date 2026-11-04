# Status against INSTRUCTIONS.md

> **Moved 4 Nov 2026:** this folder is now `handtracking/egofound/` (was `handtracking-egofound/` next to
> `handtracking/`). Paths below and absolute paths in run records (`metrics.json`, tables) are as they were when
> each run happened; on this machine a symlink `handtracking-egofound -> handtracking/egofound` keeps them valid.

2 Nov 2026, branch `egofound3r-eval`. Check-in 1 reviewed; user said go ahead. **Working on check-in 2** (and F0 on Track 3).

## Setup: done

- `weights-vggt/` gitignored; both checkpoints present, hashes in [DOWNLOADS.md](DOWNLOADS.md).
- EgoFound3R env: `../handtracking/methods/.venv-egofound3r` (repo at `264f4df`), config `configs/egofound3r.toml`.
  CPU smoke test on 2 frames of Track 3 episode 41 runs end to end.
- Standalone E0 env: `methods/.venv-e0` via `tools/setup_e0_env.sh` (HaMeR, WiLoR-mini, MoGe import; torch 2.5.0
  because WiLoR-mini pins `torch<=2.5`). Upstream code exported read-only at `8c778f9` (`methods/v2d_upstream_8c778f9`).
  Weights read from `/usr1/e2s/video_to_data/reconstruction/data/weights`; HaMeR's `_DATA/data/{mano,models}` are not
  readable, so MANO comes from the WiLoR folder.
- AssemblyHands val (option B, 4.01 GB, approved over budget): 2 sequences, ~29,600 hand-frames, 98 % seen by 2
  cameras. Facts in [DOWNLOADS.md](DOWNLOADS.md).

## Check-in 1: ready for review

- [x] EgoFound3R (F0) on Track 3 episode 41 (299 frames), RTX 4090: 91 s wall, 7.5 GB peak (nvidia-smi), forward
  3.7 s. Outputs + `overlay_multimode.mp4` in `data/egofound3r/track3/episode_000041/`, provenance in `run_meta.json`.
  Run by `tools/run_egofound3r_when_gpu_free.sh` (backed off once when another user's job took GPU 1).
  Input: `tools/prepare_track3_video.py` (undistorted cam_a pinhole, 20.2541 fps), resize `keep_aspect` (384×512;
  instructions say 512×512, stretching distorts geometry).
  Note: hands come out at 10 fps (158 frames, `H_fps`), camera at 2 fps; evaluation must use those frames or interpolate.
- [x] AssemblyHands file list + subset (done, user chose option B).

## Since check-in 1 (2 Nov)

- Comparison video ep 41 (E0 / F0 / best handbench): `results/track3/videos/ep41_E0_F0_best.mp4`.
- F0 -> handbench: `tools/egofound3r_export.py` (EgoFound3R env; MANO fit to the 778 vertices: median 3.3 mm mean /
  13 mm max per hand-frame, does not drop with 5x more iterations, so EgoFound3R's mesh is not exactly MANO) and
  `tools/egofound3r_to_hands.py` (bench env; scores EgoFound3R's own joints; `--joints mano` = fitted MANO joints).
  Results root `results/track3/` (HANDBENCH_RESULTS), E0/E1best/E8 linked from handbench.
- Ep 41 only: F0 3D error 107 mm vs E0 241 mm (depth bias +98 vs +238 mm), cam_c kp 36.7 vs 42.7 px. EgoFound3R's own
  focal 277/300 px (fx/fy, model px) vs true 290; known intrinsics can be passed (`--root-k-source external`): not run.
- F0 on the other 19 Track 3 episodes queued (`tools/run_egofound3r_when_gpu_free.sh`, log `logs/egofound3r_watch.log`).
- Standalone E0: `tools/run_e0_standalone.py` (upstream modules via `python -m`, no Docker; env `methods/.venv-e0` now
  also has SAM2 @2b90b9f (`--no-deps`, keeps torch 2.5.0) and v2d_mv; HaMeR weights tree `methods/weights-e0/` of
  symlinks, MANO from WiLoR's copy). Validation run on ep 41 queued via `tools/when_gpu_free.sh` (input: cached
  AnyCalib-undistorted video + intrinsics + cached Grounding DINO box). Convert: `tools/e0_standalone_to_hands.py`.
- GPUs: another user's jobs on both GPUs since ~00:20 UTC 2 Nov (GPU 1 freed ~02:00).
- F0 episode_000000 timing contaminated (shared GPU with E0); re-run queued at the end of the F0 list.

## Check-in 2: standalone E0 validated (ep 41) — ready for review

Tables: `results/track3/E0_VALIDATION_TABLES.md` (handbench compare, baseline = cached E0).

| ep 41 | cached E0 | standalone, 8c778f9 as committed | standalone, cached-run settings |
|---|---|---|---|
| Tri MPJPE abc | 241.5 mm | 240.7 (-0.3 %) | 241.3 (-0.1 %) |
| depth bias | +237.5 mm | +236.7 | +237.4 |
| cam_b PCK@10 | 0.115 | 0.110 (-4.6 %) | 0.112 (-2.6 %) |
| cam_b IoU | 0.333 | 0.335 | 0.334 |
| jitter | 12.6 m/s² | 32.6 | 13.6 |

- The cached E0 (`outputs/track3_handfix`, 20 Oct) is **not** plain video_to_data@8c778f9: it ran with uncommitted
  changes to `v2d_hamer/lib/{masks_to_hands,align_hands}.py` in the shared tree (HaMeR crop on the WiLoR hand box,
  invalid-depth pixels dropped before alignment, one hand scale for both hands clamped to <= 1.25). Saved read-only as
  `tools/upstream_patches/v2d_hamer_wilor_crop_valid_depth_scale_clamp.diff` (rebuilt copies hash-identical to the shared
  files); `run_e0_standalone.py --cached-track3-settings` uses it. Without it, left hand frames 240-253 align to
  background depth (3-5 m), hence the jitter.
- Both standalone variants pass the 5 % gate; the cached-settings one matches on every metric (jitter +8 %).
- Compute (RTX 4090): MoGe 150 s, WiLoR 80 s, SAM2 23 s, HaMeR 67 s, align 46 s; nvidia-smi peak and the first run's
  HaMeR time (208 s) are not usable: an F0 run shared the GPU from 01:46 (fixed: watchers now flock per GPU). Clean
  re-timing of the full E0 is pending a free GPU.
- Decided (user, 2 Nov): E0 = video_to_data@8c778f9 + the HaMeR patch above (= the cached run). New E0 runs
  (AssemblyHands, HOT3D) use `run_e0_standalone.py --cached-track3-settings`; the report states the patch and gives the
  plain-8c778f9 numbers as a footnote.

## Check-in 3: F0 on all three test sets — ready for review

**Track 3** (20 episodes; `results/track3/REPORT_TABLES.md`, handbench, now with cam_c IoU from new cam_c SAM2 masks,
`handbench.build_masks_b --camera c`):

| method | 3D err (tri abc) | depth bias | cam_c kp | cam_c PCK@10 | cam_c IoU | jitter | coverage |
|---|---|---|---|---|---|---|---|
| E0 | 263 mm | +260 | 31.1 px | 0.110 | 0.365 | 13.4 | 100 % |
| F0 | 119 mm | +108 | 31.3 px | 0.137 | 0.399 | 4.8 | 50 % (10 fps) |
| E8fallback (handbench best, stereo) | 47.5 mm | +7.5 | 23.9 px | 0.348 | 0.437 | 3.1 | 98 % |

**HOT3D** (7 clips, seen in EgoFound3R training; `results/hot3d/F0/metrics.json`): F0 36.8 mm (median 32.1), bias +11,
PA 8.5 mm, 514 hand-frames (its 10 fps of 30), coverage 100 %. (Earlier HaWoR raw: 28.4 mm on all 1,433; not yet on
common frames.)

**AssemblyHands** (8 clips x 300 frames, 2 subjects, upright-rotated mono, 4,745 visible hand-frames;
`results/assemblyhands/F0/metrics.json`): F0 159 mm (lateral 118, bias +85), PA 15 mm, 2D 55 px, coverage 96 %.
Per-joint offset held out by sequence: 113 mm (not a joint-definition offset: ~110 mm "offsets").
Cause: the rectified headset cameras have f = 134 px (~120 deg across 480 px); EgoFound3R predicts f ~ 250
(pred/true focal 1.86; Track 3 0.96, HOT3D 0.99), so its camera-frame hands are squeezed ~2x towards the optical axis.
Drawn with its own K the joints sit on the hands. Known-K F0 (`--root-k-source external`) would test this directly.

Inputs: `tools/prepare_assemblyhands.py`, `tools/prepare_hot3d_video.py`, `tools/export_hot3d_gt.py`; scorer
`tools/score_groundtruth.py` (self-test: +1 cm shift -> 10.0 mm, PA 0). MANO fit residual median 2.4-3.6 mm on all sets.
Not yet: E0 on AssemblyHands / HOT3D (needs a hand reference frame per clip; no object box).

## Check-in 4: F2, F3, report — ready for review

Report: `results/egofound3r/REPORT.md` + `REPORT.pdf` (figures in `results/egofound3r/figures/`); commands in
`README.md`. Videos (gitignored, on disk): `results/track3/videos/ep{21,22,23}_E0_F0_F2_F3_camc.mp4`,
`results/assemblyhands/videos/{9012-c07c_005770,9081-c11b_000505}_E0_F0_F2_F3.mp4`.

| 3D error (mm) | E0 | F0 | F2 | F3 |
|---|---|---|---|---|
| AssemblyHands | 231 | 159 | 54.3 | **53.0** |
| HOT3D (seen in training) | 131 | 36.8 | **28.1** | 28.4 |
| Track 3 cam_c PCK@10 (held out) | 0.110 | 0.137 | 0.229 | **0.239** |

Deviations: F2 min-depth guard (shift rejected if any joint < 5 cm; 1 / 36 / 0 hand-frames); E0 = 8c778f9 + the
cached run's HaMeR patch (user decision); extra row F0-known-K. A CUDA outage (~18:20–18:27 UTC) failed 4 runs; all
re-run. Not done: F2/F3 on F0-known-K (user: not needed); Track 3 PA-MPJPE (handbench has centred error only).

## After check-in 4: F2/F3 on known-intrinsics F0 (user request, 2 Nov)

`tools/triangulated_shift.py --src data/egofound3r_known_k --out-suffix _known_k` (same keypoints and rules). Added as
extra rows to the report and tables:

| | F3 | F2-known-K | F3-known-K |
|---|---|---|---|
| HOT3D 3D err | 28.4 mm | **26.1 mm** (best overall) | 26.5 mm |
| AssemblyHands 3D err (common frames) | **50.6 mm** | 58.9 mm | 57.8 mm |
| Track 3 cam_c PCK@10 / IoU | 0.239 / 0.419 | 0.234 / 0.419 | **0.245** / 0.422 |

Known intrinsics + shift helps on normal lenses (HOT3D; marginal on Track 3), hurts on AssemblyHands' extreme lens.
F0-known-K alone keeps the best Track 3 cam_c kp (26.9 px) and IoU (0.443). All behind E8fallback on Track 3.

## Then (unchanged from INSTRUCTIONS.md)

2. Standalone E0 validated against cached E0 on Track 3 (within 5 %).
3. F0 on Track 3, AssemblyHands, HOT3D. Track 3 needs cam_c SAM2 masks (handbench has cam_b only).
4. F2, F3 (Track 3: triangulate from cam_a + cam_b only, cam_c held out), report, videos.
