# EgoFound3R vs the V2D hand baseline

Evaluation of E0 (V2D hand stage), F0 (EgoFound3R), F2 (+ triangulated-keypoint shift) and F3 (+ smoothing) on
Track 3, AssemblyHands and HOT3D. Report: [results/egofound3r/REPORT.md](results/egofound3r/REPORT.md). Task:
[INSTRUCTIONS.md](INSTRUCTIONS.md); progress log: [STATUS.md](STATUS.md); downloads: [DOWNLOADS.md](DOWNLOADS.md).

Run everything from `handtracking/egofound/`. Reuses handbench (the parent folder, `handtracking/`). Shorthands used below:

```bash
B=../.venv-bench/bin/python                          # handbench env (scoring, keypoints, conversion)
EF=$(cd ../methods/.venv-egofound3r/bin && pwd)/python  # EgoFound3R env
E0=methods/.venv-e0/bin/python                                     # standalone E0 env
```

GPU jobs go through watchers that wait for a free GPU, back off when another user's process appears, and never
put two of our jobs on one GPU (`tools/when_gpu_free.sh`, `tools/run_egofound3r_when_gpu_free.sh`).

## Environments

| env | how |
|---|---|
| EgoFound3R | `../methods/.venv-egofound3r` (repo `../methods/EgoFound3R` @264f4df), config `configs/egofound3r.toml`, weights in `weights-vggt/` (gitignored, hashes in DOWNLOADS.md) |
| standalone E0 | `tools/setup_e0_env.sh` (read-only export of video_to_data@8c778f9 + the HaMeR patch from `tools/upstream_patches/`) |
| handbench / RTMPose / SAM2 | `../.venv-bench`, `../methods/.venv-sam2` |

## Inputs and ground truth

```bash
$B tools/prepare_track3_video.py --episode 41                  # per episode: undistorted cam_a pinhole video
$B tools/prepare_assemblyhands.py                              # 8 clips + ground truth (needs data/assemblyhands, DOWNLOADS.md)
$B tools/prepare_hot3d_video.py                                # 7 HOT3D clips (handbench's pinhole view)
../methods/.venv-hawor/bin/python tools/export_hot3d_gt.py   # HOT3D ground truth (HOT3D's MANO layer)
python3 tools/write_root_k_sidecars.py data/track3_input data/hot3d_input data/assemblyhands_input   # for F0-known-K
(cd .. && methods/.venv-sam2/bin/python -m handbench.build_masks_b --camera c)          # Track 3 cam_c masks
```

## E0

```bash
# Track 3 (validation against the cached run; one episode)
V=/usr1/e2s/v2d_challenge_data/outputs/track3_handfix/episode_000041
$E0 tools/run_e0_standalone.py --video $V/anycalib/undistorted.mp4 --intrinsics $V/anycalib/undistorted_intrinsics.json \
    --object-detections $V/dino_detections.json --out data/e0_standalone_cached_settings/track3/episode_000041 \
    --gpu 0 --cached-track3-settings
E0_STANDALONE_RUNS=data/e0_standalone_cached_settings/track3 $B tools/e0_standalone_to_hands.py --episodes 41 \
    --method E0-standalone-cached-settings
# Track 3 E0 in the tables = the cached run (handbench results/E0), linked into results/track3/E0
# AssemblyHands + HOT3D (all clips, resumable)
tools/when_gpu_free.sh logs/e0_gt_sets.log -- tools/run_e0_on_gt_sets.sh GPU
$B tools/e0_standalone_to_groundtruth_hands.py --dataset assemblyhands     # and --dataset hot3d
```

## F0 (and F0-known-K)

```bash
tools/run_egofound3r_when_gpu_free.sh episode_000000 episode_000002 ...                           # Track 3
INPUT_DIR=data/assemblyhands_input OUT_DIR=data/egofound3r/assemblyhands tools/run_egofound3r_when_gpu_free.sh 9012-c07c_000520 ...
INPUT_DIR=data/hot3d_input OUT_DIR=data/egofound3r/hot3d tools/run_egofound3r_when_gpu_free.sh clip-001849 ...
# known intrinsics: KNOWN_K=1 and OUT_DIR=data/egofound3r_known_k/<set>
OMP_NUM_THREADS=4 $EF tools/egofound3r_export.py data/egofound3r/<set>/<clip>          # per clip: joints, MANO fit
$B tools/egofound3r_to_hands.py --episodes 0,2,...,45 --method F0                      # Track 3 -> handbench format
```

## F2 and F3

```bash
$B tools/detect_two_view_keypoints.py --dataset assemblyhands     # and hot3d (Track 3 uses handbench refs tri_ab)
$B tools/triangulated_shift.py --dataset track3                   # and assemblyhands, hot3d -> data/egofound3r_f2, _f3
$B tools/egofound3r_to_hands.py --episodes 0,2,...,45 --method F2 --runs data/egofound3r_f2/track3   # F3: _f3
```

## Scoring and tables

```bash
# Track 3 (handbench; results root = results/track3)
cd .. && HANDBENCH_RESULTS=$(realpath egofound/results/track3) OMP_NUM_THREADS=1 \
    .venv-bench/bin/python -m handbench.run_eval --method F3 --set v2d --jobs 8       # per method
HANDBENCH_RESULTS=... .venv-bench/bin/python -m handbench.compare --methods E0,F0,F0-known-K,F2,F3,E1best-sgbm,E8fallback-e1best-smooth \
    --set v2d --baseline E0 --out $HANDBENCH_RESULTS/REPORT_TABLES.md
# AssemblyHands / HOT3D
$B tools/score_groundtruth.py --dataset assemblyhands --method F0                                    # EgoFound3R exports
$B tools/score_groundtruth.py --dataset assemblyhands --method F2 --pred-dir data/egofound3r_f2      # F3: _f3; known-K: _known_k
$B tools/score_groundtruth.py --dataset assemblyhands --method E0 --source hands --pred-dir data/e0_hands/assemblyhands
$B tools/compare_groundtruth.py --dataset assemblyhands --methods E0,F0,F2,F3,F0-known-K
```

## Videos

```bash
cd .. && HANDBENCH_RESULTS=... .venv-bench/bin/python -m handbench.overlay --episode 23 --methods E0,F0,F2,F3 \
    --second-view c --infilled-colour --out $HANDBENCH_RESULTS/videos/ep23_E0_F0_F2_F3_camc.mp4
$B tools/overlay_groundtruth.py --dataset assemblyhands --clip 9012-c07c_005770 --methods E0,F0,F2,F3
```

## Layout

`tools/` code (named by what it does); `data/` inputs, model outputs, keypoints (gitignored); `results/<set>/` scores
and tables (`REPORT_TABLES.md`), videos; `results/egofound3r/` report and figures.
