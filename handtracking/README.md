# handtracking: hand tracking for the Ego2Sim pipeline, and the benchmark that measures it

Two things live here: a better pipeline hand stage (multi-view hand tracking) and **handbench**, which scores hand
tracks against the held-out cameras of the ego rig.

- Best approach, results and limits: [BEST_APPROACH.md](BEST_APPROACH.md)
- Code architecture: [ARCHITECTURE.md](ARCHITECTURE.md) (folder) and [handbench/ARCHITECTURE.md](handbench/ARCHITECTURE.md) (modules)
- Full results: [results/REPORT.md](results/REPORT.md), [report/handtracking_summary.pdf](report/handtracking_summary.pdf); task spec: [../HAND_TRACKING.md](../HAND_TRACKING.md)

## Usage: multi-view hand tracking in the Ego2Sim pipeline

**One switch:** `HAND_TRACKER=multiview scripts/ego2sim_public_episode.sh EP OBJS GPU`.
Without it (default `HAND_TRACKER=wilor`) the pipeline is unchanged.

It replaces only the hand stage, `tools/ego_hands.py`, and writes the same folder,
`$E/hands/hands_by_side/{left,right}/<frame>.json`, with the same record fields (`mano`, `camera.cam_t`,
`camera.intrinsics`, `hand_scale`, `interpolated`) in the same camera (undistorted cam_a, factory K). Everything after it
runs unchanged:

```
SAM2 hand masks ─► [ego_hands.py | multiview_hands] ─► clean_hand_tracks ─► fill_hand_gaps_object ─► build_ego_bundle ─► retarget ─► sim
                      hands_by_side/{left,right}           (unchanged)        (object-aware fill)      (unchanged)
```

Each record also says how it was corrected (`"correction": keypoints | carried | stereo | none | fallback | filled`);
what the steps do is in [BEST_APPROACH.md](BEST_APPROACH.md). `fallback` frames carry a per-frame HaMeR hand: used where
HaWoR's hand is implausible in cam_a (outside its SAM2 mask or far from the cam_a keypoints, e.g. HaWoR on another
person's hand) or missing, with HaMeR's depth corrected like HaWoR's (keypoints, else stereo). `--no-hamer-fallback`
turns it off; it is also off, with a message, when no HaMeR environment is found (see below).

| Inputs (all already produced by the pipeline) | Path |
|---|---|
| raw cam_a, cam_c videos | `$DATA/videos/chunk-000/observation.images.ego_cam_{a,c}/episode_<EP>.mp4` |
| stereo dir | `$S` (`intrinsics_undist.json`, `stereo_depth/`) from `tools/ego_stereo_depth.py` |
| hand masks | `$E/hand_masks/<track>/<frame>.png` (SAM2) |
| rig calibration | `handtracking/refs/calib/rig.json` (in git; one ego rig for all episodes) |
| **Outputs** | |
| hand tracks | `$E/hands/hands_by_side/{left,right}/<frame>.json`, `$E/hands/hands_by_side/multiview_hands_info.json` |
| intermediate | `$E/hands/multiview_work/` (HaWoR export, keypoints, HaMeR records; reused when present) |

Standalone, without the episode script:

```bash
cd handtracking
.venv-bench/bin/python -m handbench.pipeline.multiview_hands --video-dir $DATA/videos/chunk-000 --episode $EP \
    --stereo-dir $S --hand-masks $E/hand_masks --out $E/hands/hands_by_side --gpu 0
```

Cost: about 4 min of GPU (HaWoR) + 2 min of GPU (HaMeR fallback, ~2 GB) + 3–5 min of CPU (keypoints) per 300-frame
episode.

HaMeR fallback environment: an env that imports video_to_data's `v2d.hamer.lib` (e.g. the pipeline's
`~/v2d/envs/v2d_hamer`, from `scripts/setup_recon_envs.sh ... v2d_hamer`) and HaMeR weights (a folder with `_DATA/`).
Found automatically in that order: `--hamer-python` / `--hamer-weights`, `$HAMER_PYTHON` / `$HAMER_WEIGHTS`,
`~/v2d/envs/v2d_hamer/.venv/bin/python` / `~/v2d/vtd_pinned/reconstruction/data/weights/hamer`, then
`handtracking/egofound/methods/` (the standalone E0 env, `egofound/tools/setup_e0_env.sh`).

To run and score the stage off the pipeline machine, `python -m handbench.pipeline.inputs_from_refs --episode EP
--out DIR` rebuilds its inputs (stereo depth, SAM2 hand masks) from handbench data.

### One-time setup (in `handtracking/`)

```bash
# CPU env: RTMPose, triangulation, correction (also runs the benchmark; add matplotlib pytest mediapipe for all of it)
uv venv --python 3.11 .venv-bench
uv pip install --python .venv-bench/bin/python numpy scipy opencv-python-headless pandas pyarrow trimesh rtmlib onnxruntime
# MANO (licensed, never committed): MANO_RIGHT.pkl -> assets/mano_right.npz (needs an env with chumpy, e.g. the repo's .venv)
../.venv/bin/python handbench/tools/convert_mano.py /path/to/MANO_RIGHT.pkl
# RTMPose hand model -> assets/models/rtmpose/.../end2end.onnx
mkdir -p assets/models/rtmpose && cd assets/models/rtmpose && \
  curl -sSfLO https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip && \
  unzip -q rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip && cd -
# HaWoR (GPU): code, venv (methods/.venv-hawor), DROID-SLAM extension and weights (~5 GB)
handbench/tools/setup_hawor.sh
CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.9 FORCE_CUDA=1 \
  uv pip install --python methods/.venv-hawor/bin/python --no-build-isolation -r methods/HaWoR/requirements.txt
```

HaWoR also needs `MANO_LEFT.pkl` in `methods/HaWoR/_DATA/data_left/mano_left/` (the setup script copies a mirrored
stand-in; replace it with the official file when available).

## Usage: the benchmark

```bash
cd handtracking
.venv-bench/bin/python -m pytest -q handbench/tests     # format + synthetic metric checks
```

Metric and visualisation code is NumPy + OpenCV only (no CUDA, no GL) and runs on a Mac CPU. Reference building needs the
videos and the cached upstream run (`data/`), not a GPU.

### Add a new method (adapter + one command)

1. Run the method on cam_a of each episode. It may use the cam_a video (`data/track3/videos/...ego_cam_a`), the rig
   (`refs/calib/rig.json`) and cached cam_a masks/depth. If it uses stereo depth (cam_b), set
   `consumed_stereo_depth: true` in meta.
2. Write `handbench/adapters/<method>.py` that fills `format.empty(n_frames, t)` per episode: for each frame and side
   (`L`/`R`): `valid`, `source` (`observed`, or `infilled` for anything not seen in the image), `conf`, MANO
   `global_orient/hand_pose/betas/transl/scale` and `joints_cam` (21x3, OpenPose order) **in the benchmark frame**. If
   your camera differs from the undistorted factory cam_a (e.g. self-calibrated intrinsics), map points by pixel + depth
   like `adapters/v2d_upstream.py`. `fit.refit_global` re-expresses a mesh as MANO params. Save with
   `format.save(results/<method>/<episode>/hands.npz, arrays, meta)`; meta needs
   `method, version, checkpoint, args, gpu, wall_time_s, fps, consumed_stereo_depth`.
3. Evaluate and compare:

```bash
python -m handbench.run_eval --method <method> --set v2d
python -m handbench.compare            # -> results/REPORT_TABLES.md (--episodes 2,11,... for a subset)
python -m handbench.overlay --episode 12 --methods E0,<method>
```

Conventions: left hands are the right MANO model mirrored in x (see `mano.py`); `t` comes from the parquet timestamps
(~20.25 fps, never assumed); rows with `source=infilled` are excluded from accuracy metrics. Never run `run_eval
--episodes` on a method whose full results matter: it overwrites that method's `metrics.json`.

### Rebuilding the references

```bash
python -m handbench.tools.calib_check        # rig extrinsics refined on SIFT matches -> refs/calib/rig.json
python -m handbench.tools.sensitivity        # cam_b depth sensitivity -> refs/sensitivity.json
taskset -c 0-7 python -m handbench.build_refs --jobs 6          # 2D keypoints + triangulated reference
taskset -c 0-7 python -m handbench.build_refs --stereo --jobs 6 # stereo depth crops
methods/.venv-sam2/bin/python -m handbench.build_masks_b        # cam_b SAM2 masks (GPU), then:
taskset -c 0-7 python -m handbench.build_refs --gate --jobs 10  # mask-consistency gate + re-triangulation
```

The machine is shared: pin CPU jobs with `taskset`; set `OMP_NUM_THREADS=1` for scoring workers.
