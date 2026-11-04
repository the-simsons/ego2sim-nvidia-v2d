# ego2sim-nvidia-v2d

**Team The Simsons**: Ego2Sim, our entry to Track 3 of the NVIDIA Video-to-Data (V2D) Challenge 2026: egocentric
human video → object meshes, 6-DoF object tracks and hand motion → floating Sharpa Wave hand references → per-episode
FlashSAC policies, scored on the simulated object trajectories they produce.

This repo holds **only our code**. It runs against [nvidia-isaac/video_to_data](https://github.com/nvidia-isaac/video_to_data)
at the pinned evaluator commit `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`; upstream files are unchanged, and the
official evaluator `robotic_grounding/flash_chord/scripts/evaluate_policy.py` is used as is. No checkpoints, data or
MANO model files are included (MANO's licence forbids redistribution). Paths assume a `~/v2d` working layout.

## Layout

| Path | What |
|---|---|
| `scripts/` | Pipeline drivers and batch jobs (reconstruction per episode, evaluation-set rebuilds, reference cleaning, validation, demo videos) |
| `tools/` | Pipeline stages and diagnostics (stereo depth, hands, VO, tracking checks, mesh building and scoring, reference cleaning, media) |
| `agentstar_harness/` | Our additions to the AgentSTAR "Ego2Sim shape-only" mesh harness (a Claude agent modelling each object in Blender from the video) |
| `policy_s0/` | Per-episode S0 policy training, evaluation and submission packing (FlashSAC in FlashCHORD); snapshot of the final submission code (see below) |
| `policy_gm/` | Experimental decoupled grasp (G) + movement (M) policies built on the S0 code: shared behaviour-cloned grasp policy, per-episode movement policy (see `policy_gm/README.md`) |

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtual environment and install all dependencies (from the repo root):

```bash
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

Alternatively, you can run files without activating the environment with:

```sh
uv run python tools/ego_vo.py ...
```

### Using `uv`

- `.python-version` pins the Python version (3.11). uv downloads that interpreter if it isn't already installed.
- Dependencies:
   - Add a dependency with `uv add <package>` (e.g. `uv add open3d`) 
   - Remove a dependency with `uv remove <package>` 
   - Both commands update `pyproject.toml`, `uv.lock` and `.venv` together
   - Dependencies are listed in `pyproject.toml`
   - `torch` (2.14.1) is pulled from the PyTorch cu126 index, configured under `[tool.uv.sources]`; the CUDA 12.6
     build needs an NVIDIA driver for CUDA 12.6 or newer (`nvidia-smi` shows the supported CUDA version)
- `uv.lock` records the exact version of every dependency so every machine gets the same environment - commit it when changed
   - To upgrade a dependency: `uv lock --upgrade-package <package>`, then `uv sync`

### Extra Dependencies

`uv sync` installs the packages used by this repo's own `tools/` and `scripts/`. Some tools run in other
environments instead (their imports say so): FlashCHORD and Warp for `collision_hulls.py`, `settle_spawn_v2.py`,
`render_recon_gl.py` and `read_wandb_offline.py` (FlashCHORD's venv); FoundationStereo's code for
`ego_stereo_depth.py --method fs`; `v2d` (the submodule below) for `ego_hands.py` and `build_ego_bundle.py`;
`v2dlb` (submission kit) for the CD-O tools. The full pipeline also needs:
- the [video_to_data](https://github.com/nvidia-isaac/video_to_data) repo at the pinned commit (provides the `v2d`
  modules imported by a few tools). It is included as a git submodule in `third_party/video_to_data`
  - Use `git submodule update --init` to fetch (or clone this repo with `--recurse-submodules`)
  - The scripts expect it at `~/v2d/vtd_pinned`: `ln -s "$PWD/third_party/video_to_data" ~/v2d/vtd_pinned`
- separate environments for the models (SAM2, MoGe, FoundationPose, WiLoR, ...). Upstream runs each in its own Docker
  image (see upstream's [reconstruction modules README](https://github.com/nvidia-isaac/video_to_data/blob/8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef/reconstruction/README.md)
  and [ego reconstruction setup guide](https://github.com/nvidia-isaac/video_to_data/blob/8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef/reconstruction/docs/ego_e2e_setup.md)).
  If you have Docker with the NVIDIA Container Toolkit, you can follow those instead. Otherwise
  `scripts/setup_recon_envs.sh` rebuilds the images as uv venvs under `~/v2d/envs/<name>/.venv`:
  ```bash
  scripts/setup_recon_envs.sh host v2d_moge v2d_sam2 v2d_wilor v2d_hamer v2d_foundation_pose v2d_task_library_loader rg_retarget
  ```
  - The script sources `~/v2d/env.sh` (not in this repo; create it, e.g. with your PATH/CUDA exports, or an empty file)
  - It expects the upstream repo at `~/v2d/vtd_pinned` (see the symlink above)
  - `v2d_foundation_pose` compiles CUDA extensions and needs the CUDA 12.4 toolkit at `~/v2d/cuda/12.4`
    (e.g. `micromamba create -p ~/v2d/cuda/12.4 -c nvidia/label/cuda-12.4.1 cuda-toolkit`)
- the MANO hand model files. These are released by the Max Planck Institute for Intelligent Systems under the
  [MANO licence](https://mano.is.tue.mpg.de/license.html) (non-commercial scientific research only, no
  redistribution), so they are not in this repo. To install them:
  1. Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) and accept the licence
  2. From Downloads, get "Models & Code" (`mano_v1_2.zip`)
  3. Unpack it so the tools find `~/v2d/mano/models/MANO_RIGHT.pkl` and `MANO_LEFT.pkl`:
     ```bash
     mkdir -p ~/v2d/mano && unzip mano_v1_2.zip -d /tmp/mano && cp -r /tmp/mano/mano_v1_2/models ~/v2d/mano/
     ```

### Submission kit

Scoring against public mocap (`tools/score_public_episode.py`, `tools/local_cd_o.py`) and packing the submission
(`scripts/pack_eval.sh`) need the challenge submission kit: the official packer, scoring code, eval object list, and
`v2dlb` for mesh decimation. It is distributed to challenge participants:
1. Download it from the challenge page and unpack to `~/v2d/submission_kit/v2d_submission_kit`
2. Create its env at `~/v2d/envs/kit/.venv` (the scripts call that Python) and install the kit's requirements into it

## Reconstruction (per evaluation episode)

Driver: `scripts/ego2sim_public_episode.sh EP OBJ1[,OBJ2] GPU` with `SPLIT=evaluation TAG=_v2 MESH_SRC=meshes_final
DEPTH=stereo_fs FP_KW=n_particles=16` (all 20 episodes: `scripts/eval_rebuild_v2.sh`).

1. **Stereo depth**: undistorted cam_a + SGBM (`tools/ego_stereo_depth.py`); FoundationStereo depth for object
   tracking (`--method fs`).
2. **Segmentation**: SAM 2 masks. Object prompts `tools/obj_prompts.py`; hand prompts from WiLoR
   (`tools/auto_hand_prompts.py`), hand-placed where the automatic ones failed (`scripts/ep20_hand_prompts.json`,
   `scripts/ep25_hand_prompts.json`; checked with `tools/hand_sheet.py`).
3. **Depth fusion and camera motion**: MoGe fused with stereo (`tools/fuse_depth.py`), stereo VO (`tools/ego_vo.py`).
4. **Hands**: WiLoR + depth alignment (`tools/ego_hands.py`), repair (`tools/clean_hand_tracks.py`), occluded-grasp
   gaps carried with the object (`tools/fill_hand_gaps_object.py`).
5. **Object meshes**: AgentSTAR harness runs (`agentstar_harness/`, `tools/agentstar_mesh_export.py`), depth fusion
   (`tools/depth_fusion.py`), size checks (`tools/size_scale_check.py`). Best source per object:
   `tools/build_mesh_final.py` → `meshes_final` (tracking and simulation); `tools/build_mesh_pack.py` → `meshes_pack`
   (union-cleaned interiors for packing, same body frame). Local CD-O scoreboard with the kit's metric code:
   `tools/cdo_board.py`.
6. **6-DoF object tracking**: FoundationPose on FoundationStereo depth, 16 pose hypotheses; frame-0 static start
   (`tools/fp_static_start.py`), because the scorer aligns each scene on frame 0.
7. **Simulation reference**: bundle (`tools/build_ego_bundle.py`) → task-library loader → retarget to floating Sharpa
   → table support (`scripts/ego2sim_episode.sh`, `tools/make_table_support.py`) → multi-object reference
   (`tools/merge_sharpa_objects.py`); packaged with `tools/build_share_package.py`.

## Reference cleaning (before training)

`scripts/clean_v2_package.sh` (all episodes) / `scripts/clean_v2_settle.sh`:

1. **Stage 1** `tools/clean_refs_v2.py`: cup spin about the symmetry axis removed, look-alike flips unwrapped
   (checked against the masks with `tools/clean_to_cam.py` + `tools/fp_track_diag.py`), spikes re-interpolated,
   objects lifted onto the per-episode table.
2. **Stage 2**: frame-0 physics settle (`tools/settle_spawn_v2.py`) applied with blend-out. Static objects keep the
   settled pose for the whole clip; settles that tip an object over (> 30°) are not applied.
3. **Stage 3** `tools/depenetrate_refs.py`: objects whose collision hulls (the simulator's CoACD hulls,
   `tools/collision_hulls.py`, `tools/collision_check.py`) overlap are pushed apart by the shortest clearing move.
4. **Nested cups** `tools/nest_fix.py`: merged cup tracks rebuilt as one rigid stack at the mesh-derived nesting depth.

Checks: `tools/ref_hygiene.py`, `tools/cup_spin.py`, `tools/clean_diff.py`, `tools/branch_depth.py` (stereo depth vs
look-alike flips), replay videos `scripts/sim_video.sh`. `tools/flip_object_ref.py` builds a reference variant with
one object turned 180° about an axis.

## Validation against public motion capture

`scripts/public_chain_validate.sh` rebuilds the public episodes with the same chain and scores each reference alone
against mocap with the kit's own metric code (`tools/ref_parquet_as_rollout.py`, `tools/score_public_episode.py`).
Every pipeline change above was accepted or rejected this way. Diagnostics: `tools/reg_vs_track.py`,
`tools/axis_tilt.py`, `tools/oracle_reg_rollout.py`, `tools/frame0_variant.py`.

## Policy training and submission (`policy_s0/`)

Per-episode S0 policies (FlashSAC, 96.8M environment steps, pickup-reset recipe), every checkpoint evaluated on 64
worlds, best checkpoint per episode across reference versions, official evaluator, packed with the kit's
`tools/pack_submission.py`. `policy_s0/` is a snapshot of the training/submission repo at its final commit
`4dec20d` ("Final Track 3 submission: ep36 spoon face-up reference, ep19 contact-reward variant, ep8 v1 policy").

## Licence

Apache 2.0 (`LICENSE`). Third-party models and code (video_to_data, SAM 2, MoGe, WiLoR, HaMeR, FoundationPose,
FoundationStereo, MANO) keep their own licences and are not redistributed here.
