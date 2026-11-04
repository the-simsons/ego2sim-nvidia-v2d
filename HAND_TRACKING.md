# Task: build a hand-tracking benchmark for our V2D Track 3 pipeline, then run experiments against it

You are working on a GPU machine for a CMU team in the NVIDIA Video-to-Data (V2D) Challenge, Track 3 (egocentric video → robot policy). You have no memory of earlier work, so everything you need is below. Read it all before acting. 

## Context

- Track 3 data: head-worn **stereo** egocentric video (cam_a / cam_b), Vicon object poses, scanned object meshes. There is **no hand ground truth** in Track 3.
- Our pipeline's hand branch (baseline): WiLoR hand detections → SAM2 hand masks → HaMeR/WiLoR MANO per frame → depth alignment against stereo/MoGe depth → glitch cleanup → gap filling during occluded grasps. There is a `--hand_tracking` option; one value is `hawor`.
- We **cannot run the full V2D pipeline here yet**. So the benchmark must work on the hand stage alone: video + calibration (+ masks/depth if cached) in, a hand-tracks file out, metrics computed on that file.
- **We do not care about hands that leave the frame.** A hand out of view is almost never carrying an object. Score only frames where the hand is visible; infilled/extrapolated frames are excluded from accuracy metrics and only counted in a separate coverage line.

Fill in or discover (grep / find, don't guess):
- Pipeline repo: `<REPO_PATH>` (look for the `--hand_tracking` flag, `ego_hands`, `clean_hand_tracks`, WiLoR/HaMeR wrappers)
- Track 3 data: `<DATA_PATH>` (expect `track_3/public/{data,videos,mesh,urdf,meta}`; read its README)
- MANO model files: `MANO_RIGHT.pkl`, `MANO_LEFT.pkl` (licensed; find them on disk, never download from unofficial mirrors)

## Machine rules

- 2× RTX 4090. **GPU 1 is in use by someone else (≈21 GB, 100% util). Use GPU 0 only:** `export CUDA_VISIBLE_DEVICES=0`. Run `nvidia-smi` before any heavy job and back off if GPU 0 is busy.
- Make a separate conda/venv per method (baseline, HaWoR, EgoFound3R). Don't touch shared/system envs or other people's files.
- Ask me before: any download > 20 GB, any `sudo`, deleting anything you didn't create, or editing upstream `video_to_data` files.
- Put all new code in `handbench/` inside our repo on a new git branch `handbench`, commit often with clear messages. It must also run on a Mac CPU for the metric/visualization parts (no CUDA imports at module top level in metric code).
- Record for every run: git commit, method + version/commit, checkpoint name, args, GPU, wall time, fps. Fix random seeds.

## Phase 1: the benchmark (do this first, fully, before any experiment)

### 1.1 Common output format (every method writes this)

`results/<method>/<episode>/hands.npz` (or parquet), per frame × hand (`L`/`R`):
- `frame_idx`, `t` (seconds, from the video's real fps; don't assume 30), `side`, `valid` (bool), `source` (`observed` | `infilled`), `conf`
- MANO: `global_orient` (3), `hand_pose` (45), `betas` (10), `transl` (3), all in the **undistorted cam_a OpenCV camera frame**, metres
- `joints_cam` (21×3), `verts_cam` (778×3, optional if large: regenerate from MANO)
- `joints_world` (21×3) if the method gives a world frame, plus `T_world_cam` per frame
- An adapter per method converts its native output into this. Write `handbench/format.py` with a loader + validator and test it.

### 1.2 Evaluation sets

The primary benchmark is **cam_b held out**: every method estimates hands from cam_a, and cam_b is used only for scoring. No external dataset is needed for the main table.

**A. V2D Track 3 dev set with cam_b as the held-out view (primary).** All public Track 3 episodes; list in `handbench/sets/v2d_dev.txt`. Build the cam_b references once and cache them under `handbench/refs/<episode>/`:
  1. **cam_b hand masks**: SAM2 on cam_b, prompted from cam_a detections transferred through the stereo calibration, then checked by eye on a contact sheet.
  2. **cam_b 2D keypoints**: run the 2D hand keypoint detector independently on cam_b (not reusing cam_a results).
  3. **Triangulated pseudo-GT**: run the same 2D detector on cam_a too, triangulate each joint where both views are confident, and keep only joints with reprojection error < 3 px in both views. This gives 3D joints in cam_a frame from images alone.

**B. Small hand-labelled V2D set (I will help label).** Write a tiny labelling tool (OpenCV or a static HTML page) for:
  1. grasp intervals per episode: object id, hand, start frame, end frame (contact onset → release);
  2. ~60 frames total with 21 2D keypoints for visible hands in **both** cam_a and cam_b (spread across episodes, biased to grasp frames). These check the pseudo-GT in A.3.
  Generate the frame list and the tool, then stop and tell me how long you estimate labelling takes. Don't fabricate labels.

Don't download any external dataset (HOT3D etc.) in this task; see "Future work".

### 1.3 First, check how sensitive cam_b is (do this before building anything else)

The stereo baseline limits what cam_b can detect. A depth error ΔZ at distance Z shifts the hand in cam_b by about `f · b · ΔZ / Z²` pixels.
- Read `f` (px) and the baseline `b` from the calibration, and the typical hand depth Z from the stereo depth inside hand masks.
- Report the cam_b pixel shift for ΔZ = 1, 2 and 5 cm, and the triangulation depth uncertainty for ±1 px keypoint noise.
- If a 2 cm depth error gives < 3 px in cam_b, cam_b can't separate methods on depth. Tell me, note it as a limitation in the report, and carry on (lean on lateral error and cam_b IoU).

Independence caveat: the baseline's depth alignment uses SGBM stereo depth, which already uses cam_b. Mask- and keypoint-based cam_b metrics are still informative (they come from cam_b appearance, not SGBM matching), but flag in the report which methods consumed stereo depth.

### 1.4 Visible-frame definition

A hand is "visible" in frame f if the SAM2 hand mask exists with ≥ 400 px in cam_a (and, for cam_b metrics, in cam_b). Only visible frames are scored.

### 1.5 Metrics

Primary (set A, cam_b held out), on visible frames:
- **Triangulated MPJPE** (mm): method's cam_a-frame joints vs triangulated pseudo-GT, no alignment. Also report the depth-only and lateral-only components.
- **cam_b keypoint error** (px, PCK@10px): method's 3D joints projected into cam_b vs cam_b detector keypoints.
- **cam_b IoU**: rendered MANO silhouette projected into cam_b vs cam_b hand mask.

Secondary (set A, cam_a), on visible frames:
- **Mask IoU** and **2D keypoint error** in cam_a (fit to the input view).
- **Stereo depth residual** (mm): median |rendered hand depth − stereo depth| over mask pixels; also the signed median (bias).
- **Temporal**: joint jitter (mean ‖accel‖), % frames with wrist speed > 2 m/s, bone-length std across the episode, betas std, handedness swaps per minute.
- **Grasp physics** (inside labelled grasp intervals from set B): fingertip-to-object-surface distance for contacting fingers, max penetration depth into the object mesh, and **hand pose in the object frame** std over the grasp (a rigid hold should be near constant). Object pose: use Vicon if Track 3 provides a camera→Vicon transform or a tracked headset; otherwise use our pipeline's object 6-DoF tracks if cached, and label these numbers "relative to estimated object pose". Check the data README to decide and write the decision in the report.
- **Coverage** (separate, not scored): % visible frames with a valid hand; % infilled frames.

Set B: 2D keypoint error in both views on the labelled frames (this also validates the cam_b detector and pseudo-GT); grasp-interval agreement (temporal IoU) for any method that predicts contact.
### 1.6 Reports and visuals

- `handbench/run_eval.py --method X --set v2d` (keep `--set` so other datasets can be added later) → `results/<method>/metrics.json` + per-episode CSV.
- `handbench/compare.py` → `results/REPORT.md`: one table, methods × metrics, baseline first, best in bold, plus Δ vs baseline.
- Overlay MP4 per episode: cam_a with mesh overlay, cam_b with projected mesh, and a small plot of depth residual over time. Side-by-side across methods for the same episode. Keep each video < 30 MB.
- Unit-test the metrics on a synthetic case: render a known MANO hand into both cameras with the real calibration; the GT must score ~0, and a copy shifted 2 cm in depth must give the cam_b pixel shift predicted in 1.3.

### 1.7 Freeze the baseline

Run the current baseline hand stage on set A. This is **E0**. Write the numbers to `REPORT.md` and commit them; every later experiment is compared to this frozen run. Then send me a short summary (sensitivity numbers from 1.3, E0 numbers, 3 worst episodes with why) before Phase 2. If numbers look broken (e.g. triangulated MPJPE > 100 mm), debug the harness first.

## Phase 2: experiments (one at a time, each is a new `results/<method>/` + a row in REPORT.md)

Promotion rule: a variant "wins" if it improves triangulated MPJPE **and** cam_b keypoint error or cam_b IoU, without regressing cam_a mask IoU or cam_a keypoint error by more than 5%. Report wins and losses honestly; negative results are useful.

- **E1, cheap fixes on the baseline** (do first, they're fast):
  a. one shared `betas` per hand per episode (median of confident frames, then re-fit pose with it fixed);
  b. depth alignment variants: current method vs robust (Huber) fit of MANO surface to stereo points inside the mask vs scale-only;
  c. temporal smoothing (One-Euro vs Savitzky-Golay vs small trajectory optimisation with a 2D reprojection term);
  d. handedness-swap repair.
- **E2, HaWoR** (`--hand_tracking hawor`, CVPR 2025): run (i) HaWoR end to end with its own SLAM + metric scale, and (ii) HaWoR camera-frame hands + our stereo depth alignment. Its motion infiller is irrelevant for us (out-of-frame), so mark infilled frames and exclude them. Record fps.
- **E3, EgoFound3R** (arXiv 2610.01210; https://github.com/MINT-SJTU/EgoFound3R): end-to-end world-space hands at metric scale shared with the scene, plus point-wise visibility/contact/distance. First check whether code and weights are actually released and the licence. If yes: run on cam_a, align its world frame to ours (Sim3 on the camera trajectory), and also evaluate its **contact predictions** against set C grasp intervals. If not released, write that down and skip.
- **E4, per-frame quality classifier ("second model")**: a cheap verifier that flags bad hand meshes per frame from cam_a-only features (cam_a mask IoU, depth residual, jitter, conf). Its training target is the cam_b error (triangulated MPJPE or cam_b keypoint error) on set A, so it learns to predict held-out-view error without seeing cam_b. Split by episode, and test whether dropping or replacing flagged frames improves the primary metrics. Optional: compare against a VLM judging overlays on a sample of frames. Report precision/recall for "triangulated error > 20 mm".
- **E5, ensemble/selection**: per frame, pick the method (E0/E1/E2/E3) with the best verifier score; then smooth. Only after E1–E4 exist.
- **E6, object-anchored grasp hold**: inside grasp intervals, estimate one hand-in-object-frame pose (robust mean over confident frames) and propagate it with the object pose. Evaluate with the grasp-physics metrics. Then the reverse direction, as a stretch: use the hand trajectory to fill object-pose gaps and report ADD against Vicon on the frames where the object tracker lost the object.
- **E7, grasp/contact extraction**: from the best hand tracks + object mesh, produce per-grasp records: object, hand, start/end time, wrist pose in object frame, which fingers touch, contact points on the object (mesh vertex ids + positions). Schema in `handbench/grasps_schema.md`. Validate start/end against set C.

## Deliverables at the end

1. `handbench/` code on branch `handbench`, with a README saying how to add a new method (adapter + one command).
2. `results/REPORT.md`: the comparison table, the promotion verdicts, failure-case notes with frame numbers, runtime/fps per method, and a recommended configuration for the pipeline.
3. Overlay videos for the 3 most informative episodes.
4. A short "what I'd try next" list.

## Future work (not for this task; don't start it)

**HOT3D as a ground-truth check.** cam_b only gives pseudo-GT, and the short stereo baseline may hide 1–2 cm depth errors. Later we may add a small HOT3D sanity set to confirm the cam_b metrics rank methods the same way true error does:
- 8 Aria clips from the HOT3D-Clips train split (150 frames each, ~92 MB per clip, so ~0.75 GB and ~1,200 frames), chosen for grasps, object occlusion and fast motion; clips must include MANO hand GT. Aria rather than Quest 3, because Aria is head-worn RGB like ours.
- Metrics: MPJPE-cam (no alignment), PA-MPJPE, and the rank correlation between our cam_b metrics and true MPJPE across methods. It's a check on the benchmark, not a headline number.
- The E4 quality classifier could also be trained against true error there.
- Needs the HOT3D licence click-through; the hand annotations are CC BY-NC-SA.

To make this cheap later: keep the harness dataset-agnostic (`--set`, a dataset adapter interface in `handbench/datasets/`), and put a stub `hot3d.py` adapter with a TODO, no download code. Mention in the final report whether the 1.3 sensitivity numbers suggest we need it.

Keep me updated after Phase 1 and after each experiment with a 3–5 line summary. If something blocks you (licence, missing data, broken install > 45 min), write it down, skip to the next item, and tell me.
