# Task: evaluate EgoFound3R for hand tracking against the V2D baseline, on Track 3 and on ground-truth datasets

You are working on a GPU machine for a CMU team in the NVIDIA Video-to-Data (V2D) Challenge, Track 3 (egocentric video → robot policy). You have no memory of earlier work: everything you need is below or in the repo. Read all of it before acting. The results will be published (non-commercial academic research), so every number must be reproducible and every caveat written down.

## Background (short)

- Our earlier study (repo branch `pbasq/handbench`, folder `handtracking/`, report `handtracking_summary`) built **handbench**, a hand-tracking benchmark for Track 3. Read that report and the handbench README first; reuse its code, output format (`hands.npz` per method/episode), adapters and metrics. Don't rebuild what exists.
- Main finding there: the baseline puts hands in the right place in the image but ~26 cm too far (MoGe depth is ~1.45× too large). Moving hands onto **triangulated keypoints** was the best correction on HOT3D (HaWoR 28.4 → 21.6 mm against true hands).
- Track 3 head rig: `cam_c` (left, mono 1280×800), `cam_a` (centre, colour 2028×1520, the only input camera), `cam_b` (right, mono). cam_a–cam_b and cam_a–cam_c are ~3.7 cm apart, cam_b–cam_c 7.5 cm. Video fps is 20.254 (MP4 headers say 20; don't trust them). Use the refined calibration from handbench, not the factory one (factory was 23–36 px off).
- **EgoFound3R** (arXiv 2610.01210, repo `MINT-SJTU/EgoFound3R`): one end-to-end model, monocular RGB video in, world-space hand geometry at metric scale out, plus camera/scene geometry and per-point visibility/contact/distance. Frozen VGGT-Ω backbone. Input 512×512. Outputs **joints and vertices (195 → densified to the 778-vertex MANO mesh), not MANO parameters.**

## Experiments (exactly these four, increasing scope)

| ID | Method | What it isolates |
|---|---|---|
| **E0** | Upstream baseline: WiLoR detection → SAM2 hand masks → HaMeR MANO per frame → shift to MoGe depth (upstream `align_hands`) | reference point |
| **F0** | EgoFound3R alone, on cam_a | effect of swapping the whole hand stage |
| **F2** | F0 + triangulated-keypoint shift: RTMPose 2D keypoints in the input camera + one extra camera, triangulated; in each frame with ≥ 4 well-placed triangulated joints (reprojection < 3 px), translate the whole hand (no rotation, no scale) so its joints best match them; carry the last shift across frames without enough joints (up to 30 frames); otherwise keep F0 | effect of multi-camera metric correction |
| **F3** | F2 + Savitzky–Golay smoothing of joint trajectories (same settings as handbench's E1best; report the window and order) | effect of temporal smoothing |

Notes:
- F2 uses only translation. Fitting rotation/scale to few noisy joints made results worse on HOT3D; don't add it.
- In F2, apply the shift to EgoFound3R's whole scene for that frame's hands only in camera frame; record the per-frame shift so it can be inspected. Mark carried-shift frames separately from frames with their own triangulation.
- For downstream compatibility, also fit MANO parameters to EgoFound3R's 778 vertices (same topology, should be near-exact; report the fit residual). Evaluate on the 21 joints.

## Test sets

1. **Track 3 handbench** (our domain; 20 public episodes; pseudo-ground truth).
   - **Leakage rule:** F2/F3 use triangulated keypoints, so they must not be judged by the same keypoints. For F2/F3 on Track 3, triangulate **only from cam_a + cam_b**, and keep **cam_c fully held out**.
   - Primary Track 3 metrics for all methods: **cam_c keypoint error (px) and PCK@10 px** (method's 3D joints projected into cam_c vs RTMPose detections in cam_c) and **cam_c IoU** (projected hand silhouette vs SAM2 hand mask in cam_c; build cam_c masks if handbench lacks them).
   - Also report the existing handbench 3D error against the triangulated reference, but mark F2/F3 rows as **circular** there (they used cam_b keypoints that also built the reference).
2. **AssemblyHands** (clean ground truth: egocentric, 3D keypoints ~4 mm accurate; **not** in EgoFound3R's training data).
   - Data is on Google Drive. First list the available files and sizes **without downloading them**, then choose the smallest validation/test subset with public 3D labels that still gives ≥ ~1,000 egocentric hand-frames across several sequences and subjects. It must fit the 3 GB download budget (see Setup). Drive often packs data into large zips: if no meaningful subset fits in 3 GB, stop and tell me the options and sizes instead of downloading. Use `gdown` for public links. If a file needs sign-in or hits a quota, **stop and give me the exact file names/links** so I can download them on my Mac and copy them over.
   - Input is an egocentric **monochrome** headset camera: feed it as 3 identical channels to every method (same for all). Check the calibration and synchronisation of the headset cameras: F2 needs a second camera seeing the same hands. If no usable second egocentric camera exists, run F2/F3 with whatever second view is valid and say so, or skip F2/F3 here and say so.
   - Map AssemblyHands' 21-keypoint convention to MANO's joints; report the mapping. Expect a fixed per-joint offset (wrist/fingertip definitions differ, ~8 mm in our earlier RTMPose-vs-MANO check); report metrics both raw and with a per-joint offset estimated on a held-out split.
3. **HOT3D** (Aria clips with motion-capture MANO; already on disk from the earlier study).
   - **EgoFound3R was trained on HOT3D Aria**, and our 8 clips came from HOT3D's train split, so F0–F3 here show *seen-domain* performance only. Label every HOT3D table "seen in EgoFound3R training". Use the same 7 clips / 1,433 hand-frames as the earlier report so numbers line up with its HaWoR rows.
   - Undistort Aria fisheye to pinhole as the earlier HOT3D pipeline did.

Training-data overlap to state in the report: EgoFound3R trained on H2O, HOT3D-Aria, OakInk-v2, TACO, HOI4D, ARCTIC, STERA-10M. Check and state whether WiLoR/HaMeR training data includes AssemblyHands or HOT3D.

## Running E0 off Track 3

The upstream pipeline is Docker-based and couldn't run on HOT3D before. Build a **lightweight standalone E0** (no Docker) that reuses the upstream library code: WiLoR detection, SAM2 hand masks, HaMeR per frame, MoGe-2 depth and the upstream `align_hands` rule (median depth difference inside rendered-hand ∩ SAM2 mask, shift along the centroid ray, per-track `hand_scale`). **Validate it first**: on Track 3, standalone E0 must reproduce the cached upstream E0 handbench numbers within 5% (3D error, depth bias, cam_b PCK). If it doesn't, debug before using it elsewhere, and report what differs.

## Metrics (report all, per test set, per method)

Only on frames where the hand is visible in the input camera; never score infilled/extrapolated frames as observed.

- **3D joint error, unaligned, camera frame** (mm, mean and median), split into **depth** (along the viewing ray), **lateral**, and **pose** (after removing translation). Ground-truth sets and Track 3 (with the circular flag).
- **PA-MPJPE** (mm): finger articulation only.
- **Depth bias** (mm, signed mean; positive = too far).
- **Held-out camera keypoints**: px error and PCK@10 px (Track 3: cam_c).
- **Held-out camera IoU** (Track 3: cam_c).
- **Jitter** (mean joint acceleration, m/s²) and **bone-length std** (mm).
- **Coverage**: % of visible hand-frames with a hand from the method; for F2, % frames with own triangulation / carried shift / no correction.
- **Compute** (for a trade-study table): peak VRAM per model, measured both with `torch.cuda.max_memory_allocated()` (reset per model) and an `nvidia-smi --query-gpu=memory.used --format=csv -lms 200` background sampler; wall time per episode and frames/s. Per method: peak = max over its models, time = sum. Include CPU time for RTMPose, triangulation and smoothing.
- On the 3D-error tables, compare methods also on the **common frames** all of them cover.
- Note the noise floor: Track 3's pseudo-GT is ~20 mm off true joints, so differences under ~5 mm there aren't meaningful.

## Setup and machine rules

- Paths (fill in or discover; don't guess):
  - repo: `<REPO_PATH>`, branch `pbasq/handbench`; put new work on a new branch `egofound3r-eval`
  - Track 3 data: `<TRACK3_DATA>`; cached E0 outputs: `<E0_OUTPUTS>`
  - HOT3D clips: `<HOT3D_PATH>`; MANO files: `<MANO_PATH>`
  - **EgoFound3R checkpoint and VGGT-Ω weights are pre-placed at `<WEIGHTS_DIR>`** (`EgoFound3R.pt`, `vggt_omega_1b_512.pt`). Do **not** log in to Hugging Face or try to download gated weights. If a file is missing or wrong, stop and tell me.
- GPU: run `nvidia-smi` first and use a free 4090 (`CUDA_VISIBLE_DEVICES`); GPU 1 has been used by someone else before. Back off if the GPU you picked becomes busy.
- One venv/conda env per method (EgoFound3R, standalone E0, RTMPose). Don't touch shared envs or other people's files.
- **Download budget: 3 GB in total** for everything you download in this task (dataset files, extra model checkpoints such as RTMPose/WiLoR/HaMeR/MoGe if not already on disk). It excludes pip/conda packages and the weights I place in `<WEIGHTS_DIR>`. Check what's already on disk before downloading anything. Keep a running log of downloads and sizes. Going over 3 GB needs a real reason: stop, explain what and why, and wait for my OK.
- Also ask me before: `sudo`, deleting anything you didn't create, or editing upstream `video_to_data` files.
- Record for every run: git commit, model checkpoint names and hashes, args, GPU, seeds, wall time. Fix seeds.
- Commit often with clear messages.

## Check-ins (stop and send me a short summary at each)

1. After setup: EgoFound3R runs on one Track 3 episode (overlay video), and the AssemblyHands file list with your proposed subset and size.
2. After standalone E0 is validated against cached E0 on Track 3.
3. After F0 on all three test sets.
4. Final: F2, F3, report.

If something blocks you (licence, missing data, install broken > 45 min), write it down, skip to the next item, and tell me.

## Deliverables

1. Code on branch `egofound3r-eval`, reusing handbench; README with exact commands per method and test set.
2. `results/egofound3r/REPORT.md` (and a PDF), written for a reader with no context:
   - one results table per test set (E0, F0, F2, F3), best in bold, Δ vs E0;
   - an ablation reading: what F0, F2, F3 each add, on which metric;
   - the compute table;
   - the leakage/overlap statement and the Track 3 circularity note;
   - 3 failure cases with frame numbers and pictures;
   - Future work (below), updated with what you learned.
3. Overlay videos for 3 informative Track 3 episodes and 2 AssemblyHands sequences (input camera + held-out camera, all four methods), each < 30 MB.

## Future work (not in this task; list in the report, don't run)

- **E1best** (E0 hands + SGBM stereo rescale + one size per hand + smoothing) on the ground-truth sets.
- **F1**: EgoFound3R + one stereo depth scale per video applied to its whole scene (hands, camera, points), gated on stereo coverage and stability.
- **HaWoR** (+ stereo / + triangulated shift) on the same sets.
- **Contact**: score EgoFound3R's per-point contact against labelled grasp intervals on Track 3.
- More ground-truth sets: DexYCB (third-person; in HaMeR's training data), EgoDex test split (on-device labels, large scale), ARCTIC (in EgoFound3R's training data).
- Feed EgoFound3R hands (with fitted MANO) through the simulation replay and compare with the team's current hands.
- Use EgoFound3R's scene for object reconstruction so hands and objects share one metric scale.