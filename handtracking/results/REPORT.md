# Hand-tracking benchmark: V2D Track 3 hand stage

**Checkpoint 2026-10-28 (evening).** Phase 1 complete except set B (needs hand labels). E0 re-frozen on the corrected
(gated) references; E1 and E2 scored; E3 skipped. Full tables (all methods, own frames + common frames, run metadata):
[REPORT_TABLES.md](REPORT_TABLES.md), regenerate with `python -m handbench.compare`.

## 1. Setup and decisions

* **Benchmark frame**: undistorted cam_a OpenCV camera (factory K, no distortion), metres. Rig: cam_c (left) –
  cam_a – cam_b (right), cam_b 3.74 cm from cam_a, cam_c 3.79 cm; b–c is the 7.5 cm stereo pair.
* **Calibration**: the factory extrinsics as stored are 23–36 px off on the published (180°-rotated) videos; conjugating
  them for the rotation fixes a–b (1.2 px) but not cam_c (8–11 px). Refined on pooled SIFT matches of 7 episodes
  (rotation + translation direction, factory baseline lengths): median Sampson error a–b 0.71, a–c 0.85, b–c 0.51 px
  (`refs/calib/rig.json`, `handbench/tools/calib_check.py`). Factory intrinsics + 14-coef distortion fit the videos as-is.
* **Time**: parquet timestamps, 20.254 fps (the mp4 container says 20.0; HaWoR's own frame extractor resamples to
  30 fps, so the runner writes frames itself).
* **2D keypoints per view** (cam_a, cam_b, cam_c, each from its own pixels): RTMPose-m hand (per-joint scores) on a
  perspective crop, two passes (crop on the SAM2 mask, then re-crop on the confident pass-1 keypoints). MediaPipe on the
  same crops as a cross-check (median disagreement 16 px cam_a at 2028 px wide, 4.3 px cam_b, 5.3 px cam_c).
* **cam_b hand masks**: SAM2.1 Hiera-L video propagation on cam_b (both GPUs, 6.5 min for 20 episodes), prompted from
  cam_a masks lifted with stereo depth + cam_b's own confident keypoints (cam_b sees more of a hand near cam_a's border:
  127° vs 108° FOV). Contact sheet: `refs/masks_b_contact_sheet.jpg`.
* **Mask-consistency gate (reference fix)**: with two hands close together the keypoint detector sometimes locked onto
  the other hand (e.g. 124 frames in ep40). A view's keypoints for a hand are kept only if >= 60 % of its confident
  keypoints fall inside that hand's mask in that view; otherwise that view's keypoints (and, for cam_b, its mask) are
  dropped for that hand-frame. Dropped: cam_a 6.0 %, cam_b 5.4 % of visible hand-frames; usable cam_b masks 93.1 %.
* **Triangulated pseudo-GT** (from the gated keypoints): per joint from every confident view, cam_a always included,
  reprojection < 3 px in every used view. `tri_abc` (a+b+c, primary) and `tri_ab` (a+b, as in the spec). Check vs SGBM
  stereo at the same pixels (ep12): median depth difference −1.5 mm, MAD 14 mm.
* **Visible frames**: cam_a SAM2 hand mask >= 400 px. SAM2 masks include the forearm, so IoU is computed inside a hand
  region (dilated hull of that view's own detector keypoints, method-independent); unclipped IoU kept as `iou_*_full`.
* **Grasp-physics object pose (decision)**: Track 3 gives no camera→Vicon transform or headset track (object poses are
  `world_T_object`, xy re-centred per episode, offset discarded). Grasp physics will use the pipeline's object 6-DoF
  tracks, rescaled to stereo depth, labelled "relative to estimated object pose". Not run yet (needs set B grasp labels).

## 2. How sensitive is cam_b? (1.3)

Hand depth (SGBM inside hand masks, 1006 hand-frames): median **0.40 m** (p10–p90 0.33–0.47 m). f_b = 567 px,
b(a–b) = 3.74 cm.

| depth error ΔZ | formula f·b·ΔZ/Z² at median Z | exact cam_b shift, median (p10–p90) | frames < 3 px |
|---|---|---|---|
| 1 cm | 1.4 px | 1.1 px (0.6–1.7) | 99.9 % |
| 2 cm | 2.7 px | **2.1 px** (1.2–3.3) | 85 % |
| 5 cm | 6.8 px | 4.9 px (2.9–7.6) | 14 % |

The exact shift is smaller than the pinhole formula because hands sit at the bottom of cam_b's 127° image, where the
lens compresses. Triangulation depth uncertainty per 1 px of keypoint noise: a–b **0.81 cm**, a–c 0.78 cm, b–c 0.50 cm,
a+b+c **0.50 cm** (lateral ~3–5 mm).

**Limitation: a 2 cm depth error gives < 3 px in cam_b, so cam_b keypoints/IoU cannot separate methods on depth at the
1–2 cm level.** Depth is judged by the triangulated MPJPE (a+b+c is 1.6x tighter than a+b); a true-GT check (HOT3D,
future work) is worth doing.

**Independence**: SGBM stereo (cam_b + cam_c) is consumed by every E1 variant and by E2-hawor-cam-stereo (flagged in
the tables). The pseudo-GT shares cam_b/c images with them but not the signal (keypoint detection vs block matching).

## 3. E0, the frozen baseline (re-frozen on the gated references)

E0 = cached upstream run (`video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer`,
`hamer_aligned_filled`): AnyCalib/GeoCalib self-calibration (fx ≈ 1000 vs factory 1150) → WiLoR detection → SAM2 hand
masks → HaMeR MANO → alignment to **MoGe** monocular depth (+ per-track hand_scale, clamped at 1.25) → short-gap
interpolation. Converted into the benchmark frame by preserving each point's pixel and depth (verified sub-pixel).
20 episodes, 9950 visible hand-frames, all scored (coverage 100 %, 1.2 % of valid rows infilled).

| metric | E0 |
|---|---|
| Triangulated MPJPE a+b+c | **263.4 mm** |
| ↳ depth / lateral component | 261.3 / 26.4 mm |
| ↳ signed depth bias | **+259.6 mm** (too far) |
| Triangulated MPJPE a+b | 258.8 mm |
| translation-removed MPJPE | 42.4 mm |
| cam_b keypoints | 25.3 px, PCK@10px 13.8 % |
| cam_c keypoints | 31.1 px |
| cam_b IoU (hand region) / full mask | 0.363 / 0.292 |
| cam_a IoU (hand region) / full mask | 0.284 / 0.215 |
| cam_a keypoints | 42.2 px |
| stereo depth residual / bias | 200.9 / +195.4 mm |
| jitter, wrist > 2 m/s, bone std, swaps/min | 13.4 m/s², 1.9 %, 1.05 mm, 0.19 |

MPJPE > 100 mm, so the harness was debugged before accepting it: in E0's own camera (no adapter) its silhouette IoU
against the same masks is the same; overlays show the meshes on the hands in cam_a; upstream MoGe depth inside hand
masks is **1.45x** the SGBM depth (ep12), matching E0's 1.50x depth ratio to the pseudo-GT. **E0 is image-aligned but
~1.5x too far and correspondingly oversized**: inherited from MoGe + self-calibration, not a hand-pose problem. Across
episodes the bias correlates with the upstream MoGe scale (Spearman 0.51), GeoCalib focal (0.58) and hand_scale (0.59).

**3 worst episodes (E0, triangulated MPJPE):**

| episode | task | MPJPE | why |
|---|---|---|---|
| ep39 | Dustpan_Solo | 383 mm | largest depth bias (+378 mm) **and** largest lateral error (48.5 mm); hands leave the view often (13 % infilled) |
| ep24 | Planter_Stand | 316 mm | MoGe scale 1.61 (highest of all episodes) → +314 mm bias |
| ep23 | Planter_Stand | 316 mm | +313 mm bias; GeoCalib fx 1061 (outlier); lateral fine (21 mm) |

## 4. Phase 2 results

Promotion rule: improve triangulated MPJPE **and** cam_b keypoint error or cam_b IoU, without regressing cam_a mask
IoU or cam_a keypoint error by more than 5 %. All on the gated references; each method on its own scored frames
(common-frame table in REPORT_TABLES.md). "Stereo" = consumes SGBM depth from cam_b+cam_c.

| method | stereo | Tri MPJPE | depth bias | lateral | cam_b kp (PCK@10) | cam_b IoU | cam_a IoU | cam_a kp | jitter | bone std | cov | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E0** | no | 263.4 | +260 | 26.4 | 25.3 (13.8 %) | 0.363 | 0.284 | 42.2 | 13.4 | 1.05 | 100 % | baseline |
| E0-refined (upstream gsplat refinement) | no | 261.8 | +255 | 33.0 | worse | – | 0.267 | 52.0* | 12.6 | 1.93 | 100 % | **loses** |
| E1b-dz (upstream rule, SGBM instead of MoGe) | yes | 59.6 | +22 | 24.9 | ~E0 | – | 0.224 | 55.0* | 8.4 | 0.93 | 100 % | **fails** (inflated mesh: cam_a −21 % IoU) |
| E1b-scale (rescale about camera to SGBM) | yes | 49.7 | +6.7 | 17.5 | 21.2 (23.4 %) | 0.410 | 0.283 | 42.0 | 7.8 | 3.69 | 100 % | **wins** |
| E1b-huber (robust surface fit) | yes | ~49.5 | +6 | 17.5 | ~E1b-scale | – | 0.276 | 43.2* | 8.7 | 10.1 | 100 % | wins, ties scale; unstable bones |
| E1a-betas (shared betas) | yes | = E1b-scale | | | | | | | | | | **no-op** (E0 already uses fixed betas per track) |
| **E1a-size-dz** (one hand size per side + per-frame depth) | yes | **49.6** | +6.8 | 17.4 | 21.1 (23.7 %) | 0.413 | 0.286 | 41.6 | 8.5 | **0.11** | 100 % | **wins** (= scale, no size flicker) |
| **E1c-savgol** (Savitzky-Golay, w7, on E1b-scale) | yes | **49.6** | +6.7 | 17.4 | 21.1 (23.5 %) | 0.410 | 0.283 | 41.9 | **2.6** | 3.56 | 100 % | **wins** (= scale, 3x smoother) |
| E1c-oneeuro | yes | 51.2 | +5.7 | 20.2 | 30.2 (15.3 %) | 0.356 | 0.253 | 54.3 | 2.3 | 3.42 | 100 % | **loses** (lag on fast motion) |
| E1c-traj (accel prior + cam_a 2D term) | yes | 55.4 | −1.4 | **12.2** | **17.9 (32.4 %)** | **0.439** | 0.304 | 30.7† | 11.5 | 3.69 | 100 % | **fails** (MPJPE worse); best 2D |
| E1d-swaps | yes | = E1b-scale | | | | | | | | | | **no-op** (0 swaps found) |
| E2-hawor (end to end, own SLAM) | no | 121.1 | +109 | 21.6 | 28.5 (21.1 %) | 0.415 | 0.296 | 44.7 | 9.9 | 1.43 | 93.7 % | wins vs E0 on MPJPE + cam_b IoU; cam_a kp +6 % on own frames, −10 % on common frames |
| E2-hawor-cam | no | 121.2 | +109 | 21.6 | 28.6 (21.1 %) | 0.415 | 0.295 | 44.9 | 9.8 | 1.43 | 93.7 % | = e2e on observed frames |
| E2-hawor-cam-stereo (HaWoR + E1b-scale recipe) | yes | 53.2 | +15.8 | 19.1 | 26.5 (24.8 %) | 0.438 | 0.297 | 44.9 | 4.1 | 2.05 | 93.7 % | **wins vs E0** on common frames; trade-off vs E1b-scale |

\* common-frame value (own-frame value not re-tabulated here). † E1c-traj fits to the cam_a detector keypoints that
are also the cam_a reference: its cam_a kp is not independent.

**Common hand-frames (5316, every method scored)**: E1b-scale 49.4 mm / cam_b kp 19.9 px / PCK 24.2 % / cam_a kp 42.9;
E2-hawor-cam-stereo 53.2 mm / **19.4 px / 28.2 %** / **39.0**; pose error (translation removed) 36.6 vs **34.9** mm.
HaWoR has the better pose/2D fit and is 2x smoother; E1b-scale has the better depth (+6.5 vs +15.8 mm bias).

**HaWoR notes**: commit 66c7d41; 20 episodes on 2x RTX 4090 in ~35 min total, **1.0–1.6 fps** per GPU (incl. DROID-SLAM +
Metric3D). End to end and camera-frame are identical on observed frames: its SLAM metric scale (Metric3D) does not
change the hands' camera-frame depth, which comes from its own monocular estimate (+109 mm bias, vs MoGe's +260 mm).
7.7 % of its valid rows are infiller output (excluded). Coverage 93.7 %. MANO_LEFT = mirrored-right stand-in. Adapter
refit error ~5.5 mm max-vertex (median).

**Where the remaining error is** (E1b-scale, per hand-frame): hand position error 31.4 mm (23.5 mm along the viewing
ray), pose/shape error with position removed 36.6 mm. HaWoR: 36.1 (27.6) / 34.9 mm. Two different hand models land
within 2 mm on pose, suggesting pose is near the measurement floor (RTMPose vs MANO joint conventions + pseudo-GT
noise); position (stereo depth noise) is the more actionable part.

**Recommended configuration (measured): E1best-sgbm** = E0 meshes (HaMeR) → one hand size per side + per-frame depth
from SGBM stereo (E1a-size-dz) → Savitzky-Golay smoothing: **49.5 mm** Tri MPJPE (+6.7 mm bias), cam_b 21.0 px /
PCK 23.8 %, cam_a IoU 0.286, jitter **2.7 m/s²** (E0 13.4), bone std 0.99 mm, coverage 100 %. On common frames 49.3 mm.

**FoundationStereo (2026-10-29)**: NVIDIA deployable small ONNX (local copy), onnxruntime CUDA, both GPUs (~40 min).
Depth check vs the triangulated reference at joint pixels (`refs/stereo_check.json`): SGBM median |err| 19.0 mm,
51 % within 2 cm, 60 % of mask pixels with depth; FoundationStereo 22.2 mm, 47 %, 72 %. Methods: E1b-scale-fs 50.0 mm
(SGBM 49.7), E1best-fs 49.8 (49.5), HaWoR+stereo-fs 52.8 (53.2); bias slightly lower (+3.7 vs +6.7 mm) but more jitter.
**Not promoted.** Swapping the depth source leaves the ~43 mm depth component unchanged, so the remaining depth error is
probably not stereo noise (reference noise / joint-convention floor).

**Simulation (teammate, `v2d_challenge_data/outputs/handtracking_sim_compare/`)**: E1best-sgbm vs the team's current hands
(`objfill_v2s`), FlashCHORD physics replay with GT objects, 9 episodes, medians: fingertip gap 1.05 vs 0.90 cm, contact
while the object moves 93 % vs 92 %, robot-hand contact 73 % vs 84 %. On par; the team's hands use object-aware gap
filling, ours are ~15 % too big in sim (the loader ignores the per-side size factor).

Write-up for the team: `report/handtracking_summary.pdf`. 3D comparison videos: `results/overlays/*_3d_E0_vs_E1best-sgbm.mp4`.

## 5. Next steps and blocked items

| item | status |
|---|---|
| Score the team's current hands (`objfill_v2s`) on this benchmark | next, cheap |
| Pass the per-side hand size into the sim loader; re-run the teammate's replay | next |
| FoundationStereo instead of SGBM | **done, not promoted** (see above) |
| E1c-traj variant with the 2D pull lateral-only | idea: keep its 2D gain without hurting depth |
| E4 verifier, E5 per-frame selection (E1 vs HaWoR+stereo) | code for E4 written; needs the final E1/E2 set |
| E3 EgoFound3R | **skipped (user decision)**: backbone `facebook/VGGT-Omega` gated (manual approval by Meta), no HF token. Code + env ready in `methods/` |
| Set B labels (60 frames x 2 views, grasp intervals) | tool ready (`labeling/label.html`); est. 2.5–3 h keypoints + 1–1.5 h grasps; needed for grasp physics, E6, E7 and to calibrate the pseudo-GT noise floor |
| Official MANO_LEFT | missing; mirrored-right stand-in used (HaWoR) |
| Overlay / 3D hand videos | deferred by user until there are clear wins |
