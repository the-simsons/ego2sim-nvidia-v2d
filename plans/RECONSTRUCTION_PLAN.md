# Ego2Sim reconstruction: implementation plan

For a coding agent picking this up cold. Read all of it before changing anything. Section 9 is the order of work.

## 1. Goal

Ego2Sim turns a head-mounted video of a person manipulating objects into a simulation scene and a robot policy that
reproduces the object motion. This plan covers the reconstruction half: object mesh, object 6-DoF trajectory, camera
path and hand motion, from video.

**The goal is the most accurate reconstruction we can get from video alone.**

Two rules define "from video alone" and they are not negotiable:

1. **The final pipeline takes video and nothing else.** No motion capture, no scanned meshes, no hand-measured object
   sizes at inference time. A separate video of the same object (a "scan clip", section 5.2) is allowed: it is video.
2. **Motion capture is for measuring and for training, never an input to reconstruction.** We have an OptiTrack rig.
   Its trajectories may be used to (a) score every stage, (b) train or tune models and thresholds that then run on
   video only, and (c) stand in for the object trajectory when training policies, but only until the video
   trajectories pass the gate in section 8. After that, policy training uses reconstructed trajectories only.

## 2. Where things stand

Numbers are from the NVIDIA Video-to-Data Challenge (Track 3) and from our first own recordings. Treat them as the
baseline to beat; re-measure before relying on any of them.

| Stage | Current method | Measured | Main weakness |
|---|---|---|---|
| Depth | FoundationStereo on the challenge's stereo head rig; MoGe on our single phone camera | MoGe is 3-6% short against mocap distance (3 clips) | our own recordings have one camera and uncalibrated intrinsics |
| Object mesh | Claude builds it in Blender, render-and-compare (the "Ego2Sim agent"); depth fusion for one static object | Chamfer distance to scan (CD-O) 0.19-0.87 cm per object, 0.29 on the hidden set | the loop optimises silhouette overlap, which does not predict CD-O (section 2.1) |
| Object tracking | FoundationPose, 16 hypotheses, on metric depth; frame 0 set from the still start | against mocap, 9 public episodes: AUC 0.833, mean error 2.85 cm with scanned meshes; **0.691 with our own meshes** | our meshes; look-alike poses of symmetric objects; frame-0 error rotates a whole scene in the scorer |
| Camera path | stereo visual odometry (challenge); **derived from mocap** on our recordings | box overlap 0.79-0.82 median on our clips | on our recordings it is not video-only yet |
| Hands | WiLoR + SAM 2 masks + depth alignment + gap filling; teammates' multi-view tracker on the challenge rig | 48 mm 3D error with the multi-view tracker; on our phone clips the hand is detected in only 40-60% of frames | detection gaps, depth of the hand, no ground truth on our rig |
| Reference cleaning | de-spin, flip unwrap, settle, push apart overlapping hulls | hidden-set AUC 0.35 -> 0.65 together with better policies | rules were tuned by hand on 20 episodes |

### 2.1 Findings that should shape the work

- **Silhouette overlap is the wrong target for meshes.** Across 10 agent runs, final overlap and CD-O are unrelated
  (rank correlation +0.07). The object with the lowest overlap (0.85) had the best CD-O. Modelling a pitcher as a
  hollow shell took CD-O from 0.87 to 0.64 at the same overlap. Only 1 of 11 runs met the 0.93 overlap gate; the rest
  spent their whole budget on it. Sample is small: confirm on the benchmark (section 4).
- **The agent's poses drift along the camera ray.** Final passes show 2-8 cm median depth error with a one-sided bias,
  because pose sweeps ran with depth weight 0. Until that is fixed the depth residual cannot show shape errors.
- **Meshes lose their score on height and relief the head camera barely sees.** Error maps of four agent meshes
  against their scans (2026-11-17): a dust pan was built as a flat tray 2.3 cm tall where the real one is 4.5 cm,
  with a bulge under the tray and raised walls at the wide end; a lid's handle arch was a third too low. The
  underside is hidden while an object rests on the table. Scale was not the problem (the metric's own rescale was
  3-5%). Two runs also modelled motion-capture marker balls as parts of the object.
- **The planner/builder split has not shown a gain.** First comparison, one run per object: lid CD-O 0.277 -> 0.341,
  dust pan 0.373 -> 0.400 (both worse, possibly run-to-run noise). Depth-weighted posing did cut depth error
  (5.4 -> 2.4 cm on the pan) without improving CD-O, because it cannot show hidden surfaces. Do not make the split
  the default without repeats (P2.3).
- **Our own meshes cost more accuracy than anything else in tracking** (0.833 -> 0.691 AUC).
- **Symmetric and near-symmetric objects flip.** A plain cuboid has 4 equivalent orientations; a spoon read
  upside-down cost a whole episode on the leaderboard. Resolving flips needs both a symmetry declaration per mesh
  and temporal reasoning.
- **A previous video replay tool cut clips at 60%** (frame count vs control rate). Fixed in `scripts/sim_video.sh`.
  Check lengths whenever a tool converts between video frames and simulator steps.

## 3. Code, data and machines

- Repo: `the-simsons/ego2sim-nvidia-v2d`. **Branch from `main`** (it has teammates' hand tracking and policy code).
  The tiered agent harness, plateau stop and cost report are on branch `recon-plan-and-tiered-harness` with this
  plan; check whether it has been merged before building on them.
- `tools/` (reconstruction tools), `scripts/` (drivers), `agentstar_harness/` (agent brief, contract, planner agent),
  `policy_s0/` (policy training). `scripts/ego2sim_public_episode.sh` is the reference driver for one episode.
- Compute: `riddle` (2x RTX 4090, shared lab machine). Read `CLAUDE.md` and `NOTES.md` first: they are not in the
  repo; they live at `~/v2d/` on riddle, mirrored at `D:\v2d\` on the laptop. `NOTES.md` is the running log of what
  was installed, measured and tried, with exact commands. Hard rules:
  never run `sudo`; use a GPU only when it is completely free; never reboot; keep everything under `~/v2d` or
  `/usr1/e2s/rborah_v2d`; before any run over about 30 minutes, state the command, duration and GPU and get a yes.
- Our recordings: `/usr1/e2s/rborah_v2d/aims/` holds 3 clips (`vid/*.MOV`, phone, about 30 fps, one camera) with
  OptiTrack exports (`traj/*.csv`, 240 Hz, Z up, metres, one rigid body `BlueBox`), plus working scripts that are
  **not in the repo yet** (`sync_clips.py`, `fit_camera.py`, `cam_from_fp.py`, `refine_camera.py`, `mocap_table.py`,
  `aims_*.sh`, `check_reference.py`, `replay_measure.py`, `make_aims_package.py`). They hard-code paths and one
  object. Task P0.1 brings them in.
- Challenge data with mocap ground truth: 20 public episodes under `~/v2d/v2d_data/track_3/public`, scans included.
  Scoring tools: `tools/ref_parquet_as_rollout.py`, `tools/score_public_episode.py`, `tools/cdo_board.py`.

## 4. First build the measuring stick (phase P0)

Nothing else in this plan can be judged without it. Deliver a benchmark that scores any reconstruction against
mocap, per stage, with one command.

**P0.1 Bring the mocap tools into the repo** as a package `mocap/` (or `e2s/mocap/`): CSV loader (Motive export:
units, up axis, quaternion order, frame rate from the header), time sync, rigid-body-to-mesh transform, table
height from rest poses. Remove hard-coded paths and the single-object assumption. Keep the scripts' behaviour:
their outputs on the 3 existing clips are the regression test.

**P0.2 Clip manifest.** One JSON per clip: video path(s), intrinsics, mocap file, rigid bodies with their mesh and
transform, sync offset, which hand, split (`train` / `val` / `test`). Splits are by **object and by recording
session**, never by frame, so a model tuned on mocap is always tested on objects and sessions it has not seen.

**P0.3 Metrics**, all computed against mocap after a single rigid alignment of the reconstructed world to the mocap
world, estimated on the first still stretch only (so drift later in the clip is not hidden):

| Stage | Metric |
|---|---|
| Object trajectory | position error (cm) and rotation error (deg): median, 95th percentile, max; AUC and mean per-point error with the challenge code; count and duration of look-alike flips (rotation error within 10 deg of a symmetry of the mesh); frame-0 error |
| Object trajectory, symmetric objects | the same, with rotation error taken modulo the declared symmetry, reported next to the plain numbers |
| Mesh | CD-O against the scan or CAD with the challenge code; longest-extent error (%); watertight yes/no |
| Camera path | needs markers on the camera (P1.1): translation and rotation error, drift per metre |
| Hands | needs a marker cluster on the wrist (P1.1): wrist position and orientation error; fraction of frames with a real detection; hand-to-object surface distance in frames where mocap shows the object moving |
| Timing | sync offset found from video alone vs the offset from a sync event (P1.2) |
| End to end | the reference replayed in simulation with the assist off, and later the trained policy, scored against mocap |

**P0.4 Report.** `bench/run.py --split val --pipeline <config>` writes a table per clip and a summary, and appends
one row to `bench/history.csv` (git commit, config, every summary number). Every later task is accepted or rejected
on this table. Add a `--stage` option so a single stage can be re-scored with the others taken from cache.

**P0.5 The mocap firewall.** Make the rule in section 1 hard to break by accident:
- Reconstruction code lives in a package that must not import the mocap package. Add a test that fails if it does.
- Every output (bundle manifest, reference parquet metadata, package `quality.csv`) carries
  `trajectory_source: video | mocap` and `mocap_used_for: []`. The policy trainer reads it (section 8).
- The benchmark runs reconstruction in a process that cannot see the mocap files (separate directory, passed only to
  the scoring step).

Acceptance for P0: the 3 existing clips score through `bench/run.py`; the existing mocap-assisted outputs reproduce
within 1 mm; a **video-only baseline** exists for those clips (camera from video, object from FoundationPose, table
from depth) with its numbers in `bench/history.csv`. That baseline is expected to be poor. It is the number to beat.

**Status (2026-11-19): P0 accepted, with the gaps below.** Code is in `bench/`, `e2s/`, `tests/`, `tools/rgbd_vo.py`.
- P0.1: `e2s/mocap/` (Motive loader, rigid-body fits, sync, ground truth) and `e2s/mocap/assist/` (sync, camera,
  table), configured from the manifest. Re-running them reproduces the 2026-11-08 outputs on all 3 clips at 0.000 mm
  (camera paths, sync lags, table depth). The outline-refine step is sensitive to the library versions: it reproduces
  in the `v2d_sam2` env (scipy 1.17, OpenCV 5.0) but differs by 30-44 mm with the system scipy 1.11 / OpenCV 4.6.
  Run it in `v2d_sam2`.
- P0.2: `bench/clips/*.json`. All 3 clips are `val`, because there is only one object and one session (needs P1.5).
- P0.3: object-trajectory metrics are in place. Missing: frame-0 error (needs a second alignment source), mesh CD-O
  in the bench, camera and hand metrics (need P1.1 markers), and timing (needs P1.2). Alignment tries the 4
  look-alike start orientations of a cuboid. Without this, FoundationPose starting 180 deg turned scored as a
  140 deg error for the whole clip.
- P0.4: `python -m bench.run --pipeline video_baseline | camera=X,object=Y`. The stage spec plays the role of `--stage`.
- P0.5: `tests/test_firewall.py`. Provenance is written by `build_ego_bundle.py` (`--trajectory-source`), copied
  into each retargeted partition by `ego2sim_episode.sh`, and checked by `train_episode.sh` via
  `tools/check_provenance.py` (`FINAL=1` refuses non-video). Not done: provenance in package `quality.csv`, and running
  reconstruction in a process that cannot see the mocap files (today the separation is only by convention: the
  manifest's `inputs` block).

| pipeline (val, 3 clips, pooled) | source | pos median / p95 cm | rot median deg | ADD-S AUC |
|---|---|---|---|---|
| mocap_assisted (sanity) | mocap | 0.00 / 0.00 | 0.00 | 0.997 |
| diag: FoundationPose + mocap camera | mixed | 2.33 / 7.95 | 1.46 | 0.840 |
| **video_baseline**: rgbd_vo (MoGe depth) + FoundationPose | **video** | **11.52 / 27.36** | **5.50** | **0.382** |

The diagnostic row flatters FoundationPose, because the mocap camera was itself fitted to FoundationPose poses. Even
so, the gap between the two rows puts most of the video-only error in the camera path: the MoGe-depth odometry
drifts 8-19 cm per clip. Rotation error is small without the camera error. That makes stereo/LiDAR depth (5.1a) and
camera work (P4) worth more than this plan's order suggests. P3.1 (symmetry) comes next as planned.

## 5. Accuracy work, in order of expected gain

Each item states what to build, how to judge it, and what would make us drop it. Do them in the order of section 9,
one at a time, each as its own change scored on the benchmark.

### 5.1 Capture (P1): fix what the pipeline cannot recover later

The first recordings used a phone's ultra-wide camera with a guessed focal length, no distortion model, no depth
sensor and no sync event. Fixing capture is cheaper than compensating in software.

- **P1.1 Ground truth for more stages.** Rigid marker clusters on the camera and on the back of each hand or wrist.
  One-time calibration from the camera's marker cluster to its optical centre (hand-eye calibration with a
  checkerboard seen by the camera while mocap tracks the cluster). These are for scoring only (firewall applies).
- **P1.2 Sync.** The user is building a system that synchronises the mocap and video recordings in hardware
  (2026-11-19). Once it exists, the video-to-mocap offset is a recorded value: it goes into the clip manifest
  (`ground_truth.sync`, with `method: hardware`), and no pipeline step estimates it. Do not build more software sync.
  Until then, `e2s/mocap/assist/sync.py` and the lag searches in `e2s/mocap/assist/camera.py` stay, only for the
  existing unsynchronised clips. Give `camera.py` an option to keep the recorded lag fixed, and use it for
  hardware-synced clips. Sync is a scoring/mocap-side quantity either way: reconstruction never uses it.
- **P1.3 Camera.** Calibrate intrinsics and distortion once per camera and lens (checkerboard, fisheye model if the
  ultra-wide is kept) and undistort before anything else. Prefer a rig with real depth: a calibrated stereo pair
  like the challenge's, or the phone's LiDAR depth recorded with the video. Record at 60 fps with a short exposure:
  motion blur costs tracking during fast moves.
- **P1.4 A recording protocol.** Per object: one scan clip (5.2); per take: 2 s still at the start and end, taps,
  the object set down in a different resting pose at least once.
- **P1.5 More objects.** At least 8 objects for a meaningful benchmark: boxes, a cylinder, a cup, a pan, a tool with
  a handle, something thin, something textureless, something symmetric. Use objects with scans or CAD where possible
  so CD-O can be scored.

Judge by: the same clip scored with and without each capture change (depth sensor vs monocular depth; calibrated
vs guessed intrinsics). Keep monocular-only as a tracked configuration, because it is the hardest and most general.

### 5.1a Depth: use the camera's sensor when it has one

Every stage that needs metric geometry (object size, mesh heights, pose along the ray, hand depth, table plane,
camera scale) reads depth through one interface, which picks the best source the recording has:

| Priority | Source | Notes |
|---|---|---|
| 1 | calibrated stereo pair | FoundationStereo on the rectified pair, as on the challenge rig; dense and metric |
| 2 | LiDAR / time-of-flight (for example the phone's) | metric but low resolution and short range; upsample with the colour image, and fuse with stereo or monocular depth for edges and thin parts |
| 3 | monocular metric depth (MoGe) | last resort; 3-6% short on our clips and unreliable on flat or tilted surfaces; rescale against anything metric that is available |

- One module returns depth, a per-pixel confidence and `depth_source`, and writes the source into the capture
  manifest (`source.depth_values`) so the agent brief and every report can say what they relied on.
- When two sources exist, fuse them (`tools/fuse_depth.py` already fuses stereo with MoGe) and keep the
  disagreement as the confidence.
- The first recordings were made with a phone that has LiDAR but saved colour video only. Record depth with the
  video from now on (P1.3). If the recording app also offers the phone's own camera tracking, treat that as an
  optional input to compare against visual odometry; it is on-device sensing, not motion capture, but ask the
  user before depending on it.
- Report every benchmark row with its depth source. A gain that only exists with sensor depth is still a gain,
  but the monocular row shows what the pipeline can do on arbitrary video.

### 5.2 Object mesh (P2)

Largest single loss today, and the diagnosis in 2.1 says why: the task clip does not show enough of the object.
So the order is: get more evidence (a scan clip), get a prior for what stays unseen (a generative model), and use
the agent to repair and verify rather than to model from nothing. Then a selector.

The original AgentSTAR worked from monocular video with estimated depth. We have, or can have, sensor depth
(section 5.1a): use it everywhere a size, a height or a distance along the camera ray is needed.

- **P2.1 Scan clip reconstruction (new, expected to be the best source).** A 20-30 s video orbiting the object on
  the table, then flipped and orbited again, so the underside is seen. Segment with SAM 2, estimate cameras from
  the background, reconstruct (a surface-oriented method: 2D Gaussian splatting or neural SDF, or classical
  multi-view stereo, with sensor depth as supervision where available), merge the two halves by registering the
  object, scale from metric depth, make watertight, simplify. Output in a fixed canonical frame with a symmetry
  file (P2.4). This uses video only, so it is allowed in the final pipeline.
- **P2.2 A generative prior for the unseen sides (new).** Before any agent work, generate a complete mesh from a
  few masked views of the object with a multi-view image-to-3D model conditioned on mask, camera and depth. The
  reference design is NVIDIA's GATOR (Wu et al., arXiv:2610.11215): it reconstructs complete, scene-aligned objects
  from 1-10 casual views, improves with more views, and places the result by refining translation and isotropic
  scale against silhouette and depth. Its code is not released yet ("coming soon"); until it is, use the open
  models it builds on or compares with (TRELLIS.2, Pixal3D, SAM 3D, multi-view variants) and swap GATOR in when
  available. Feed it the views that differ most, including a lifted or tilted frame when the clip has one. Metric
  scale comes from our depth, not from the model.
- **P2.3 The agent as a repairer, not a modeller.** GATOR's own numbers argue for this: an agent modelling from
  scratch produced no valid mesh in 1 minute and needed 5 for all ten test objects, a generated mesh took about
  30 s and was more accurate than the agent at 20 minutes, and agent edits on top of the generated mesh helped
  appearance and local topology but slightly worsened geometry on one real dataset. So:
  - start the agent from the best available mesh (P2.1 or P2.2); model from primitives only when both fail;
  - targeted repairs that keep the frame and the scale: holes, thin walls, flat bottoms that should not be flat,
    parts fused that should be separate, watertightness, levelling;
  - **every edit is checked and reverted if it does not help**: render before and after under the same cameras,
    compare depth agreement and silhouette, keep the edit only if neither gets worse (brief rule 12). GATOR names
    automatic geometric verification of agent edits as its open problem; we have metric depth to do it with;
  - a verdict at the end: `done` or `needs_review`, with what is unverified;
  - a time or iteration budget per object, small by default (GATOR uses 10 minutes).
  The existing brief `agentstar_harness/AGENT_TASK_EGO2SIM_V2.md` already has: depth-weighted posing, heights
  measured from depth, the underside taken from lifted frames, undo of unhelpful edits, and a symmetry file. It
  also tells the agent to skip unmistakable motion-capture marker balls, as a minor rule that must never remove
  real parts: final recordings will not have markers, so do not build on it. Still to build: a gate on **surface agreement** in millimetres (overlap stays only as a
  sanity floor, for example 0.85); a fixed canonical origin so a proportion edit does not invalidate every pose; a
  fitting tool that tunes the numeric parameters of `build()` with a derivative-free optimiser; and a start-from-
  mesh mode. Repeat the planner/builder comparison on at least 6 objects with 3 runs each before deciding on it.
- **P2.3b A physical sanity check.** Drop each mesh on a plane in the simulator with its collision hulls: it must
  come to rest in the pose it rests in on the table in the video, not rock or topple. Cheap, and it catches
  missing feet and bases that are not flat where they should be.
- **P2.4 Symmetry declaration** for every mesh: `{"revolution_axis": ..., "flips_180": [...]}` in the canonical
  frame, written by the source that made the mesh and checked by rendering (a declared symmetry must reproduce the
  silhouettes and depth within noise). Consumed by tracking (5.3) and cleaning.
- **P2.5 A selector that needs no ground truth.** Score every candidate mesh by render-and-compare on the task
  clip (depth residual, silhouette, and photometric error if textured) and pick the best. Then check on the
  benchmark that the selector's choice has the best CD-O; if it does not, fix the score, not the choice.

Judge by: CD-O and longest-extent error per object on `val`, and the object-trajectory metrics when tracking uses
that mesh (a mesh is only as good as the tracking it supports). Target for P2: tracking with our meshes within
0.03 AUC of tracking with scans, on `val`.

### 5.3 Object trajectory (P3)

- **P3.1 Symmetry-aware tracking.** Pass the declared symmetries to FoundationPose, and replace per-frame winner
  selection with a global choice over time: keep the top hypotheses per frame (including symmetric twins) and solve
  for the smoothest consistent path (Viterbi), with terms for the table plane and for hand contact. A prototype
  exists for a cuboid in `cam_from_fp.py` (`pick_flips`).
  *Status 2026-11-19 (P3.1 + P3.4, first pass): `tools/sym_track.py`, bench stage `fp_sym`, preset `video_sym`.
  - It runs a Viterbi over the look-alike twins of the tracker's pose in the video-camera world.
  - Frame 0 is the robust mean of the frames before the first hand contact, taken from the object and hand masks.
    Contact comes at frames 24/22/10, while mocap shows the box first moving at 52/72/29, so this cue is safe.
  - World speed cannot detect stillness: MoGe-odometry drift makes a resting box move about 5 cm/s.
  - Result on val: 11.52 -> 11.58 cm, 5.50 -> 5.51 deg (no change). FoundationPose made no twin switches on these
    clips, and the benchmark already aligns on the still start.
  - Still to do: top-k FoundationPose hypotheses, which need a wrapper around the pinned module and a GPU run.
    Table and hand terms in the Viterbi.*
  *Status 2026-11-19 (tracker failures): `tools/fp_repair.py`, stage `fp_rep`.*
  - *It splits the object track at implausible jumps and solves a rotation offset per segment, from continuity and
    outline fit.*
  - *It partly fixes the one FoundationPose failure in the val clips (clip 001: lost during a turn, wrong for 40
    frames). Rotation p95 goes from 31 to 25 deg; medians unchanged.*
  - *A full fix needs a per-frame failure signal. Outline fit should give one once the intrinsics are calibrated
    (P1.3). FoundationPose re-registration triggered by that signal belongs in P3.2.*
- **P3.2 Render-and-compare refinement** after tracking: per frame, optimise the pose against depth, silhouette and
  edges (and texture when the mesh has it), with a temporal smoothness term, over a sliding window. Initialise from
  P3.1. This is where sub-centimetre accuracy has to come from.
- **P3.3 Physical constraints.** A resting object lies on the table plane in a stable pose; an object does not pass
  through the table or another object; an object moves only while a hand is near it. Apply as soft terms in P3.2
  first, and keep the existing cleaning stages as a final pass.
- **P3.4 Frame 0.** The first pose anchors everything downstream. Estimate it from the whole still stretch (many
  frames, multi-hypothesis, symmetry resolved using later motion), not from one frame.
- **P3.5 A learned corrector, trained on mocap (allowed use b).** Only after P3.1-P3.4 plateau. A small model that
  takes the tracker's output and image/depth crops and predicts a pose correction and an uncertainty, trained on
  `train` clips with mocap as the label. It runs on video only. Accept it only if it improves `test` objects it has
  never seen; otherwise it has memorised objects and must be dropped.
- **P3.6 Confidence per frame and per clip**, calibrated on mocap: a number that predicts the error without mocap
  (from residuals, hypothesis agreement, occlusion). Needed for section 8: once mocap is gone, this is the only way
  to reject a bad reconstruction.

Judge by: the object-trajectory table on `val`, then `test`. Report flips separately: one flip is worse for policy
training than a centimetre of noise.

### 5.4 Camera path (P4)

*Status 2026-11-19: on the 3 BlueBox clips the camera path, not the object tracker, causes most of the
object-trajectory error.*
- *FoundationPose in the camera frame is within about 1-2 cm and 1.5 deg of the mocap reference.*
- *The `rgbd_vo` camera on MoGe depth drifts 4-6 deg (median) in rotation, and its scale is inconsistent from clip to
  clip (best global scale 0.33-0.84).*
- *`tools/camera_pose_graph.py` (stage `video_pg`) adds per-frame table planes and the resting object to the
  odometry. It did not help: 11.85 cm against 11.52, because the planes come from the same monocular depth.*
- *This moves 5.1a (sensor depth: LiDAR or stereo) and a SLAM camera with global bundle adjustment ahead of further
  tracker work.*
- *Pi3X as the camera (`tools/pi3_camera.py`, stage `video_pi3`) is the new best video-only pipeline: 8.15 cm,
  5.29 deg, ADD-S AUC 0.549 (rgbd_vo: 11.52, 5.50, 0.382).*
  - *Two fixes were needed. Each chunk's Sim(3) scale has to be removed from the rotations. Pi3X's metric scale is
    1.4-2x too large, so the path is rescaled to MoGe depth.*
  - *Chunk scale still varies 0.6-1.2 within a clip. Longer chunks (24) did not change the result.*
  - *Feeding the uncalibrated intrinsics (f = 713) made it worse. Calibrate the camera (P1.3) before trusting
    anything that uses f.*
- *Joint camera + object refinement on Pi3X (`camera_pose_graph.py --no-table --scale-every 10`, stage
  `video_pi3_joint`) adds resting-object constraints (no hand contact) and one odometry scale per 10 frames.
  Result: 5.62 cm / 5.26 deg / ADD-S AUC 0.648, the current best video-only pipeline. End-still error drops to
  1.4 / 4.9 / 7.7 cm per clip.*
  - *Variants are within noise of each other. Do not tune further on these 3 clips. The next gains need more clips
    (P1.5), calibrated intrinsics (P1.3), and a look at the FoundationPose rotation outliers (clip 001 rot p95 68 deg).*

Needed to put hands and objects in one world, and it must come from video.

- Visual odometry or SLAM on the static background with hands and objects masked out, metric scale from depth.
  Compare stereo VO (existing `tools/ego_vo.py`), a learned SLAM system, and depth-sensor odometry where available.
- Use the table plane and gravity as constraints: the table is flat and still.
- Jointly refine camera and static objects over the clip (an object that mocap would show as still should be still).
- Score against the camera marker cluster (P1.1). Until P1.1 exists, score indirectly: reconstructed object path in
  the world against mocap.

### 5.5 Hands (P5)

- Start from the teammates' multi-view tracker on a rig with more than one camera; on a single camera, WiLoR or
  HaMeR with metric depth alignment.
- Close detection gaps: tracking across frames, re-detection with a crop guided by the previous pose, and a second
  detector as fallback. Target: a real detection in over 90% of frames where the hand is in view.
- **Contact refinement.** When the object moves, a hand is holding it. Optimise the hand pose so fingertips lie on
  the object surface during those frames and do not penetrate it, keeping the image evidence. The object-carried
  gap fill in `tools/fill_hand_gaps_object.py` is a crude version of this.
- Score against the wrist marker cluster (P1.1) and by hand-to-surface distance during mocap-confirmed motion.
- The object-centric policy tolerates hand error better than hand imitation does, so rank this below 5.2 and 5.3
  unless the benchmark shows hands limiting the policy.

### 5.6 Cost and throughput (P6)

- Record tokens and dollars per agent run (`tools/token_usage.py` prints dollars; the log is root-owned on riddle,
  so the user runs it).
- Keep the cheaper-builder split only if CD-O per dollar improves on the benchmark.
- The plateau stop saved 9% of iterations on past runs: keep it, do not expect more from it.
- Cache every stage by content hash so re-scoring one stage does not re-run the others.

## 6. What mocap may train

Allowed, with the test in brackets:
- thresholds and weights of any stage (re-scored on `test`);
- the confidence model of P3.6 (calibration on `test`);
- the learned pose corrector of P3.5 (must improve unseen objects);
- depth scale and bias corrections for a camera (must hold on a different recording session);
- the mesh selector's score (its choice must match the CD-O ranking on `val`).

Not allowed: any per-clip quantity taken from mocap at inference (object pose, camera pose, table height, sync
offset, object size), and tuning on `test`.

## 7. Risks and how to notice them

- **Overfitting to a few objects.** Split by object and session; report `train`, `val`, `test` side by side.
- **Mocap is not perfect.** Marker-to-mesh transforms are estimated (the box's was fitted from rest poses to
  0.14 mm). Record the residual of every such fit and do not chase errors below it.
- **Alignment hiding errors.** Align worlds on the first still stretch only; report drift.
- **A good benchmark number with a bad policy.** Keep the end-to-end row in P0.3 and run it at every phase boundary.
- **Shared machine.** Long jobs wait for a free GPU; plan runs so nothing needs both GPUs.

## 8. Policy training data and the switch to video

Policy training gets a `trajectory_source` setting with three modes:

| Mode | Object trajectory | Hands | When |
|---|---|---|---|
| `mocap` | motion capture | video | now, while video trajectories are poor |
| `mixed` | video where the clip's confidence is high, mocap elsewhere, labelled per clip | video | transition, to see whether video trajectories train as well |
| `video` | video only | video | the final pipeline |

The trainer refuses `mocap` and `mixed` when a `final` flag is set, and writes the mode into every checkpoint's
metadata. A policy trained in `mocap` mode is a development result, never a result reported for the pipeline.

**Gate for switching an object class to `video`** (starting values; revise once the benchmark has 8 or more
objects), measured on `test` clips against mocap:

- object position error: median at most 1.0 cm, 95th percentile at most 2.5 cm;
- rotation error: median at most 5 deg, 95th percentile at most 15 deg (modulo declared symmetry);
- no unresolved look-alike flip longer than 0.25 s;
- frame-0 error at most 1 cm and 3 deg;
- and the decisive check: a policy trained on the **video** reference and scored against **mocap** reaches at least
  95% of the score of a policy trained on the mocap reference, on the same clips.

For reference, the current best with scanned meshes on the challenge rig is 2.85 cm mean error, so the gate needs
roughly a threefold improvement. When a class passes, record the commit and numbers in `bench/history.csv` and
switch its default. Mocap recording continues after that only as a periodic check.

## 9. Order of work

1. **P0** benchmark, firewall, video-only baseline. Nothing else starts before this.
2. **P1.1-P1.3** capture fixes that need the lab (markers, calibration, sync), in parallel with step 3 because they
   need a person in the lab. Ask the user to schedule them.
3. **P3.1 + P3.4** symmetry-aware tracking and frame 0: cheapest large gain, code mostly exists.
4. **5.1a** the depth interface, so sensor depth is used wherever a recording has it.
5. **P2.2** a generated starting mesh, then **P2.3** the agent in repair mode with checked and reverted edits and
   the surface gate, then the planner/builder comparison with repeats.
6. **P2.1** scan clips (needs the recording protocol of P1.4), then **P2.3b, P2.4, P2.5** sanity check, symmetry
   files and the selector.
7. **P3.2-P3.3** refinement and physical constraints.
8. **P4** camera path from video, scored once camera markers exist.
9. **P5** hands.
10. **P3.5-P3.6** learned corrector and confidence, then the gate of section 8, class by class.

After each step: run the benchmark, add the row to `bench/history.csv`, write what changed and what it cost in
`NOTES.md`, and stop for review if a number moved the wrong way.

## 10. Working rules for the agent

- One change per commit, each with its benchmark row. Do not batch improvements: we need to know which one helped.
- Do not edit the pinned upstream code (`video_to_data`) or the challenge evaluator; wrap it.
- Small test first: one clip, one object, a few frames, before any full run.
- Report results as measured, including regressions and things not run. Say when a number comes from fewer than
  5 clips.
- Ask before deleting anything large, before any run over about 30 minutes, and before anything that needs `sudo`
  (the user runs those).
- Never handle API keys or tokens; the user enters them.
