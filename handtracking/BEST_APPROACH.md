# Hand tracking: current best approach

Status 31 Oct 2026. How to run it in the pipeline: [README.md](README.md) (Usage). Code map: [ARCHITECTURE.md](ARCHITECTURE.md).
Full write-up: [report/handtracking_summary.pdf](report/handtracking_summary.pdf); all numbers:
[results/REPORT_TABLES.md](results/REPORT_TABLES.md) (20 episodes) and [results/REPORT_TABLES_team9.md](results/REPORT_TABLES_team9.md)
(the 9 episodes that also have the team's current hands).

## Recommendation

Use the **multi-view hand tracker** (`HAND_TRACKER=multiview`) as the pipeline's hand stage, followed by the team's
existing `clean_hand_tracks` and object-aware gap filling (`fill_hand_gaps_object`), which are unchanged.

| Same visible hand-frames (benchmark experiment label in brackets) | 3D error | depth bias | sideways | held-out cam_b: joints within 10 px |
|---|---|---|---|---|
| **9 episodes, 2,350 frames** | | | | |
| team's current hands (`objfill_v2s`) | 70.7 mm | +43 mm | 20.4 mm | 24 % |
| stereo-rescaled HaMeR ("E1best-sgbm") | 52.6 mm | +10 mm | 17.3 mm | 27 % |
| **multi-view tracker ("E7tri-hawor")** | **52.3 mm** | +12 mm | **11.0 mm** | **36 %** |
| **20 episodes, 5,316 frames** | | | | |
| upstream baseline ("E0") | 262.5 mm | +259 mm | 26.3 mm | 13 % |
| stereo-rescaled HaMeR ("E1best-sgbm") | **49.3 mm** | +7 mm | 17.2 mm | 25 % |
| multi-view tracker ("E7tri-hawor") | 53.3 mm | +9 mm | **16.1 mm** | **39 %** |

The 3D error is measured against our triangulated reference, which is itself about 21 mm off the truth per joint
(measured on HOT3D), so 3D differences of a few mm between the two stereo-corrected methods are within that noise. The
held-out cam_b column is the cleaner comparison: the multi-view tracker places hands best in the camera no method sees.
On HOT3D, with true hand poses, the same correction cut HaWoR's error from 28.4 to 21.6 mm.

## How the multi-view tracker works

1. **Head camera rig**: three cameras side by side: a centre colour camera (`cam_a`) and two black-and-white cameras
   (`cam_c` left, `cam_b` right), 7.5 cm apart. The rig calibration is refined on the videos (factory values were 23–36
   pixels off; now under 1 pixel), `refs/calib/rig.json`.
2. **SAM2 (Segment Anything Model 2) hand masks**: the pipeline's existing per-hand outlines in cam_a.
3. **HaWoR (Hand World Reconstruction, CVPR 2025)**: reads the cam_a video and estimates a **MANO** hand per frame. MANO
   is the standard 3D hand model (778-point mesh, 21 joints, about 61 numbers for shape, finger bends and wrist pose).
   HaWoR uses the whole clip, so its hands are steadier than per-frame estimators.
4. **RTMPose hand keypoints**: a 2D detector finds 21 hand joints in cam_a and in cam_c, each on a zoomed crop placed by
   the SAM2 masks; keypoints that fall outside their hand's mask are dropped (they belong to the other hand).
5. **Triangulation**: for each joint seen confidently in both cameras, the two viewing rays are intersected to give its
   3D position.
6. **Keypoint correction**: in frames with at least 4 triangulated joints, the whole HaWoR hand is moved (not rotated or
   resized) so its joints line up with them; the move is carried across frames without joints for up to 1 s, because
   HaWoR's depth error changes slowly.
7. **Stereo fallback**: where no move can be carried, the hand is scaled towards or away from the camera so it matches
   **SGBM** (Semi-Global Block Matching) stereo depth from cam_c + cam_b inside the hand mask.
8. **Savitzky–Golay smoothing**: each joint's path is fitted with a smooth curve over 7 frames, removing jitter.
9. **Output**: per hand and frame, the MANO hand in the centre camera's frame, in the same files as the previous hand
   stage; gaps up to 15 frames are filled and marked, longer gaps are left to the team's object-aware filling.

cam_b's keypoints and masks are never used here, so cam_b remains an independent check in the benchmark.

## Known limits

- **Pipeline version checked on one episode only.** On episode 41, run through the pipeline command with the team's
  SAM2 masks, the hands score 45.9 mm 3D error and +2.6 mm depth bias, against 43.8 mm / +1.3 mm for the benchmarked
  version and 72.3 mm / +44.8 mm for the team's current hands; held-out cam_b scores are slightly better. (A first
  version placed its keypoint crops on the union of the pipeline's hand masks; those masks include the forearm and track
  the same hand several times, so crops were far too large and depth bias rose to +22 mm. Crops now follow HaWoR's own
  hand, as in the HOT3D study.)
- **A few bad frames.** About 3 % of frames are off by more than 100 px in cam_a (median 28 px), mostly where HaWoR fails
  or follows the wrong hand (episodes 24, 22, 0). The stereo-rescaled HaMeR hands do not have these failures; on all 20
  episodes they have the lower mean 3D error.
- **Coverage.** HaWoR finds the hand in about 94 % of visible frames; longer gaps rely on the team's object-aware filling.
- **Stereo quality matters.** The stereo fallback only helps where stereo is dense and reliable, as on our rig (60 % of
  hand pixels get a depth). On HOT3D's Aria glasses, where it is sparse, stereo rescaling made hands worse.
- **Simulation.** The sim loader ignores the per-hand size factor, so hands are about 15 % too big in simulation.
- **MANO left hand.** HaWoR uses a mirrored right-hand model as a stand-in for the official left-hand file.

## Next steps

1. **Per-frame fallback**: detect the bad frames (for example, a large disagreement with the SAM2 mask or with the
   stereo-rescaled HaMeR hand) and use the HaMeR hand there. This targets the 3 % failures that dominate the means.
2. **Sim check**: re-run the teammate's physics replay with the multi-view hands plus object-aware filling, against the
   current hands, to confirm the gain carries over to grasps.
3. **Hand size in the sim loader**: pass the per-hand size so fingertips land where they should.
