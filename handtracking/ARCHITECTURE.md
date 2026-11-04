# handtracking: architecture overview

How the pieces in `handtracking/` fit together. Module-level detail: [handbench/ARCHITECTURE.md](handbench/ARCHITECTURE.md).
How to use them: [README.md](README.md); current best approach: [BEST_APPROACH.md](BEST_APPROACH.md).

```mermaid
flowchart LR
  %% ---------------- inputs
  subgraph IN["Inputs (data/, read only)"]
    T3["Track 3 public data<br/><i>3-camera ego videos, calibration, GT objects</i>"]
    CACHE["Cached upstream run<br/><i>E0 hands, SAM2 masks, depth</i>"]
    TEAM["Team hands snapshot<br/><i>objfill_v2s per-frame MANO JSON</i>"]
    HOT["HOT3D clips (8, Aria)<br/><i>3 cameras + mocap MANO GT</i>"]
  end

  %% ---------------- method runs
  subgraph M["Method runs (methods/, GPU)"]
    HAWOR["HaWoR runner<br/><i>cam_a video → MANO hands per frame</i>"]
    SAM2B["cam_b SAM2 masks<br/><i>cam_b video + cam_a prompts → hand masks</i>"]
    FS["FoundationStereo<br/><i>cam_c+cam_b → dense depth</i>"]
  end

  %% ---------------- references
  REFS["Reference builder (refs/)<br/><i>videos + masks → calibrated rig, 2D keypoints per camera,<br/>triangulated joints, stereo depth crops</i>"]

  %% ---------------- methods / experiments
  ADAPT["Adapters<br/><i>native outputs → common hands.npz<br/>(benchmark camera, MANO, observed/infilled)</i>"]
  EXP["Experiments E1–E7<br/><i>hands.npz → improved hands.npz<br/>(stereo rescale, size, smoothing, triangulation shift)</i>"]
  RES["results/&lt;method&gt;/&lt;episode&gt;/hands.npz<br/><i>one file per method and episode</i>"]

  %% ---------------- evaluation
  EVAL["run_eval<br/><i>hands.npz + refs → per-frame and per-episode metrics</i>"]
  CMP["compare<br/><i>metrics of many methods → REPORT_TABLES*.md</i>"]

  %% ---------------- outputs
  subgraph OUT["Outputs"]
    REP["Reports<br/><i>REPORT.md, summary PDF, BEST_APPROACH.md</i>"]
    VID["Videos<br/><i>2D overlays, 3D side/oblique views</i>"]
    SIM["Sim bundle<br/><i>hands → result.npz for the team's<br/>loader → retarget → FlashCHORD chain</i>"]
    LAB["Labelling page<br/><i>set B frames → hand-clicked keypoints, grasp intervals</i>"]
  end

  %% ---------------- pipeline hand stage (used by the Ego2Sim pipeline, scripts/ego2sim_public_episode.sh)
  subgraph PIPE["Ego2Sim pipeline: HAND_TRACKER=multiview"]
    PIN["pipeline inputs<br/><i>raw cam_a/cam_c video, stereo depth, SAM2 hand masks</i>"]
    MV["multiview_hands<br/><i>HaWoR hands + triangulated cam_a/cam_c keypoints<br/>+ stereo fallback + smoothing</i>"]
    POUT["hands_by_side/{left,right}<br/><i>same files as tools/ego_hands.py →<br/>clean_hand_tracks → object fill → bundle (unchanged)</i>"]
  end
  CORR["hand_correction<br/><i>shift onto triangulated joints, carry over gaps<br/>(shared by pipeline and experiments)</i>"]

  %% ---------------- HOT3D check
  H3["HOT3D checks<br/><i>same refs + methods vs true MANO GT<br/>(floor of the reference, recipe tests)</i>"]

  T3 --> REFS
  CACHE --> REFS
  T3 --> SAM2B --> REFS
  T3 --> FS --> REFS
  T3 --> HAWOR
  CACHE --> ADAPT
  HAWOR --> ADAPT
  TEAM --> ADAPT
  ADAPT --> RES
  RES --> EXP --> RES
  REFS --> EXP
  RES --> EVAL
  REFS --> EVAL
  EVAL --> CMP --> REP
  RES --> VID
  RES --> SIM
  T3 --> LAB
  HOT --> HAWOR
  HOT --> H3
  HAWOR --> H3
  H3 --> REP
  PIN --> MV --> POUT
  CORR --> MV
  CORR --> EXP
  POUT -. scored via adapter .-> ADAPT
```

## Blocks

| Block | What it does | In | Out |
|---|---|---|---|
| Track 3 public data | the 20 public episodes: ego videos (cam_a colour, cam_b / cam_c mono), factory calibration, GT object poses | – | `data/track3` (link) |
| Cached upstream run | the upstream ego pipeline already run on all episodes (WiLoR, SAM2, HaMeR, MoGe) | – | `data/cache/<ep>` (copied / linked) |
| Team hands snapshot | the team's current hands (`objfill_v2s`), copied read-only for comparison | `/home/rborah/v2d/ego2sim` | `data/team_objfill_v2s` |
| HOT3D clips | 8 Aria clips with true hand poses from motion capture | Hugging Face | `data/hot3d` |
| HaWoR runner | runs HaWoR (CVPR 2025) on cam_a (Track 3) or the Aria RGB view (HOT3D), one worker per GPU | video frames | `data/hawor`, `data/hot3d_hawor` (`export.npz`) |
| cam_b SAM2 masks | SAM2 video segmentation on cam_b, prompted from cam_a | cam_b video, cam_a masks, stereo | `refs/<ep>/masks_b` |
| FoundationStereo | learned stereo depth, an alternative to SGBM | cam_c + cam_b video | `refs/<ep>/stereo_fs.npz` |
| Reference builder | calibrates the rig, detects 2D hand keypoints in each camera, gates them with masks, triangulates them, builds stereo depth | videos, masks | `refs/` (`rig.json`, `kp2d`, `kp_gate`, `tri`, `stereo`) |
| Adapters | convert every hand source into the common format, in the benchmark camera | native outputs | `results/<method>/<ep>/hands.npz` |
| Experiments E1–E7 | benchmark variants made from existing methods (stereo rescale, shared size, smoothing, verifier, triangulation shift); the labels are benchmark row names only | `hands.npz`, refs | new `hands.npz` |
| run_eval | scores one method on one set | `hands.npz`, refs | `per_frame_*.csv`, `per_episode_*.csv`, `metrics_*.json` |
| compare | side-by-side tables, own frames and common frames, optional episode subset | metrics of several methods | `results/REPORT_TABLES*.md` |
| Reports | human write-ups of the results | tables, figures | `results/REPORT.md`, `report/*.pdf`, `BEST_APPROACH.md` |
| Videos | overlays in cam_a / cam_b, 3D side and oblique views | `hands.npz`, videos | `results/overlays/*.mp4` |
| Sim bundle | packs hands into the bundle format of the team's sim chain | `hands.npz`, stereo | `sim/<clip>/bundle` |
| Pipeline hand stage (`multiview_hands`) | multi-view hand tracker as a drop-in for `tools/ego_hands.py` ([README.md](README.md), Usage) | raw cam_a / cam_c videos, stereo dir, SAM2 hand masks | `hands_by_side/{left,right}` |
| hand_correction | shifts single-camera hands onto triangulated keypoints and carries the shift across gaps; shared by the pipeline stage and the benchmark experiment | joints, triangulated joints | corrected joints |
| Labelling page | HTML tool for hand-clicked keypoints and grasp intervals (set B) | frames | `labeling/`, `refs/setB.json` |
| HOT3D checks | the same reference protocol and correction recipes, scored against true hands | HOT3D clips, HaWoR | `results/hot3d_*.json` |

## Folders

| Folder | Content | In git |
|---|---|---|
| `handbench/` | all code (Python package) | yes |
| `data/` | inputs, caches, method raw outputs | no |
| `refs/` | references; small summary files are tracked | partly |
| `results/` | per-method metrics (tracked), `hands.npz` and videos (not tracked) | partly |
| `methods/` | third-party method code and their venvs (HaWoR, EgoFound3R, HOT3D toolkit) | no |
| `assets/` | MANO (licensed) and detector models | no |
| `report/` | LaTeX summary, figures, figure script | yes |
| `sim/`, `labeling/`, `logs/` | sim bundles, labelling page + images, run logs | no |
| top level | `README.md` (usage), `BEST_APPROACH.md` (recommendation, results, limits), `ARCHITECTURE.md` (this file) | yes |
