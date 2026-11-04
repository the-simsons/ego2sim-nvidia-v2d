# TODO: per-frame fallback and follow-ups

Status 31 Oct 2026. The per-frame fallback works on the benchmark (commit 05b40cf) but is not yet in the pipeline stage.

## Where we are

HaWoR sometimes latches onto another person's hand (e.g. episode 24, left hand, frames 54–96: about 1 m off, on a
bystander; video `results/overlays/fallback_ep24_left.mp4`). `handbench/hand_fallback.py` catches those frames from
cam_a signals only (share of projected joints inside the hand's SAM2 mask, distance to the cam_a keypoints) and swaps in
a per-frame HaMeR hand, then one Savitzky–Golay pass over the spliced track.

Benchmark, 20 episodes (`results/REPORT_TABLES.md`):

| | 3D error | sideways | cam_b kp | jitter | coverage |
|---|---|---|---|---|---|
| stereo-rescaled HaMeR (E1best-sgbm) | 49.5 mm | 17.3 mm | 21.0 px | 2.73 | 100 % |
| multi-view tracker (E7tri-hawor) | 53.3 mm | 16.1 mm | 27.4 px | 3.20 | 94 % |
| **+ per-frame fallback, smoothed (E8fallback-e1best-smooth)** | **47.5 mm** | **10.7 mm** | **17.9 px** | 3.09 | 98 % |

Reproduce: `python -m handbench.experiments.fallback --primary E7tri-hawor --fallback E1best-sgbm --name
E8fallback-e1best-smooth --smooth`, then `run_eval` and `compare`.

## 1. Fallback in the pipeline stage (`handbench/pipeline/multiview_hands.py`)

- [ ] **HaMeR pass**: run the upstream per-track HaMeR regression (`run_ego_wilor.py` step 14, `--run_hamer_pass`)
      on our SAM2 hand masks, from the teammates' `~/v2d/envs/v2d_hamer` env, importing (not editing) their code.
      Output to `$E/hands/multiview_work/hamer/`; reuse when present.
- [ ] **Correct the HaMeR hands** like the benchmark fallback: stereo scale (+ keypoint shift where available) in the
      benchmark camera. The benchmark used E1best-sgbm (stereo scale, shared size, dz, smoothing); decide which of those
      steps the pipeline needs.
- [ ] **Decide per frame** with `hand_fallback.plausibility` / `is_plausible` / `better` / `choose_fallback`; the stage
      already has the cam_a masks and keypoints.
- [ ] **Splice, then smooth once** (`seam_offsets`, then Savitzky–Golay w7 o2 over the spliced runs), then `fill_gaps`.
- [ ] Mark swapped frames in the output (`"correction": "fallback"`) and count them in `multiview_hands_info.json`.
- [ ] Unit test for `choose_fallback` / `seam_offsets` in `handbench/tests`.

Blocked on GPU: both GPUs were full with another user's training jobs (31 Oct, about 3–4 GB free each).

## 2. Validate

- [ ] Episode 41 through the pipeline command (as `results/pipeline_multiview_check/`): no regression against the
      current 45.9 mm / +2.6 mm.
- [ ] A failure episode (24, 22 or 0) through the pipeline. These have no SAM2 hand prompts in `~/v2d/ego2sim/` yet;
      prompts exist only for episodes 2, 11, 13, 18, 21, 23, 26, 40, 41.

## 3. Docs

- [ ] `BEST_APPROACH.md`: recommendation, results table, known limits (the ~3 % HaWoR failures are now handled).
- [ ] `ARCHITECTURE.md`, `handbench/ARCHITECTURE.md`: `hand_fallback` block, HaMeR pass in the pipeline block.
- [ ] `README.md` Usage: new `correction` value, extra cost of the HaMeR pass.

## 4. Open questions

- [ ] Fast wrist motion (> 2 m/s) is 1.10 % of frames with the fallback vs 0.28 % without; most of it comes from
      filling frames HaWoR missed (replace-only: 0.66 %). Try filling only stretches of several frames, or smoothing
      the filled stretches harder.
- [ ] Thresholds (mask hit 0.2, 150 px) were set by eye on episode 24; check sensitivity.

## 5. Earlier next steps (from BEST_APPROACH.md)

- [ ] Sim check: re-run the teammate's physics replay with the new hands plus object-aware filling.
- [ ] Hand size in the sim loader: pass the per-hand size factor (hands are about 15 % too big in sim).
