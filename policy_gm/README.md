# G+M: decoupled grasp (G) + movement (M) policies

Experimental alternative to the per-episode S0 policies in `policy_s0/` (it did not beat S0 and is not part of the Track 3
submission). One grasp policy **G** is behaviour-cloned from the fingers of several S0 teachers and shared across episodes;
a per-episode movement policy **M** (FlashSAC, the S0 recipe) drives the wrists and an engage switch while the frozen G
drives the fingers. The question: does a shared G transfer grasps to episodes it has not seen?

Runtime: the same FlashCHORD image and S0 code as `policy_s0/` (`flashchord-cuda12-575-test:193382a1-20261018`).

## Layout

| Path | What |
| --- | --- |
| `gmd/PLAN.md` | The original design and test plan (stages, gates, risks) |
| `gmd/rollout.py`, `gmd/label.py` | Teacher rollouts (nominal, perturbed, action noise) and per-hand phase / engage labels |
| `gmd/gfeat.py`, `gmd/train_g.py` | G's 254-d per-hand input (palm frame) and its behaviour cloning (MLP 3 x 512); `--neg_frac`, `--off_net` etc. for the causal / gated variants |
| `gmd/genv.py`, `gmd/train_m.py`, `gmd/run_m.sh`, `gmd/eval_m.py` | G inside the simulator (M action = wrist 6 + engage per hand), M training with early stopping, checkpoint scoring |
| `gmd/meshmix*.py` | Per-world v1/v2 object meshes (both meshes' hulls in every world, the unused set parked; per-world shape code) |
| `gmd/episodes.py` | Episode tables (S0 teacher checkpoints, reconstruction, world counts) per run (`GMD_SET`) |
| `gmd/v2_pipeline.sh` ... `gmd/v6_pipeline.sh` | End-to-end runs: teachers, G, Gate 1, M on every episode, summary and videos |
| `gmd/gdiag.py`, `gmd/summarize_full.py`, `gmd/render_m.py` | Diagnostic rollouts, summaries and curves, videos |
| `gmd_ft/` | Fine-tuning experiments: a bounded finger correction R on top of a frozen G, and joint M + G training |
| `voc/voc_ff.py` | Feedforward virtual object controller (the helper force without its 0.1 s lag), used by M (`GMD_VOC=ff`) |

The scripts expect the `policy_s0/` layout next to them, mounted as `/env` (`s0/`, `cfg/`, `train_logged.py`, plus
`gmd/` and `voc/` from here), as in `gmd/run_fc.sh`.

## Runs

| Run | G | Teachers | Notes |
| --- | --- | --- | --- |
| 1 (`full`) | G-9, plain behaviour cloning | 41 40 26 18 2 35 8 3 36 | v1 references |
| 2 | g9v2 | same 9 | v2 references, feedforward VOC |
| 3 | g9v2negB | same 9 | causal engage bit (counterfactual release frames, contact dropout, spoon oversampling) |
| 4 | g9gated | same 9 | separate engage-on / engage-off networks |
| 5 | g12 | + 1 14 28 | 23 episodes, per-world v1/v2 mesh mixing on 14 15 16 19 20 30 38 |
| 6 | g17 | + 38 30 6 32 15 | M on the 6 remaining unseen episodes (37 5 16 19 20 13) |

Findings: with the teacher's own wrists, G keeps 87-101 % of the teacher's score on 15 of 17 teachers (Gate 1, run 6), but
M + G stayed below S0 on almost every episode (best: ep30 0.866 vs 0.865 in run 3). The gap is mostly M not finding the
wrist motion S0 learns (e.g. pot-lid removal in ep1 / ep2), and for unseen episodes G's grasp not transferring to a new
mesh (e.g. ep38's dust pan, Gate 1 22 % unseen vs 99 % as a teacher).
