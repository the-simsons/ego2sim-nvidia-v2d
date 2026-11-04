# Final submission (4 Nov): per-episode Kaggle diagnosis and three episode fixes

The 3 Nov package (`selection.txt` / `submission_best_1103`) scored 0.60861 AUC. On 4 Nov we measured each episode on Kaggle,
diagnosed the weak ones, and replaced three episodes. Final package: 0.65069 AUC.

## Scripts

| Script | What |
| --- | --- |
| `make_probe_packages.py` | 20 diagnostic packages, one per episode: only that episode keeps its rollout, the others keep frame 0 and are moved +1 m from frame 1 on (checkpoint field `masked-probe-not-a-rollout`). Episode AUC ~ 20 x (score - control). |
| `swap_eval.sh` | Official kit evaluation of one checkpoint on a chosen reference version (`PKG_OVERRIDE` / `FEAT_OVERRIDE` for corrected references); used for single-episode swap tests: the 3 Nov package with one episode replaced, episode change = 20 x (score - 0.60861). |
| `overlay_track.py` | Shaded overlay of the tracked object (and its 180 deg flips) on the egocentric video, to check orientation ambiguities. |
| `flip_reference.py` | Copies a reference package and turns one object over (180 deg about a body axis through the mesh centre) in every frame. |
| `add_reset_frames.py` | Adds a pickup reset every 3rd frame where some object moves, to an episode's S0 extras. |
| `run_after.sh` | Runs S0 jobs (`s0/s0_queue.sh`) on a GPU after another queue finishes, with per-job reference package and extras. |
| `assemble_final.sh` | Builds and packs the final package from the 3 Nov package and the three replaced episodes. |

`cfg/curriculum/tracking_96m_pickup_ray5decay_contact.yaml`: `tracking_96m_pickup_ray5decay` with the object-contact
reward at 5 in every stage and the hand-ray reward halved.

## The three replaced episodes

**ep36 (spoon).** In every reference version the tracker started the spoon face-down (bowl opening pointing down at frame 0);
the egocentric video shows it face-up, and ep33/35/37 (same mesh) start face-up. The host aligns a single-object episode on
the object's frame-0 pose, so this rotated the whole trajectory (Kaggle episode AUC about 0.1). Fix:

    flip_reference.py <ego2sim_recon_v2clean_pkg> <ego2sim_recon_v2clean_spoonflip_pkg> 36 wooden_spoon x
    # S0, same recipe as ep36_v2clean_ray5decay (2048 worlds):
    PKG_DIR=<spoonflip pkg> FEAT_DIR=<s0_features_v2clean> NO_COMPARE=1 sg e2sshare -c "s0/s0_queue.sh 0 jobs.txt"
    #   jobs.txt: single 36 eval/ep36_v2cleanflip_ray5decay - 2048 overrides_ep36.txt tracking_96m_pickup_ray5decay wrench 0.001

Checkpoint 67.1M (64-world AUC 0.77 against the flipped reference). Swap test: +0.550 episode AUC.

**ep19 (sweeper + dust pan).** The reference hands do not reach the objects, so the policy got no guidance to them.

    add_reset_frames.py <ego2sim_recon_v2_pkg> 19 <s0_features_v2> <s0_features_v2_carry>
    PKG_DIR=<ego2sim_recon_v2_pkg> FEAT_DIR=<s0_features_v2_carry> ... s0/s0_queue.sh with
    #   single 19 eval/ep19_v2recon_contact_ray5decay - 1024 overrides.txt tracking_96m_pickup_ray5decay_contact wrench 0.001

Checkpoint 67.1M. Swap test: +0.194 episode AUC.

**ep8 (beige cup + pot).** The existing v1-reference policy (ep8_v1_cleanedref_ray5decay, 92.3M) instead of v2clean.
Swap test: +0.097 episode AUC.

Other swaps tested and rejected (episode AUC change): ep28 v2raw -0.129, ep33 v1 -0.161, ep20 v2raw -0.078, ep19 v1 -0.348,
ep30 v1 -0.053, ep10 contact variant -0.038, ep33 contact variant -0.001.
