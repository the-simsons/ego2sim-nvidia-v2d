# GMD: decoupled grasp (G) + movement (M) policy — execution plan and tests

Code: `/usr1/e2s/retarget_env/gmd/`. Outputs: `/usr1/e2s/v2d_challenge_data/outputs/gmd/`.
Rules: no edits to FlashCHORD, FlashSAC, the S0 modules or system files (copies only); every container has a memory cap and
a watchdog; nothing is uploaded or committed without approval.

## 0. Fixed facts this plan is built on

- Robot: 2 Sharpa Wave hands. Per hand: wrist 6 (3 position + 3 rotation, integrated deltas) + 22 finger joints.
  S0 action = 56 numbers in [-1, 1] -> smoothing filter (EMA 0.3) -> targets (3 cm, 0.1 rad, 0.1 rad per step). Control 20 Hz.
- Left/right hands come from separate URDFs (`left_sharpawave.urdf`, `right_sharpawave.urdf`) in FlashCHORD (read-only).
- Teachers (S0 policies, 64-world ADD-AUC >= 0.75), 12 total:
  - public, scored vs GT: 41 pot (1.00), 40 dust pan (0.99), 26 rack+pot (0.94), 18 sweeper+dust pan (0.86), 2 pot+lid (0.82)
  - eval, scored vs cleaned reference: 38 dust pan (0.97), 37 spoon (0.91), 35 spoon (0.89), 6 pot+spoon (0.87),
    8 beige cup+pot (0.83), 3 pot+spoon (0.80), 36 spoon (0.76)
- Weak eval episodes with object coverage: 28 rack+pot (0.08), 19 dust pan+sweeper (0.28), 5 pot+spoon (0.36),
  30 sweeper+rack (0.42), 1 pot+lid (0.48), 16 rack+spoon (0.53). Half covered: 33, 14, 10. Not covered: 20, 25.
- Reward, action space, resets, VOC gate, EMA from S0 are reused unchanged in M.

## 1. Stage map

| step | what | trains | gate to continue |
|---|---|---|---|
| P0 | infrastructure prototypes | nothing | recorder parity, G-in-loop correct and cheap, mirror tests |
| P1 | teacher dataset, labels, grasp library | PCA only | labels match video, enough hold segments, default-code selection works |
| P2a | Stage A: behaviour cloning of G (Hybrid, d only) | G | per-phase offline error, d-swap has effect |
| P3 | Stage B: Gate 1 + DAgger | G | teacher arm + G fingers >= 90% of teacher AUC |
| P2b | add learned residual r (conditional) | G + E_r | r used (KL > 0) and helps |
| P4 | Stage E: per-episode M | M (G frozen) | Gate 4: beats S0 beyond noise |
| P5 | Stage D: RL hardening of G (conditional) | G | Gate 3: fewer drops, no Gate 1 regression |
| P6 | Stage C: gap filling (conditional) | short-horizon grasp teachers | new teachers pass Gate 1 |
| P7 | Option C ablation (conditional) | G + E + prior | wins Gate 1 / drift without losing Gate 4 |
| P8 | Stage F: selection and submission | — | official kit evaluation |
| BL | pot-only shared-RL baseline (parallel) | one policy | answers "does plain shared RL work with a fair budget" |

---

## P0. Infrastructure prototypes

### P0.1 Rollout recorder
- In: teacher checkpoint + its episode scene (same Hydra overrides as its training run).
- Does: runs N worlds with the teacher; optional object-pose perturbation at reset; optional DART noise on executed actions;
  optional "finger override" (fingers from another source, e.g. G) for P3.
- Out per step, per world (npz shards, fp16 where safe): full S0 observation, raw teacher action (56), executed action,
  robot q/qd, wrist poses, object poses/velocities, contacts per hand link, per-hand wrench share s_L/s_R, object height,
  reference frame index, done flags.
- Tests:
  - parity: recorder's object trajectory metrics on the nominal setting match `evaluate_policy.py` (same checkpoint,
    64 worlds) within 64-world noise.
  - replay: re-executing the recorded executed actions open-loop in 1 world reproduces the object trajectory (target < 1 mm;
    if contact solving is non-deterministic, record the drift size; labels remain valid because they are closed-loop).
  - teacher query: the teacher can be evaluated on arbitrary recorded states (needed for DAgger labels).
- Risk: the evaluator's internals don't expose a per-step hook -> write the loop on the FlashCHORD runtime API directly.

### P0.2 G inside the simulation loop
- Two candidate implementations; the prototype picks one:
  - (a) Env wrapper: `step(m_action)` -> build G inputs from sim state -> G forward in JAX (frozen params) -> assemble the
    56-d action -> inner env step. SAC sees only M's action space, G is part of the dynamics (correct for entropy terms).
  - (b) G as Warp kernels inside a custom action module (weights loaded from file).
- Tests: G output in the loop equals an offline numpy forward on the same inputs (max abs error < 1e-4); throughput at
  2048 worlds within 15% of S0; GPU memory fits 24 GB with FlashSAC at 2048 worlds.
- Risk: (a) needs a per-step host round trip -> too slow; then (b). If both fail, Stage E is redesigned (stop here).

### P0.3 Mirroring
- Left<->right finger joint map built from the two URDFs: name pairing, axis directions, sign, limits.
- Reflection R (det -1) for everything G sees: positions/vectors `v' = R v`; object pose `T' = R T R`;
  angular velocity and rotation deltas are pseudovectors: `w' = -R w`, `dR' = R dR R` (via rotation matrices, never on rpy).
  Surface-point vectors as vectors. BPS: precompute a mirrored code per object (recomputed on the reflected mesh).
  Other-hand flag: roles swapped. Outputs un-mirrored the same way.
- Tests: map is a bijection; mirror(mirror(x)) == x for every input/output block; mirrored left joint values stay inside
  right-hand limits (report the out-of-range fraction); a rendered mirrored left grasp looks like a right grasp.
- Risk: the hands are not exact mirrors (different limits/offsets) -> clip + handedness bit as an extra G input.

### P0.4 Per-hand wrench share
- Copy of the S0 wrench-support computation (`s0_objective.py`, ours) under `gmd/`, evaluated per hand.
- Tests: on single-hand holds, the holding hand's share ~ combined support; combined support >= max per-hand share.

### P0.5 Engineering budget checks
- Disk space for ~2-4 GB of shards; measured env steps/s at 2048 worlds (sets every later budget).

**Gate to P1:** P0.1 parity + teacher query, P0.2 correct and fast enough, P0.3 tests pass.

---

## P1. Teacher dataset, labels, grasp library

### Data generation
- Per teacher, 64 worlds each:
  - nominal; object pose perturbed (+-2 cm, +-10 deg); DART noise sigma 0.1 and 0.25 on executed actions (labels = clean).
  - scaled objects (+-10-15% anisotropic) only if rebuilding scaled assets is cheap; otherwise later.
- Keep worlds that finish without object termination and with rollout AUC >= 0.7.
- Size estimate: 12 teachers x 64 worlds x ~300 frames x 4 settings x 2 hands ~ 1.8M hand-samples.

### Labels (per hand, per object; thresholds in config, in frames at 20 Hz)
1. median filter s_h over 3 frames.
2. hold start: (s_L + s_R) > 0.5 AND s_h > 0.2 for >= 4 frames; end: combined < 0.3 OR s_h < 0.1 for >= 2 frames;
   merge gaps < 3 frames; drop holds < 6 frames.
3. prehensile = rigid (object-in-wrist translation std < 1.5 cm, rotation < 8 deg over 6 frames) AND hand bears load;
   non-prehensile = contact while object slides/rotates relative to the hand with table support.
4. acquire = [contact onset - 6, hold start + 10]; contact onset = first frame of the contiguous contact/d_h < 1 cm run before the hold.
5. release = hold end until contacts lost and d_h > 2 cm (cap 10 frames).
6. approach = pre-acquire frames with d_h < 15 cm and decreasing; otherwise idle.
7. engage = 1 from contact onset - 6 frames to hold end; non-prehensile rule decided once from the data.
8. pair_id when two holds on the same object overlap > 50% of the shorter one.

### Grasp descriptor and library
- T_grasp = inv(T_wrist) * T_obj averaged over the rigid part of each hold (mirrored for left hands).
- Finger PCA: fit on all hold frames (both hands, mirrored), k chosen from the explained-variance curve (start k=5).
- d = [T_grasp position 3, rotation 6-d, PCA k], z-scored. Code per step: own hold's d; between holds the next hold's d;
  after the last hold an "open" code.
- Library entry: {object, hand, d, pair_id, teacher episode, onset frame, hand-ray wrist pose at onset}.
- Coverage table: object x grasp mode (cluster of T_grasp) x teacher episode, and which weak eval episodes each serves.

### Tests
- phase timelines overlaid on rollout videos for 3+ episodes (you review).
- hold count per episode matches the video; distribution of hold durations and drift.
- hold onsets agree with the reference pickup onsets within +-0.3 s.
- PCA: explained variance and per-joint reconstruction error (rad) for k = 3..8.
- d consistency: d of the same hold across worlds forms a tight cluster; modes of the same object separate.
- mirroring: mirrored left holds land in the same d cluster as matching right-hand holds.
- default-code selection: choosing d from the hand rays (as at deployment) picks the true hold's d; report accuracy.

**Expected / gate to P2:** labels look right in the videos; >= ~30 hold segments and >= 2 per covered object;
selection accuracy >= ~70%. Decide here whether Option C (P7) is worth running (needs several modes per object).

**Risks:** teachers exploiting the simulator (object wedged against the table, penetration) -> visually screen teachers,
drop bad holds; DART noise breaks most rollouts -> lower sigma; too few distinct grasps -> rely on Hybrid only.

---

## P2a. Stage A: behaviour cloning of G (Hybrid, d only)

### Inputs (per hand, wrist frame, mirrored for the left hand) — about 230 numbers
| block | size |
|---|---|
| target object pose (position 3 + rotation 6-d) + linear/angular velocity | 15 |
| surface vectors: 11 hand points (palm, 5 fingertips, 5 middle phalanges) to the nearest point of the target object | 33 |
| same 11 points to the nearest point of any other object (obstacle sense) | 33 |
| BPS code of the target object (mirrored code for left), dropout p=0.3 | 32 |
| finger joint positions + velocities | 44 |
| contact flags (palm + 5 fingertips) + contact force magnitudes if available | 6-12 |
| commanded wrist delta / velocity / acceleration this step | 18 |
| previous G output | 28 |
| other hand's contact flag, engage flag | 2 |
| code d (r added in P2b) | ~14 |

Target object = the object of the current or next hold (carried by the library entry).

### Outputs
- 22 finger actions in [-1, 1] (fed through the normal S0 action filter).
- 6-d wrist residual: an offset on M's wrist target, bounded by tanh to 2 cm / 0.2 rad. Teachers give no label for it,
  so in Stages A/B its target is 0 (small L2 penalty); it only becomes active in RL hardening (P5).

### Training
- Offline, JAX + optax in the FlashCHORD container, one GPU, a few hours.
- MLP 512 x 3 with LayerNorm; input normalisation from the training data, frozen.
- Batches: 40% acquire / 15% release / 35% hold / 10% approach, balanced across grasp-mode clusters.
- Loss: weighted MSE on finger actions (2x on acquire/release) + L2 on the residual.
- Split: held-out worlds (validation) + one held-out teacher episode per object where possible (generalisation).

### Tests
- per-phase validation error vs training error (overfitting check), acquire error not > 2x hold error.
- d-swap (offline): on approach/acquire frames, swap in another mode's d -> predicted finger configuration moves toward
  that mode's PCA mean. If not, G ignores d -> drop the observed object pose on approach frames in training and retrain.

**Risks:** G ignores d (pose shortcut); overfitting to 12 episodes; compounding error in closed loop (fixed by P3).

---

## P3. Stage B: Gate 1 + DAgger

- Setup: recorder in finger-override mode on each teacher episode: teacher drives the wrists, G drives the fingers;
  engage and code from the teacher's labelled timeline at the same reference frame. 64 worlds.
- Starts: 60% at 0.5 s before grasp onset, 20% before release, 20% uniform.
- DAgger: at each visited state, label = teacher's finger action (teacher query from P0.1). Aggregate, retrain G, repeat
  3-5 rounds; round 1 executes teacher fingers with probability 0.5.
- Metrics: 64-world AUC ratio (teacher arm + G) / teacher; drop rate; per-phase failure attribution.

**Gate 1:** ratio >= 0.9 on >= 9 of 12 teachers. Also record leave-one-episode-out (held-out teacher) ratios.

**Risks:**
- grasp not separable (teacher arm relies on its own finger timing) -> inspect failing phases; give G more wrist authority;
  else stay with S0.
- teacher labels meaningless on states far from its own distribution -> keep starts near grasp onset, use mixing.

## P2b. Learned residual r (only after Gate 1)
- E_r: segment summary of finger trajectories + contact pattern -> Gaussian; prior N(0, I); beta 1e-3..1e-2.
- Keep r only if KL per dim is clearly > 0 AND Gate 1 ratio or hold drift improves. Otherwise d alone.

---

## P4. Stage E: per-episode movement policy M (G frozen)

### Inputs
- S0 observation (proprioception, object state, future trajectory, BPS, hand rays, image features).
- + per hand: engage state, contact flags, wrench share s_h, current code d, frames until the next scheduled hold.

### Outputs (per hand)
- wrist deltas (S0 action space), engage (continuous, threshold with hysteresis), bounded code residual dd
  (+-3 cm / +-10 deg on T_grasp, +-0.5 sigma on PCA dims).

### Code schedule
- Per episode, before training: for each reference hold (from pickup onsets in the reference), pick the library entry
  (or pair) whose stored hand-ray wrist pose best matches this episode's hand rays at onset. M adds dd to it.

### Training
- FlashSAC, S0 rewards (object tracking 20, wrench support 10, hand rays 5 -> 0, termination -100), S0 action filter,
  2048 worlds, actor EMA, pickup resets, VOC gated on support with a small floor, best checkpoint by 64-world AUC.
- Budget: start at 30-40M steps and compare learning curves with S0 at equal steps.
- Order: 41 (regression check: should match S0 within noise), then 28, 19, 5.

**Gate 4:** M+G 64-world AUC beats S0 by more than the measured noise floor on >= 2 of the 3 weak episodes.
Noise floor = spread of 64-world AUC across 2 seeds / repeated evaluations of the same S0 policy.

**Risks:**
- M moves the wrist differently from teachers -> drops -> P5.
- engage chatters or fires late -> tune hysteresis; give engage timing a small shaping reward.
- wrong code from hand rays -> M's dd may not reach the right grasp -> allow M to switch library entries.
- eval-episode scores are vs the cleaned reference, not the hidden mocap; gains measured here are proxies.

---

## P5. Stage D: RL hardening of G (only if P4 shows drops or acceleration sensitivity)
- Env: grasp trials (3-4 s): reset to a recorded teacher pre-grasp state (+ noise), engage on, scripted wrist lift/carry
  (teacher wrist paths time-warped 0.7-1.5x + acceleration bursts, or synthetic paths), engage off, release.
  Needs a reset that restores a full recorded sim state (robot q + object pose) — check in P0.1.
- Codes: 50% teacher, 50% perturbed/interpolated (novel synthesised codes only after P6).
- Rewards: acquire = wrench support + intended-side contact, penalty if not lifted 0.5 s after the lift starts;
  hold = penalty on object-in-wrist drift, support bonus, large drop penalty; release = settle near release point,
  fingers clear in 0.5 s.
- Training: FlashSAC with a behaviour-cloning term toward teacher actions and replay seeded 50/50 with teacher transitions
  (needs a learner hook like the existing EMA hook); fresh critic + temperature; LayerNorm critic.
- Gate 3: drop rate and drift at 1.0x/1.5x improve; Gate 1 ratio does not drop. Otherwise keep the P3 G.
- Risks: RL destroys the imitation behaviour (BC weight too low); learner hooks harder than expected.

## P6. Stage C: gap filling (blue cup, water pitcher, wooden pieces)
- Short-horizon grasp teacher per object: 3 s trials from the hand-ray-implied pre-grasp, RL with support + lift + hold
  rewards and gated VOC. If that fails: force-closure grasp synthesis -> verify in sim -> use as the start.
- Successful rollouts -> P1 pipeline -> retrain G. Remaining gaps keep S0.
- Risk: these objects are the hardest (S0 also fails on them); budget-limited.

## P7. Option C ablation (only if P1 shows several grasp modes per object)
- Learned z (8-16-d) from a segment encoder; multimodal prior (mixture of 4 Gaussians, or VQ codebook).
- Same data, architecture and schedules. Compare Gate 1, drift/drop, leave-one-out, Gate 4. Hybrid wins ties.
- Risks: latent collapse (z-swap no effect); prior averages modes; too few segments to fit the prior.

## P8. Stage F: selection and submission
- Per episode: better of S0 and M+G by 64-world AUC. Official kit evaluation (1 world, frame 0). Meshes unchanged.
- Packing needs the commit URL and your approval.

## BL. Pot-only shared-RL baseline (parallel, GPU 1)
- In: episodes 41/43/44/45 concatenated (grasp_multi code), 2048 worlds (memory-tested first), gated VOC with floor 0.05-0.1,
  ~150M steps, S0 rewards.
- Out: per-episode scores, pick success, videos.
- Expected: if it lifts the pot reliably, plain shared RL works given budget and the comparison with M+G is fair.
- Risk: OOM at 2048 worlds -> 1024 worlds with a longer run.

## Overall risks
1. Gate 1 fails (grasp not separable) — the main scientific risk; costs P0-P3 (~4-6 days) before we know.
2. G-in-the-loop too slow — main engineering risk; found in P0 within ~1 day.
3. Too few distinct grasps — limits generalisation claims; Hybrid still works for covered objects.
4. Improvements measured vs cleaned references may not transfer to the hidden mocap.
