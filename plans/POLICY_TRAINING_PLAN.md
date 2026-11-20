# Faster grasping-policy training with a learned differentiable world model (after SuperMimic)

Plan for a coding agent. Written 2026-11-19. Nothing here has been built yet. Read `CLAUDE.md`, `NOTES.md`,
`policy_s0/README.md` and `policy_gm/README.md` first.

## 1. Why

Each Track 3 episode trains its own FlashCHORD / FlashSAC policy (the S0 recipe in `policy_s0/`):
- 96.8M environment steps per episode, at about 16-27k steps/s on one RTX 4090 with 2048 worlds (20-24 GB).
- That is roughly 1-1.7 GPU-hours per episode, about 20-34 GPU-hours for the 20 evaluation episodes, and nothing
  carries over from one episode to the next.

The final pipeline should turn many videos into policies (`plans/RECONSTRUCTION_PLAN.md`), so training cost will grow
with the number of clips.

SuperMimic (Liao et al., CoRL 2026, arXiv 2610.28480, code `github.com/HybridRobotics/supermimic`, Apache-2.0) trains
one humanoid motion tracker for 129,514 clips:
- On one H100 in 3 h, it beats a PPO tracker trained for 7 days on 128 GPUs (more than 1000x fewer GPU-hours).
- The loop is Dyna-style:
  1. Collect simulator transitions (MuJoCo Warp) into a replay buffer.
  2. Fit one differentiable world model to 8-step windows of them.
  3. Improve a deterministic policy by backpropagation through time over 32 imagined steps, with a critic that
     bootstraps the return beyond the horizon (short-horizon actor-critic, as in SHAC).

The paper's ablations attribute the gain to three things:
- **Rollout speed:** imagined rollouts run more than 10x faster than MuJoCo Warp (up to 62x at 64k parallel rollouts).
- **First-order gradients:** about 64x fewer imagined steps than PPO trained on the same world model.
- **Amortization:** one world model for all motions. Its 8-step error stays at 0.007-0.011 from 1 to 264 h of data.

Two more results matter for us:
- Without the critic, training fails at horizons of 16 steps or fewer.
- The authors list "no terrain or object interaction" as a limitation.

## 2. What transfers, and what does not

| SuperMimic ingredient | Our setting | Expectation |
|---|---|---|
| Learned world model fit on simulator logs; the simulator is never differentiated | FlashCHORD runs MuJoCo-Warp (via Newton); transitions can be logged the same way | Transfers |
| BPTT through short imagined rollouts + critic | Same reward structure (dense tracking reward, S0 objective) | Transfers |
| One model amortized across motions | Our episodes differ in object mesh and contact, not only in motion | Needs object conditioning (shape code); the model must learn hand-object contact |
| Dense, smooth dynamics (whole-body motion, floor contact) | Grasping: stiff, intermittent hand-object contact, object pose driven by contact | **Main risk.** This is exactly the limitation the authors list |
| Baseline is PPO | Our baseline is FlashSAC (off-policy, replay buffer), already far more sample-efficient than PPO | Expect a much smaller gain than 1000x. Measure it; do not assume it |
| Generalist policy | `policy_gm/` tried a shared grasp policy G (behaviour cloned from S0 teachers). With the teacher's own wrists G kept 87-101% of the teacher's score on 15 of 17 teachers, but M + G stayed below S0, and G transferred poorly to unseen meshes (dust pan 22% vs 99%) | Sharing the **dynamics model** is safer than sharing the **policy**. Start with per-episode actors on a shared model |

## 3. Hard constraints

1. **Checkpoints must load in the official evaluator, unmodified** (`flash_chord/scripts/evaluate_policy.py` at
   `8c778f9`).
   - `flash_chord/evaluation/harness.py::evaluate_checkpoint` rebuilds the experiment from the checkpoint's saved config
     and supports only `flash_sac` and `ppo` checkpoints, with one `task.parquet` per checkpoint.
   - So the trained actor must use the same network class, observation layout and action space as the S0 FlashSAC (or
     PPO) actor, and it must be written in that checkpoint format, with a config the evaluator accepts.
   - Prove this export path first (W2.1), before any world-model work pays off.
2. Do not edit the evaluator or pinned upstream code. Our code lives in this repository (for example `policy_wm/`).
3. Every checkpoint carries a provenance record (`e2s/provenance.py`; `tools/check_provenance.py`). `FINAL=1` refuses
   non-video references, exactly as for S0.
4. riddle: 2x RTX 4090 with 24 GB each. Use a GPU only when it is completely free. CUDA 12.9 at most (JAX `cuda12`).
   Store everything under `~/v2d` or `/usr1/e2s/rborah_v2d` (/usr0 is nearly full). Before any run over 30 min, state
   the command, GPU and duration, and wait for a yes. Never sudo.
5. Framework: FlashCHORD is JAX; SuperMimic is PyTorch on mjlab.
   - Preferred: implement the world model and the BPTT update in JAX inside our FlashCHORD-based code. That reuses the
     environment, the S0 observation, reward and reset code, and the checkpoint format.
   - Reuse SuperMimic's design, not its code.
   - Only if JAX blocks progress: train in PyTorch on logged data, then convert the actor weights into the FlashSAC
     parameter layout, and verify it with the evaluator.

## 4. Measurement protocol (fixed before any experiment)

- **Dev episodes:** 3, chosen to span the hard cases:
  - one plain pick-and-place
  - one cup or pour (fast in-hand rotation)
  - one pot-lid or tool episode (S0's hardest wrist motion)

  Hold out 3 more episodes for the end-of-phase check only.
- **Score:** the official evaluator's metrics on the episode's own reference (ADD-AUC, as in S0 checkpoint
  selection), plus SP-SR/MP-SR. Use the same reference version and mesh as S0.
- **Cost:** wall-clock and GPU-hours to reach 95% of S0's final score on that episode. Include data collection,
  model fitting and policy updates; checkpoint evaluation is excluded and reported separately.
- **Baseline:** S0 exactly as in `policy_s0/` (same reference, curriculum, resets), re-run on the same GPU type, at
  least 2 seeds.
- **Logging:** log every run in `NOTES.md`, with a row in a results table (`policy_wm/results.csv`: commit, episode,
  method, steps, GPU-h, score).

## 5. Phases and gates

### W0 - Baseline and logging (no new learning code)
- Re-measure S0 on the 3 dev episodes: score against environment steps and against wall-clock (the curves come from
  training checkpoints).
- Transition data. Past runs did not keep transitions: `state_*.safetensors` holds networks and optimizer state, no
  replay buffer (checked 2026-11-19). They did keep policy checkpoints at several points in training
  (`~/v2d/runs/*/policy_<step>.safetensors`, 38 runs, and the S0 checkpoints of all 20 evaluation episodes).
  - Generate the data by rolling those checkpoints out in FlashCHORD. Use early, middle and final checkpoints, add
    action noise, and use the S0 perturbed / pickup resets, so the data covers more than the reference neighbourhood.
  - Log state, action, next state, reward, termination, sim context (friction, mass, randomization parameters) and
    the episode's object id. Store fp16 on /usr1, and estimate the size per transition first.
  - Human video and mocap cannot train this model: they contain no robot actions. They stay what they are: references
    and object geometry.
- **Gate:** baseline curves and about 50-100M logged transitions per dev episode on disk. Check the disk budget
  first.

### W1 - Can a learned model predict hand-object dynamics? (offline, decides go/no-go)
- **Model state:** the S0 simulator state (wrist poses and joints of both hands; object pose and velocity), with
  contact features computed from it. Inputs: action and sim context. Heads: next state, reward, termination,
  observation (as in SuperMimic eq. 1, 8-step autoregressive loss).
- Start with one model per episode, then one model for all 3 dev episodes, conditioned on an object shape code
  (point-cloud encoding of the mesh, or per-episode embedding first).
- Report 8-step and 32-step prediction error, split by phase:
  - free-space reach
  - first contact
  - object in hand
  - release
- For each phase, report object position (mm), object rotation (deg) and hand joint error.
- Starting gate values (revise once the baseline variance is known):
  - 8-step object error under 5 mm and 3 deg in the in-hand phase
  - the shared 3-episode model within 1.5x of the per-episode models
  - model rollout throughput at least 5x the simulator's at 2048-8192 parallel rollouts on a 4090
- **If the in-hand gate fails, stop here** and go to W4. Do not build the policy loop on a model that cannot hold an
  object.

### W2 - BPTT actor-critic on one episode
- **W2.1 Export first.** Take an S0 actor's weights, write them back through our own checkpoint writer, and confirm
  the official evaluator gives the same score. Then do the same with a randomly initialised actor from our code. This
  pins the network class, observation normalisation and config fields the evaluator needs.
- **W2.2 The loop:** SuperMimic's three phases in JAX.
  - Collect with action noise; fit the model on 8-step windows; update the actor and critic on 32-step imagined
    rollouts started from replay states.
  - Soft survival weight instead of hard termination; GAE targets for the critic.
  - Use the S0 reward, observation and resets. The actor is deterministic and uses the S0 actor's architecture.
  - Sweep the horizon H at 16 / 32 / 64, with the critic (the paper: no critic fails at H <= 16).
- **Gate:** on the first dev episode, reach S0's final score with at most one third of S0's GPU-hours, and the
  exported checkpoint scores the same in the official evaluator.

### W3 - Amortize across episodes
- One world model for all training episodes, conditioned on object shape. Per-episode actors are warm-started from a
  shared actor, then fine-tuned on their own reference.
- Only then try one generalist actor with the reference as its goal. Its per-episode checkpoints would be the same
  weights with that episode's parquet in the config, if W2.1 shows the evaluator accepts this.
- **Gate:** total GPU-hours for all 20 evaluation episodes at most half of S0's, with every episode within 0.02
  ADD-AUC of S0. Check the 3 held-out episodes before claiming it.

### W4 - Fallbacks if contact cannot be modelled well enough
Try in this order, cheapest first:
1. **Decoupled gradients** (DMO-style, cited by SuperMimic): roll out in the simulator and backpropagate only
   through the learned model. Rollouts are no faster, but the gradients are first-order, and model error does not
   compound over the rollout.
2. **Model the hands only:** learn the hand and wrist dynamics. Object motion comes from the virtual object controller
   (`policy_gm/voc/`) or a simple analytic contact model inside the imagined rollout.
3. **World model as warm start:** pre-train the actor through the model for a few minutes, then finish with FlashSAC
   in the simulator. Accept it if the time to S0's score drops by at least 30%.

Recommendation if all of W4 fails: keep S0. Training cost stays roughly 20-34 GPU-hours per 20 episodes, which is not
the project's bottleneck today. Reconstruction accuracy is: object trajectories are still 5.6 cm off.

## 6. Embodiments

- Floating Sharpa hands (the Track 3 default) first, because that is what is scored.
- The Kinova arm retarget (`out/kinova_ep41`) has a fixed base and fewer contact modes than two floating dexterous
  hands. It may be an easier first test of the in-hand gate in W1. Run it only as a side experiment, with its own
  numbers.

## 7. Risks and how to notice them

| Risk | Signal |
|---|---|
| Model drifts during contact; the policy exploits model errors (imagined score rises, simulator score does not) | Imagined vs simulator return on the same rollouts, logged every iteration |
| Gains vs FlashSAC are small (FlashSAC is already off-policy) | W2 gate on GPU-hours; stop if not met |
| Evaluator cannot load our checkpoints | W2.1, done before any learning work |
| Shared model fails on new meshes (as G did in `policy_gm`) | Held-out episodes in W3 |
| Memory: model + 2048-8192 imagined rollouts + BPTT through 32 steps on 24 GB | Peak VRAM logged per run. Reduce rollouts or horizon before reducing model size |
| Mocap leaks into final training | Provenance gate (`FINAL=1`) unchanged |

## 8. Order of work

1. W0 baseline curves and logging.
2. W2.1 checkpoint export round trip. It is independent of W1, so it can run in parallel.
3. W1 world-model gate. **Stop and report to the user whatever the result.**
4. W2.2, then W3, each followed by a results row and a `NOTES.md` entry. Stop for review whenever a gate fails or a
   number moves the wrong way.
