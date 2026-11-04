"""Score M checkpoints as they appear (64 worlds, FlashCHORD evaluation composition and scorer, G in the loop) and
apply the stopping rules. Runs next to training in its own container; writes STOP for the host watcher.

  eval_m.py <run_dir> [--worlds 64] [--success 0.9 --success_n 3] [--band 0.05 --band_n 5 --band_after 16777216]
Rules (checkpoints in step order): stop when the last `success_n` scores are >= `success`; or, once past `band_after`
steps, when the last `band_n` scores lie within `band` of each other and (with --nobest_n N > 0) none of the last N
checkpoints set a new best (converged, not still climbing). Training has its own hard cap.
Writes <run_dir>/evals.jsonl, and <run_dir>/STOP (reason) when a rule fires. Exits when STOP exists or DONE appears and
every checkpoint is scored.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/env/gmd")
import numpy as np  # noqa: E402

import rollout  # noqa: E402  (build() mirrors the evaluator's environment)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--success", type=float, default=0.9); ap.add_argument("--success_n", type=int, default=3)
    ap.add_argument("--band", type=float, default=0.05); ap.add_argument("--band_n", type=int, default=5)
    ap.add_argument("--band_after", type=int, default=16777216)
    ap.add_argument("--nobest_n", type=int, default=0)
    args = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from flash_chord.evaluation import harness as H
    from flash_chord.evaluation.rollout import record_object_pose_rollout
    from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
    from flash_chord.training.flash_sac.runner import resolve_jax_device
    import flash_chord.training.environment as fenv
    from genv import GEnv

    run = Path(args.run_dir); ck_dir = run / "run"; log = run / "evals.jsonl"
    done = {json.loads(l)["step"] for l in open(log)} if log.exists() else set()
    scores = [json.loads(l) for l in open(log)] if log.exists() else []
    built = None
    while True:
        cks = sorted((int(re.findall(r"policy_(\d+)", p.name)[0]), p) for p in ck_dir.glob("policy_*.safetensors"))
        todo = [(s, p) for s, p in cks if s not in done]
        if not todo:
            if (run / "STOP").exists() or (run / "DONE").exists():
                break
            time.sleep(15); continue
        step, path = todo[0]
        time.sleep(3)                                   # let the checkpoint finish writing
        if built is None:
            orig = fenv.WarpRLEnv
            rollout.WarpRLEnv = GEnv                    # build() wraps the env with G
            built = rollout.build(str(path), args.worlds)
            rollout.WarpRLEnv = orig
        cfg, metadata, training, setup, env, genv, thresholds = built
        env.reset(frame_id=0)
        genv._reset_state(jnp.ones(genv.world_count, bool))
        _, _, _, ca, _ = genv._live()
        base = jnp.asarray(env.observation.numpy().reshape(env.world_count, env.observation_dim))
        device = resolve_jax_device(training.device)
        obs = jax.device_put(jnp.concatenate([base, genv._extra(ca, genv._eng)], 1), device)
        actor_state, policy, key, _ = load_policy_for_inference(str(path), training, obs, genv.action_dim, deterministic=True, device=device)
        ro = record_object_pose_rollout(env, genv, policy, actor_state, key, obs, reference=setup.reference,
                                        step_count=setup.reference.num_frames, start_frame=0, progress_interval=0)
        m = H.score_rollout(ro, thresholds)
        row = {"step": step, "add_auc": float(m.add_auc), "mppe_cm": float(m.mppe_cm), "chord_sr": float(m.chord_sr), "time": time.time()}
        scores.append(row); done.add(step)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[eval] {step / 1e6:.1f}M add_auc {row['add_auc']:.3f} mppe {row['mppe_cm']:.1f} cm", flush=True)
        a = [r["add_auc"] for r in sorted(scores, key=lambda r: r["step"])]
        reason = None
        if len(a) >= args.success_n and min(a[-args.success_n:]) >= args.success:
            reason = f"success: last {args.success_n} checkpoints >= {args.success} ({a[-args.success_n:]})"
        elif (step >= args.band_after and len(a) >= args.band_n and max(a[-args.band_n:]) - min(a[-args.band_n:]) <= args.band
              and (args.nobest_n <= 0 or (len(a) > args.nobest_n and max(a[-args.nobest_n:]) <= max(a[:-args.nobest_n])))):
            reason = f"converged: last {args.band_n} checkpoints within {args.band} ({[round(x, 3) for x in a[-args.band_n:]]})"
        if reason and not (run / "STOP").exists():
            (run / "STOP").write_text(f"{step} {reason}\n")
            print(f"[eval] STOP at {step}: {reason}", flush=True)
    print("[eval] finished", flush=True)


if __name__ == "__main__":
    main()
