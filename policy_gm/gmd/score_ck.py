"""Score one M checkpoint with G in the loop (as eval_m.py: evaluation composition, official scorer).
  GMD_G=<g.npz> [GMD_DEBOUNCE=K] score_ck.py <checkpoint> <out.json> [--worlds 64]"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/env/gmd")
import rollout  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("checkpoint"); ap.add_argument("out"); ap.add_argument("--worlds", type=int, default=64)
    args = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from flash_chord.evaluation import harness as H
    from flash_chord.evaluation.rollout import record_object_pose_rollout
    from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
    from flash_chord.training.flash_sac.runner import resolve_jax_device
    from genv import GEnv
    rollout.WarpRLEnv = GEnv
    cfg, metadata, training, setup, env, genv, thresholds = rollout.build(args.checkpoint, args.worlds)
    env.reset(frame_id=0); genv._reset_state(jnp.ones(genv.world_count, bool))
    _, _, _, ca, _ = genv._live()
    obs = jnp.concatenate([jnp.asarray(env.observation.numpy().reshape(env.world_count, env.observation_dim)), genv._extra(ca, genv._eng)], 1)
    device = resolve_jax_device(training.device); obs = jax.device_put(obs, device)
    actor, policy, key, _ = load_policy_for_inference(args.checkpoint, training, obs, genv.action_dim, deterministic=True, device=device)
    ro = record_object_pose_rollout(env, genv, policy, actor, key, obs, reference=setup.reference, step_count=setup.reference.num_frames, start_frame=0, progress_interval=0)
    m = H.score_rollout(ro, thresholds)
    row = {"checkpoint": args.checkpoint, "g": os.environ.get("GMD_G"), "debounce": int(os.environ.get("GMD_DEBOUNCE", "0")),
           "add_auc": float(m.add_auc), "mppe_cm": float(m.mppe_cm)}
    json.dump(row, open(args.out, "w"), indent=1)
    print(f"[score] {row}", flush=True)


if __name__ == "__main__":
    main()
