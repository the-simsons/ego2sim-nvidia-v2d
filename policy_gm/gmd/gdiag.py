"""Diagnostic rollout of an M + G checkpoint: logs G's exact per-hand input every step (same jax features as genv.GEnv),
plus the target object per hand, the engage switch, per-object contacts, G's finger output and object / reference poses.
Evaluation composition (as eval_m.py), no training. Nothing is changed in the policy or the environment.

  GMD_G=<g.npz> gdiag.py <M checkpoint> <out.npz> [--worlds 16]
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/env/gmd")
import numpy as np  # noqa: E402

import rollout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("out"); ap.add_argument("--worlds", type=int, default=16)
    args = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
    from flash_chord.training.flash_sac.runner import resolve_jax_device
    import genv as G
    from gfeat import features

    rollout.WarpRLEnv = G.GEnv
    cfg, metadata, training, setup, env, genv, thresholds = rollout.build(args.checkpoint, args.worlds)
    W, H, meta = genv._W, genv._H, genv._meta
    T = int(setup.reference.num_frames)
    env.reset(frame_id=0)
    genv._reset_state(jnp.ones(W, bool))
    _, _, _, ca, _ = genv._live()
    obs = jnp.concatenate([jnp.asarray(env.observation.numpy().reshape(W, env.observation_dim)), genv._extra(ca, genv._eng)], 1)
    device = resolve_jax_device(training.device)
    obs = jax.device_put(obs, device)
    actor, policy, key, _ = load_policy_for_inference(args.checkpoint, training, obs, genv.action_dim, deterministic=True, device=device)
    gp = dict(np.load(__import__("os").environ["GMD_G"], allow_pickle=True)); n = int(gp.pop("n_hidden")); gp.pop("engage_timeline", None); gp.pop("off_n_hidden", None)
    gp = {k: jnp.asarray(v, jnp.float32) for k, v in gp.items()}
    hs, L, B = [int(v) for v in meta["hand_slot_starts"]], int(meta["link_counts"][0]), len(meta["object_body_ids"])

    @jax.jit
    def feat(bq, jq, jqd, ca, cf, a, state):
        """G's input for this step, exactly as genv's pre() builds it (same state update)."""
        hist, prev_f, eng, prev_R, prev_p, target, fresh = state
        a = jnp.clip(a.reshape(W, H, 7), -1, 1)
        wrist, e = a[..., :6], a[..., 6]
        eng = jnp.where(e > G.ON, 1.0, jnp.where(e < G.OFF, 0.0, eng))
        hist = jnp.stack([wrist, hist[0], hist[1]])
        R_now = jnp.stack([G.quat_xyzw_to_mat(bq[:, b, 3:], jnp) for b in meta["object_body_ids"]], 1)
        p_now = jnp.stack([bq[:, b, :3] for b in meta["object_body_ids"]], 1)
        prev_R = jnp.where(fresh[:, None, None, None], R_now, prev_R); prev_p = jnp.where(fresh[:, None, None], p_now, prev_p)
        X, _, _, tgt = features(jnp, meta, bq, jq, jqd, ca, cf, hist, prev_f, eng, prev_R, prev_p, target, fresh)
        g = G.g_gated(gp, X.reshape(W * H, -1), n, eng.reshape(-1)).reshape(W, H, -1)
        cobj = jnp.stack([jnp.stack([ca[:, hs[h] + b * L:hs[h] + (b + 1) * L].max(1) for b in range(B)], 1) for h in range(H)], 1)
        return X, tgt, eng, g, cobj

    log = {k: [] for k in ("X", "target", "engage", "g", "contact_obj", "obj_pos", "engage_cmd")}
    ids = env.command.body_ids_w.numpy().reshape(W, -1)
    for t in range(T - 1):
        a, key = policy(actor, obs, key)
        bq, jq, jqd, ca, cf = genv._live()
        X, tgt, eng, g, cobj = feat(bq, jq, jqd, ca, cf, a, genv._state)
        log["X"].append(np.asarray(X, np.float16)); log["target"].append(np.asarray(tgt, np.int8))
        log["engage"].append(np.asarray(eng, np.int8)); log["g"].append(np.asarray(g, np.float16))
        log["contact_obj"].append(np.asarray(cobj, np.int8))
        log["engage_cmd"].append(np.asarray(a).reshape(W, H, 7)[..., 6].astype(np.float16))
        log["obj_pos"].append(env.state_0.body_q.numpy()[ids.reshape(-1), :3].reshape(W, -1, 3))
        obs = genv.step(a).observation
    ref = np.asarray(setup.reference.object_body_pos_w())
    np.savez_compressed(args.out, ref_obj_pos=ref, sides=np.array(meta["sides"]), **{k: np.stack(v) for k, v in log.items()})
    print(f"[gdiag] {args.checkpoint}: {T - 1} steps x {W} worlds -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
