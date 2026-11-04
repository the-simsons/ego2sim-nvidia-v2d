"""G grip test: score an M + G checkpoint as is ("m") and with one hand's wrist scripted onto a given grip ("grip",
grip_ctrl.make_hybrid). Official scorer, 16 worlds each; logs per-object error and the resulting grip of the scripted hand.
  GMD_G=<g.npz> grip_test.py <M checkpoint> <grip params .npz> <out.json> [--worlds 16] [--place F]
--place F: both modes start at reference frame F; in "grip" the scripted hand is first teleported onto the grip
(grip_ctrl.place_hand) and the policy uses t_grasp = F + 3 with a 2 cm approach.
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "/env/gmd")
import numpy as np  # noqa: E402

import rollout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("params"); ap.add_argument("out"); ap.add_argument("--worlds", type=int, default=16); ap.add_argument("--place", type=int, default=-1)
    args = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from flash_chord.evaluation import harness as H
    from flash_chord.evaluation.rollout import record_object_pose_rollout
    from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
    from flash_chord.training.flash_sac.runner import resolve_jax_device
    import genv as G
    import grip_ctrl

    rollout.WarpRLEnv = G.GEnv
    cfg, metadata, training, setup, env, genv, thresholds = rollout.build(args.checkpoint, args.worlds)
    W = env.world_count; T = int(setup.reference.num_frames); device = resolve_jax_device(training.device)
    params = dict(np.load(args.params)); h = int(params["hand"])
    F = max(args.place, 0)
    if args.place >= 0:
        params = {k: v for k, v in params.items() if k not in ("rel_p_t", "rel_R_t", "eng_t", "t_switch", "t_follow0")}
        params.update(t_grasp=np.array(F + 3), approach=np.array(0.02))
    ids = env.command.body_ids_w.numpy().reshape(W, -1)
    ref = np.asarray(setup.reference.object_body_pos_w())
    contact_slots = (int(genv._meta["hand_slot_starts"][h]), int(genv._meta["link_counts"][0]))
    out = {}
    for mode in ("m", "grip"):
        env.reset(frame_id=F); genv._reset_state(jnp.ones(W, bool))
        if mode == "grip" and args.place >= 0:
            grip_ctrl.place_hand(env, genv, params)
        _, _, _, ca, _ = genv._live()
        obs = jax.device_put(jnp.concatenate([jnp.asarray(env.observation.numpy().reshape(W, env.observation_dim)), genv._extra(ca, genv._eng)], 1), device)
        actor, m_policy, key, _ = load_policy_for_inference(args.checkpoint, training, obs, genv.action_dim, deterministic=True, device=device)
        pol = grip_ctrl.make_hybrid(env, genv, setup, m_policy, params) if mode == "grip" else m_policy
        log = {"pos": [], "touch": []}

        def logged(actor_, obs_, key_, pol=pol, log=log):
            log["pos"].append(env.state_0.body_q.numpy()[ids.reshape(-1), :3].reshape(W, -1, 3))
            c = env.contact.contact_active.numpy().reshape(W, -1); s, L = contact_slots
            log["touch"].append(c[:, s:s + L].max(1))
            return pol(actor_, obs_, key_)

        ro = record_object_pose_rollout(env, genv, logged, actor, key, obs, reference=setup.reference, step_count=T - F, start_frame=F, progress_interval=0)
        m = H.score_rollout(ro, thresholds)
        P = np.stack(log["pos"]); f = np.maximum(np.arange(P.shape[0]) - 1, 0) + F
        e = np.linalg.norm(P - ref[f][:, None], axis=-1)
        out[mode] = {"add_auc": float(m.add_auc), "mppe_cm": float(m.mppe_cm),
                     "median_err_cm_per_object": [round(100 * float(np.median(e[:, :, b])), 2) for b in range(e.shape[2])],
                     "final_err_cm_per_object": [round(100 * float(np.median(e[-1, :, b])), 2) for b in range(e.shape[2])],
                     "scripted_hand_touch_frac_after_grasp": round(float(np.stack(log["touch"])[max(int(params["t_grasp"]) - F, 0):].mean()), 3)}
        print(f"[grip_test] {mode}: {out[mode]}", flush=True)
        if mode == "grip" and hasattr(pol, "__closure__"):
            cells = []
            for c in pol.__closure__ or ():
                try:
                    cells.append(c.cell_contents)
                except ValueError:
                    pass
            st = [c for c in cells if isinstance(c, dict) and "log" in c][0]
            for t_, e_, r_ in st["log"][::10]:
                print(f"[grip_dbg] t {t_:3d} palm err {100 * e_:5.1f} cm  object in palm frame {np.round(100 * r_, 1)} cm (target {np.round(100 * params['rel_p'], 1)})", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
