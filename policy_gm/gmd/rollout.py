"""Teacher rollouts for the mini decoupling experiment: record data, or run the teacher's wrists with other fingers.

Built on FlashCHORD's own evaluation path (same composition: no VOC, failure terms disabled, frame-0 explicit reset,
`record_object_pose_rollout` + `score_rollout`), so the reported ADD-AUC is the evaluator's. A wrapper around the
teacher's policy logs the simulator state before every step and decides what is executed.

  python rollout.py <checkpoint> <out_dir> <setting> [<setting> ...] [--worlds 64] [--g g.npz] [--mix 0.0]
settings:
  nominal        teacher as is
  perturb        object pose offset at reset (+-2 cm, +-10 deg yaw)
  dart<s>        Gaussian noise (std s) added to the executed action; labels stay the teacher's clean action
  perturb_dart<s>
  fingers_open   teacher wrists, finger actions 0 (fingers stay at the reset configuration)  [lower bound]
  fingers_g      teacher wrists, fingers from G (--g); engage from G's timeline; --mix = fraction of worlds
                 that execute the teacher's fingers instead (DAgger mixing)
  fingers_g_off  as fingers_g with engage forced off all episode (is the grasp caused by the engage bit?)
Per setting: <out_dir>/<setting>.npz (per-step arrays) and metrics in <out_dir>/<setting>.json; meta.npz once.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from omegaconf import OmegaConf

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation import harness as H
from flash_chord.evaluation.rollout import record_object_pose_rollout
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig as FlashSACTrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
from flash_chord.training.flash_sac.runner import resolve_jax_device

from gfeat import Featurizer, surface_normals
import meshmix

meshmix.install()                                           # per-world v1/v2 meshes when $GMD_MESHMIX is set


def build(checkpoint, worlds):
    metadata = read_checkpoint_metadata(Path(checkpoint))
    config = checkpoint_config(metadata)
    H.apply_evaluation_composition(config, world_count=worlds, reset_mode="explicit")
    if meshmix.active():
        config["collision"]["ground"] = False                  # parked hulls must not meet the floor plane
    config["task"]["parquet"] = H._resolve_parquet(config, None)
    cfg = OmegaConf.create(config)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    training = instantiate_typed(cfg.training, FlashSACTrainingConfig)
    setup = setup_scene(
        parquet=cfg.task.parquet, control_fps=env_config.sim.fps, motion_speed=cfg.task.motion_speed,
        embodiment=instantiate_typed(cfg.embodiment, Embodiment), collision=instantiate_typed(cfg.collision, CollisionPolicy),
        world_count=cfg.scene.world_count, include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
        object_scale_min=cfg.scene.get("object_scale_min", 1.0), object_scale_max=cfg.scene.get("object_scale_max", 1.0),
        object_scale_seed=cfg.scene.get("object_scale_seed", 0), contact_friction=cfg.scene.get("contact_friction", 1.0),
        object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
        source_frame_playback=bool(cfg.task.get("source_frame_playback", False)),
        motion_start_frame=int(cfg.task.get("motion_start_frame", 0)), motion_end_frame=int(cfg.task.get("motion_end_frame", -1)))
    env = RLEnv(setup.scene, setup.reference, config=env_config)
    env.reset()
    env.capture_step()
    jax_env = WarpRLEnv(env, publish_diagnostics=True, publish_step_metrics=True)
    if training.curriculum is not None:
        stage = training.curriculum.stages[int(metadata["curriculum_stage"])]
        jax_env.set_curriculum(H._evaluation_curriculum_stage(stage, "explicit"))
    return cfg, metadata, training, setup, env, jax_env, H.evaluation_thresholds(config)


def static_meta(cfg, setup, env):
    lay = env.action.layout
    sl = setup.scene.layout
    assert tuple(env.contact.layout.sides) == tuple(lay.sides), (env.contact.layout.sides, lay.sides)
    fr = lambda f: dict(name=f.name, body_id=int(f.body_id), pos=list(f.body_to_frame_pos), quat=list(f.body_to_frame_quat_xyzw))  # noqa: E731
    frames, fq, fd, names = {}, {}, {}, {}
    for side in lay.sides:
        hand = sl.hand(side)
        frames[side] = [fr(hand.palm_frame)] + [fr(f) for f in hand.dp_frames] + [fr(f) for f in hand.fingertip_frames]
        fq[side] = np.asarray(hand.finger_q_ids, np.int64)
        fd[side] = np.asarray(hand.finger_dof_ids, np.int64)
        names[side] = list(hand.finger_joint_names)
    ex = np.load(OmegaConf.select(cfg, "objective.extras_path") or cfg.env.objective.extras_path, allow_pickle=True)
    ref = setup.reference
    cl = env.contact.layout
    obj = env.objective
    return dict(
        sides=list(lay.sides), frames=frames, finger_q_ids=fq, finger_dof_ids=fd, finger_joint_names=names,
        hand_action_starts=np.asarray(lay.hand_action_starts), finger_counts=np.asarray(lay.finger_counts),
        hand_slot_starts=np.asarray(cl.hand_slot_starts), link_counts=np.asarray(cl.link_counts),
        slots_per_world=int(cl.slots_per_world), object_body_ids=np.asarray(cl.object_body_ids),
        surface_points=np.asarray(ex["object_surface_points"], np.float64), fps=float(ref.fps),
        surface_normals=surface_normals(np.asarray(ex["object_surface_points"], np.float64)), bps=np.asarray(ex["bps"], np.float32),
        ref_obj_pos=np.asarray(ref.object_body_pos_w(), np.float64), ref_obj_quat_wxyz=np.asarray(ref.object_body_quat_w(), np.float64),
        wrench_active=obj.arrays["wrench_active"].numpy(), wrench_dirs=obj.arrays["wrench_dirs"].numpy(),
        n_dirs=int(obj.arrays["n_dirs"]), obj_rc=obj.arrays["rc"].numpy(), wrench_friction=float(obj.config.wrench_friction),
        num_frames=int(ref.num_frames), bodies_per_world=int(env.model.body_count // env.world_count),
        **meshmix.meta_extras(env.world_count))


def mlp_forward(p, x):
    """G forward pass in numpy (see train_g.py). x (N, D) raw features -> (N, 22) actions in [-1, 1]."""
    h = (x - p["mu"]) / p["sd"]
    n = int(p["n_hidden"])
    for i in range(n):
        h = h @ p[f"W{i}"] + p[f"b{i}"]
        mu, var = h.mean(-1, keepdims=True), h.var(-1, keepdims=True)
        h = (h - mu) / np.sqrt(var + 1e-6) * p[f"g{i}"] + p[f"be{i}"]
        h = h / (1 + np.exp(-h))
    return np.tanh(h @ p[f"W{n}"] + p[f"b{n}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("out"); ap.add_argument("settings", nargs="+")
    ap.add_argument("--worlds", type=int, default=64); ap.add_argument("--g", default=None)
    ap.add_argument("--mix", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import jax
    import jax.numpy as jnp
    import warp as wp
    import s0_reset

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg, metadata, training, setup, env, jax_env, thresholds = build(args.checkpoint, args.worlds)
    meta = static_meta(cfg, setup, env)
    np.savez(out / "meta.npz", meta=np.array(meta, dtype=object))
    W, nq, nd = env.world_count, env.model.joint_coord_count // env.world_count, env.model.joint_dof_count // env.world_count
    Bw = meta["bodies_per_world"]
    sides = meta["sides"]; Hn = len(sides)
    starts, nf = meta["hand_action_starts"], meta["finger_counts"]
    wrist_idx = np.stack([np.arange(s, s + 6) for s in starts])                 # (H, 6)
    fing_idx = np.stack([np.arange(s + 6, s + 6 + n) for s, n in zip(starts, nf)])  # (H, 22)
    device = resolve_jax_device(training.device)
    pol = s0_reset._REG.get("policy")
    rng = np.random.default_rng(args.seed)
    gp = dict(np.load(args.g, allow_pickle=True)) if args.g else None
    feat = Featurizer(meta)

    for setting in args.settings:
        perturb = setting.startswith("perturb")
        sigma = float(setting.split("dart")[1]) if "dart" in setting else 0.0
        if perturb:
            orig = pol.prepare_explicit
            def prep(reset_mask, voc, _orig=orig):
                _orig(reset_mask, voc)
                xy = rng.uniform(-1, 1, (W, pol.num_objects, 3)) * np.array([0.02, 0.02, math.radians(10.0)])
                pol.perturb_flag.assign(np.ones(W, np.int32))
                pol.perturb_xyyaw.assign(xy.reshape(-1, 3).astype(np.float32))
            pol.prepare_explicit = prep
            assert pol.config.perturb_probability > 0, "the perturbation hook only runs when the reset config enables it"
        env.reset(frame_id=0)
        if perturb:
            pol.prepare_explicit = orig
        obs = jax.device_put(jnp.asarray(env.observation.numpy().reshape(W, env.observation_dim)), device)
        actor_state, teacher, key, _ = load_policy_for_inference(str(args.checkpoint), training, obs, env.action.action_dim,
                                                                 deterministic=True, device=device)
        T = meta["num_frames"]
        log = {k: [] for k in ("body_q", "joint_q", "joint_qd", "raw", "exe", "contact_active", "contact_force_w",
                               "contact_pos_o", "contact_dir_o", "wrench_score", "timestep", "g_out")}
        hist = np.zeros((3, W, Hn, 6), np.float32)
        prev_f = np.zeros((W, Hn, int(nf[0])), np.float32)
        teacher_worlds = rng.random(W) < args.mix
        feat.reset()
        if gp is not None:
            eng_t = gp["engage_timeline"]                                   # (T, H)

        def wrapped(state, observation, k):
            a, k = teacher(state, observation, k)
            raw = np.asarray(a, np.float32).reshape(W, -1)
            bq = env.state_0.body_q.numpy().reshape(W, Bw, 7)
            jq = env.state_0.joint_q.numpy().reshape(W, nq)
            jqd = env.state_0.joint_qd.numpy().reshape(W, nd)
            ca = env.contact.contact_active.numpy().reshape(W, -1)
            cf = env.contact.contact_force_w.numpy().reshape(W, -1, 3)
            ts = env.timestep.numpy().copy()
            exe = np.clip(raw, -1, 1).copy()
            hist[2], hist[1] = hist[1], hist[0]
            hist[0] = np.clip(raw[:, wrist_idx], -1, 1)
            g_out = np.zeros((W, Hn, int(nf[0])), np.float32)
            if setting == "fingers_open":
                exe[:, fing_idx.reshape(-1)] = 0.0
            elif setting in ("fingers_g", "fingers_g_off"):
                eng = eng_t[np.clip(ts, 0, len(eng_t) - 1)] * (0.0 if setting == "fingers_g_off" else 1.0)
                x = feat(bq, jq, jqd, ca, cf, hist, prev_f, eng)
                g_out = mlp_forward(gp, x.reshape(W * Hn, -1)).reshape(W, Hn, -1).astype(np.float32)
                if "off_W0" in gp:                                           # gated G: off-net while engage is off
                    q = {k[4:]: v for k, v in gp.items() if k.startswith("off_")}
                    g_off = mlp_forward(q, x.reshape(W * Hn, -1)).reshape(W, Hn, -1).astype(np.float32)
                    g_out = np.where(eng[..., None] > 0.5, g_out, g_off)
                for h in range(Hn):
                    exe[~teacher_worlds, fing_idx[h][0]:fing_idx[h][-1] + 1] = g_out[~teacher_worlds, h]
            if sigma > 0:
                exe = np.clip(exe + rng.normal(0, sigma, exe.shape), -1, 1).astype(np.float32)
            prev_f[:] = exe[:, fing_idx]
            for name, v in (("body_q", bq), ("joint_q", jq), ("joint_qd", jqd), ("raw", raw), ("exe", exe),
                            ("contact_active", ca.astype(np.int8)), ("contact_force_w", cf.astype(np.float16)),
                            ("contact_pos_o", env.contact.contact_pos_o.numpy().reshape(W, -1, 3).astype(np.float16)),
                            ("contact_dir_o", env.contact.contact_force_direction_o.numpy().reshape(W, -1, 3).astype(np.float16)),
                            ("wrench_score", env.objective.wrench_score.numpy().reshape(W, -1)), ("timestep", ts), ("g_out", g_out)):
                log[name].append(v)
            return jax.device_put(jnp.asarray(exe), device), k

        rollout = record_object_pose_rollout(env, jax_env, wrapped, actor_state, key, obs, reference=setup.reference,
                                             step_count=T, start_frame=0, progress_interval=100)
        metrics = H.score_rollout(rollout, thresholds)
        m = {}
        for f in dataclasses.fields(metrics):
            v = getattr(metrics, f.name)
            if isinstance(v, (int, float, np.floating, np.integer)):
                m[f.name] = float(v)
        m.update(setting=setting, worlds=W, mix=args.mix, g=args.g)
        name = setting + args.tag
        np.savez_compressed(out / f"{name}.npz", teacher_worlds=teacher_worlds, **{k: np.stack(v) for k, v in log.items()})
        json.dump(m, open(out / f"{name}.json", "w"), indent=1)
        print(f"[{name}] add_auc={m.get('add_auc')} " + " ".join(f"{k}={v:.4g}" for k, v in m.items() if isinstance(v, float)), flush=True)


if __name__ == "__main__":
    main()
