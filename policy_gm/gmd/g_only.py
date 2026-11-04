"""G by itself: the wrists track the human reference, G drives the fingers. Same evaluation composition and scorer as
eval_m.py / rollout.py, 64 worlds.

The S0 action is reference-free (s0_action.py: integrated deltas on the robot's own targets, EMA 0.3), so "follow the
reference" is a tracking controller: every step it outputs the delta that puts the wrist target on the reference wrist
pose of the next frame (inverting the EMA; |a| <= 1 caps it at 3 cm / 0.1 rad per step).

  g_only.py <checkpoint (for the episode's config)> <out_dir> <setting> [--worlds 64]
settings:
  track_ref      wrists and fingers tracked to the reference: replay of the retargeted human motion
  track_open     wrists tracked, fingers held at the reset pose (action 0)
  track_g_gt     wrists tracked, G fingers, engage from the human contact labels in the reference parquet
  track_g_teach  wrists tracked, G fingers, engage from the S0 teacher's timeline (<out_dir>/../../ep<N>/engage_timeline.npy)
Human engage per hand: >= 2 MANO links touching an object; gaps < 6 frames filled, runs < 4 frames dropped, then switched
on 6 frames before contact (as in G's training labels). Writes <out_dir>/<setting>.json and <out_dir>/engage_gt.npy.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/env/gmd")
import numpy as np  # noqa: E402

import rollout  # noqa: E402


def runs(x):
    """(start, end) of the True runs of a 1-d bool array, end exclusive."""
    d = np.diff(np.concatenate([[0], x.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def human_engage(parquet_dir, sides, T, min_links=2, gap=6, min_run=4, lead=6):
    import pyarrow.parquet as pq
    f = parquet_dir if os.path.isfile(parquet_dir) else sorted(glob.glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True))[0]
    row = pq.read_table(f, columns=[f"mano_{s}_object_contact_part_ids" for s in sides]).to_pylist()[0]
    out = np.zeros((T, len(sides)), np.float32)
    for h, s in enumerate(sides):
        c = (np.asarray(row[f"mano_{s}_object_contact_part_ids"]) > 0).sum(1)[:T] >= min_links
        for a, b in runs(~c):                           # fill short gaps between contacts
            if a > 0 and b < len(c) and b - a < gap:
                c[a:b] = True
        for a, b in runs(c):
            if b - a >= min_run:
                out[max(0, a - lead):b, h] = 1.0
    return out


def make_tracker(env, ref, sides, T, track_fingers=False):
    """Controller for the reference-free S0 action: returns (track() -> (W, 56) action, track_err list)."""
    W = env.world_count; Hn = len(sides)
    A = env.action; arr = A.arrays; ema = float(A.ema)
    stp = arr["step"].numpy(); starts = list(A.layout.hand_action_starts); nf = list(A.layout.finger_counts)
    pq_ids = arr["wrist_pos_q_ids"].numpy().reshape(Hn, 3); oq = arr["wrist_quat_q_ids"].numpy()
    fq = arr["finger_q_ids"].numpy(); hfs = arr["hand_finger_starts"].numpy()
    ref_p = np.stack([np.asarray(ref.wrist_pos_w(s)) for s in sides], 1)                     # (T, H, 3)
    ref_q = np.stack([np.asarray(ref.wrist_quat_w(s))[:, [1, 2, 3, 0]] for s in sides], 1)   # wxyz -> xyzw
    ref_f = [np.asarray(ref.finger_joint_pos(s)) for s in sides]                             # (T, nf) per hand
    nq = env.model.joint_coord_count // W
    track_err = []

    def qmul(a, b):
        ax, ay, az, aw = np.moveaxis(a, -1, 0); bx, by, bz, bw = np.moveaxis(b, -1, 0)
        return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                         aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)

    def rotvec(q):
        q = np.where(q[..., 3:] < 0, -q, q)
        v = q[..., :3]; n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v * (2 * np.arctan2(n, q[..., 3:]) / np.maximum(n, 1e-9))

    def track():
        """(W, 56) action that moves each target onto the reference of the next frame."""
        f = np.clip(env.timestep.numpy() + 1, 0, T - 1)
        jq = env.state_0.joint_q.numpy().reshape(W, nq)
        track_err.append(np.mean([np.linalg.norm(jq[:, pq_ids[h]] - ref_p[f - 1, h], axis=-1).mean() for h in range(Hn)]))
        if os.environ.get("GO_DEBUG") and int(f[0]) % 20 == 1:
            tp0 = arr["target_pos"].numpy().reshape(W, Hn, 3)
            print(f"[dbg] t {int(f[0]) - 1} actual-ref {[round(100 * float(np.linalg.norm(jq[:, pq_ids[h]] - ref_p[f - 1, h], axis=-1).mean()), 1) for h in range(Hn)]} cm, "
                  f"target-ref {[round(100 * float(np.linalg.norm(tp0[:, h] - ref_p[f - 1, h], axis=-1).mean()), 1) for h in range(Hn)]} cm, "
                  f"ref speed {[round(100 * float(np.linalg.norm(ref_p[min(int(f[0]), T - 1), h] - ref_p[int(f[0]) - 1, h])), 2) for h in range(Hn)]} cm/step", flush=True)
        init = arr["needs_init"].numpy().astype(bool)
        tp = arr["target_pos"].numpy().reshape(W, Hn, 3); tq = arr["target_quat"].numpy().reshape(W, Hn, 4)
        tf = arr["target_finger"].numpy().reshape(W, -1).copy()
        filt = A.filtered_action.numpy().reshape(W, -1)
        a = np.zeros((W, A.action_dim), np.float32)
        for h in range(Hn):
            p, q = tp[:, h].copy(), tq[:, h].copy()
            p[init] = jq[init][:, pq_ids[h]]; q[init] = jq[init][:, oq[h]:oq[h] + 4]
            a0 = starts[h]
            want = np.concatenate([ref_p[f, h] - p, rotvec(qmul(q * [-1, -1, -1, 1], ref_q[f, h]))], 1)
            a[:, a0:a0 + 6] = (want - ema * filt[:, a0:a0 + 6]) / ((1 - ema) * stp[a0:a0 + 6])
            if track_fingers:
                cur = tf[:, hfs[h]:hfs[h] + nf[h]]
                cur[init] = jq[init][:, fq[hfs[h]:hfs[h] + nf[h]]]
                sl = slice(a0 + 6, a0 + 6 + nf[h])
                a[:, sl] = (ref_f[h][f] - cur - ema * filt[:, sl]) / ((1 - ema) * stp[sl])
        return np.clip(a, -1, 1)

    return track, track_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("out"); ap.add_argument("setting")
    ap.add_argument("--worlds", type=int, default=64)
    args = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from flash_chord.evaluation import harness as H
    from flash_chord.evaluation.rollout import record_object_pose_rollout
    from flash_chord.training.flash_sac.runner import resolve_jax_device
    from genv import GEnv

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with_g = "_g_" in args.setting
    if with_g:
        rollout.WarpRLEnv = GEnv
    cfg, metadata, training, setup, env, jenv, thresholds = rollout.build(args.checkpoint, args.worlds)
    sides = list(env.action.layout.sides); T = int(setup.reference.num_frames); W = env.world_count
    print(f"[g_only] reference {cfg.task.parquet}", flush=True)
    eng_gt = human_engage(str(cfg.task.parquet), sides, T)
    np.save(out / "engage_gt.npy", eng_gt)
    teach = out.parent.parent / out.name / "engage_timeline.npy"           # full/gonly/epN -> full/epN
    if args.setting == "track_g_teach":
        eng = np.load(teach).astype(np.float32)[:T]
    else:
        eng = eng_gt
    if teach.exists():
        tt = np.load(teach)[:T]
        print(f"[g_only] human vs teacher engage agreement per hand: {[round(float(((tt[:, h] > 0.5) == (eng_gt[:, h] > 0.5)).mean()), 3) for h in range(len(sides))]}", flush=True)
    for h, s in enumerate(sides):
        print(f"[g_only] {s} engage runs ({args.setting}): {[(int(a), int(b)) for a, b in runs(eng[:, h] > 0.5)]}", flush=True)

    env.reset(frame_id=0)
    obs = jnp.asarray(env.observation.numpy().reshape(W, env.observation_dim))
    if with_g:
        jenv._reset_state(jnp.ones(W, bool))
        _, _, _, ca, _ = jenv._live()
        obs = jnp.concatenate([obs, jenv._extra(ca, jenv._eng)], 1)
    device = resolve_jax_device(training.device)
    obs = jax.device_put(obs, device)
    Hn = len(sides)
    track, track_err = make_tracker(env, setup.reference, sides, T, args.setting == "track_ref")
    starts = list(env.action.layout.hand_action_starts)

    def policy(state, observation, key):
        a = track()
        if with_g:
            ts = np.clip(env.timestep.numpy(), 0, T - 1)
            e = np.where(eng[ts] > 0.5, 1.0, -1.0)                                           # (W, H)
            a = np.concatenate([np.stack([a[:, s:s + 6] for s in starts], 1), e[..., None]], -1).reshape(W, 7 * Hn)
        return jax.device_put(jnp.asarray(a, jnp.float32), device), key

    ro = record_object_pose_rollout(env, jenv, policy, None, jax.random.PRNGKey(0), obs, reference=setup.reference,
                                    step_count=T, start_frame=0, progress_interval=0)
    m = H.score_rollout(ro, thresholds)
    row = {"setting": args.setting, "add_auc": float(m.add_auc), "mppe_cm": float(m.mppe_cm), "chord_sr": float(m.chord_sr),
           "worlds": W, "g": os.environ.get("GMD_G"), "wrist_track_err_cm": round(100 * float(np.mean(track_err[1:])), 2),
           "wrist_track_err_p95_cm": round(100 * float(np.percentile(track_err[1:], 95)), 2)}
    json.dump(row, open(out / f"{args.setting}.json", "w"), indent=1)
    print(f"[g_only] {out.name} {args.setting} add_auc {row['add_auc']:.3f} mppe {row['mppe_cm']:.1f} cm wrist err {row['wrist_track_err_cm']} cm", flush=True)


if __name__ == "__main__":
    main()
