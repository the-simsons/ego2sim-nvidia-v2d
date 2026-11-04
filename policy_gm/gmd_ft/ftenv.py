"""FTEnv: GEnv plus a bounded finger correction R on top of the frozen grasp policy G (gmd_ft: RL fine-tuning of the v4 runs).

R's correction is an offset on the finger joint *targets*, not on the per-step finger action (finger actions are integrated
deltas in s0_action.py, so a bound on the action would not bound how far the fingers drift from G). The wrapper keeps the
target G alone would produce (G's actions through the same EMA filter, step size, integration and joint-limit clamp as
s0_action.py) and every step sends the finger action that puts the real target at
    G-only target + gate * offset,   offset = clip(delta * sum_k r_k d_k, -delta, delta)   (per joint)
d_k (dirs.npz, make_dirs.py): squeeze, pc1, pc2 per hand side, each scaled to max |entry| = 1; the first $FT_K are used.
gate (per hand) ramps 0 -> 1 over 3 steps while the hand is engaged and touching an object and back to 0 otherwise, so
approach, acquire and release stay pure G. With r = 0 the executed finger action equals G's (the parity check).

Modes ($FT_MODE):
  r  the RL agent is R, action (H x K). M (frozen v4 actor $FT_M_CKPT, deterministic) drives wrists + engage.
     observation = M's observation + gate (H) + the wrist/engage action M executes this step (7H).
     $FT_PERTURB=1 (training): per world, wrist deltas scaled by U[1, $FT_SCALE_MAX] and a wrist position oscillation of
     U[$FT_AMP_MIN, $FT_AMP_MAX] m (default 0.005-0.02) at U[1, 3] Hz on a random axis (same for both hands), plus
     sudden pushes ($FT_BURST_P per step and world, default 0: 3 steps of a random wrist position + rotation delta of
     $FT_BURST_A action units, default 0.6), ramped in linearly over $FT_RAMP env steps.
     $FT_ZERO_R=1: R's action is ignored (offset 0): the same environment without R, for comparisons.
  m  the RL agent is M: GEnv's action and observation, so v4 training states resume unchanged. R is frozen ($FT_R_CKPT);
     without it the offset is 0.
Penalty: $FT_PEN * mean((gated offset / delta)^2) is added to the action_l2 term column (weight -0.002 in every curriculum
stage; with the default 500 a full-size offset costs about a tenth of the object-tracking reward).
Diagnostics (ft_diag(), used by ft_eval.py): gated fraction of hand-steps, mean |offset| / delta while gated, fraction of
gated hand-steps with any joint at the cap.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/env/gmd")
from flash_chord.runtime.jax import to_jax  # noqa: E402
from genv import GEnv  # noqa: E402


def load_actor(path, obs_dim, act_dim, W):
    """Frozen deterministic actor from a FlashSAC policy checkpoint, compiled for (W, obs_dim)."""
    from omegaconf import OmegaConf
    from flash_chord.configuration import instantiate_typed
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import checkpoint_config
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
    from flash_chord.training.flash_sac.runner import resolve_jax_device
    cfg = OmegaConf.create(checkpoint_config(read_checkpoint_metadata(Path(path))))
    training = instantiate_typed(cfg.training, TrainingConfig)
    dev = resolve_jax_device(training.device)
    obs = jax.device_put(jnp.zeros((W, obs_dim), jnp.float32), dev)
    state, policy, key, _ = load_policy_for_inference(str(path), training, obs, act_dim, deterministic=True, device=dev)
    return [state, policy, key]


def make_fix(fing_idx, ema, delta):
    """Jitted finger correction: G's full action -> executed full action, new (shadow, fG, gate), penalty, diagnostics."""
    def fix(full, g, eng, touch, r, shadow, fG, gate, fresh, tf, fa, fq, lo, hi, D, fstep):
        fr = fresh[:, None, None]
        gate = jnp.where((eng > 0.5) & touch, jnp.minimum(gate + 1.0 / 3.0, 1.0), jnp.maximum(gate - 1.0 / 3.0, 0.0))
        tgt_prev = jnp.where(fr, fq, tf)                    # s0_action re-initialises the target from joint_q after a reset
        f_prev = jnp.where(fr, 0.0, fa)
        fG = ema * jnp.where(fr, 0.0, fG) + (1.0 - ema) * jnp.clip(g, -1.0, 1.0) * fstep
        shadow = jnp.clip(jnp.where(fr, tgt_prev, shadow) + fG, lo, hi)
        off = jnp.clip(delta * jnp.einsum("whk,hkf->whf", jnp.clip(r, -1.0, 1.0), D), -delta, delta) * gate[..., None]
        desired = jnp.clip(shadow + off, lo, hi)
        v = jnp.clip((desired - tgt_prev - ema * f_prev) / ((1.0 - ema) * fstep), -1.0, 1.0)
        for h in range(fing_idx.shape[0]):
            full = full.at[:, fing_idx[h]].set(v[:, h])
        rel = jnp.abs(off) / delta
        on = gate > 0.99
        pen = jnp.mean(rel * rel, (1, 2))
        diag = jnp.stack([gate.mean(), jnp.where(on, rel.mean(-1), 0.0).sum(), (on & (rel.max(-1) >= 0.98)).sum(), on.sum()])
        return full, shadow, fG, gate, pen, diag
    return jax.jit(fix)


class FTEnv(GEnv):
    def __init__(self, env, **kw):
        super().__init__(env, **kw)
        W, H, nf = self._W, self._H, self._nf
        self._mode = os.environ.get("FT_MODE", "m")
        self._K = int(os.environ.get("FT_K", "3"))
        self._delta = float(os.environ.get("FT_DELTA", "0.05"))
        self._pen = float(os.environ.get("FT_PEN", "500"))
        self._perturb = os.environ.get("FT_PERTURB", "0") == "1"
        self._zero_r = os.environ.get("FT_ZERO_R", "0") == "1"
        self._scale_max = float(os.environ.get("FT_SCALE_MAX", "1.5"))
        self._ramp = float(os.environ.get("FT_RAMP", "5000000"))
        self._amp = (float(os.environ.get("FT_AMP_MIN", "0.005")), float(os.environ.get("FT_AMP_MAX", "0.02")))
        self._burst_p = float(os.environ.get("FT_BURST_P", "0"))
        self._burst_a = float(os.environ.get("FT_BURST_A", "0.6"))
        sides = self._meta["sides"]
        d = np.load(os.environ.get("FT_DIRS", "/ft/dirs.npz"))
        self._D = jnp.asarray(np.stack([d[f"dirs_{s}"][:self._K] for s in sides]), jnp.float32)          # (H, K, nf)
        act = env.action
        ema = float(act.ema)
        step = act.arrays["step"].numpy()
        starts = np.asarray(act.layout.hand_action_starts)
        self._fing_idx = np.stack([np.arange(s + 6, s + 6 + nf) for s in starts])                      # (H, nf) in the 56-d action
        self._fstep = jnp.asarray(np.stack([step[s + 6:s + 6 + nf] for s in starts]), jnp.float32)       # rad per step
        self._pos_step = float(step[starts[0]])                                                         # m per step (wrist position)
        self._ffs = [int(x) for x in act.layout.hand_finger_starts]
        self._total_f = int(act.layout.total_fingers)
        lo = act.limits[0].numpy().reshape(W, self._nd); hi = act.limits[1].numpy().reshape(W, self._nd)
        fd = self._meta["finger_dof_ids"]
        self._lo = jnp.asarray(np.stack([lo[:, fd[s]] for s in sides], 1)); self._hi = jnp.asarray(np.stack([hi[:, fd[s]] for s in sides], 1))
        self._fq_ids = np.stack([np.asarray(self._meta["finger_q_ids"][s]) for s in sides])            # (H, nf)
        hs, L, B = [int(v) for v in self._meta["hand_slot_starts"]], int(self._meta["link_counts"][0]), self._B
        self._touch = jax.jit(lambda ca: jnp.stack([ca[:, hs[h]:hs[h] + B * L].max(1) for h in range(H)], 1) > 0)
        self._fix = make_fix(self._fing_idx, ema, self._delta)
        self._term_idx = list(env.objective.term_names).index("action_l2")
        self._dt = 1.0 / float(self._meta["fps"])
        self._mobs_dim = self.observation_dim                                                           # GEnv's observation
        self._robs_dim = self._mobs_dim + H + 7 * H
        self._key = jax.random.PRNGKey(int(os.environ.get("FT_SEED", "0")))
        self._env_steps = 0
        self._diag = jnp.zeros(4); self._diag_n = 0
        self._M = self._R = None
        if self._mode == "r":
            self._M = load_actor(os.environ["FT_M_CKPT"], self._mobs_dim, 7 * H, W)
            self.action_dim = H * self._K
            self.observation_dim = self._robs_dim
        elif os.environ.get("FT_R_CKPT"):
            self._R = load_actor(os.environ["FT_R_CKPT"], self._robs_dim, H * self._K, W)
        self._ft_reset(jnp.ones(W, bool))
        print(f"[ft] FTEnv mode {self._mode}: K {self._K}, delta {self._delta} rad, penalty {self._pen}, perturb {self._perturb} "
              f"(scale <= {self._scale_max}, amplitude {self._amp} m, pushes p {self._burst_p} x {self._burst_a}, ramp {self._ramp:.0f}), zero_r {self._zero_r}, M {os.environ.get('FT_M_CKPT', '-')}, "
              f"R {os.environ.get('FT_R_CKPT', '-')}, action {self.action_dim}, observation {self.observation_dim}", flush=True)

    # ---- per-world state ----
    def _ft_reset(self, mask):
        W, H, nf = self._W, self._H, self._nf
        if not hasattr(self, "_shadow"):
            self._shadow = jnp.zeros((W, H, nf)); self._fG = jnp.zeros((W, H, nf)); self._gate = jnp.zeros((W, H))
            self._fresh = jnp.ones(W, bool); self._t = jnp.zeros(W, jnp.int32)
            self._pp = self._sample_perturb(jnp.ones(W, bool), None)
            self._m_next = jnp.zeros((W, 7 * H))
            self._burst_t = jnp.zeros(W, jnp.int32); self._burst_dir = jnp.zeros((W, 6))
        self._gate = jnp.where(mask[:, None], 0.0, self._gate)
        self._burst_t = jnp.where(mask, 0, self._burst_t)
        self._fresh = self._fresh | mask
        self._t = jnp.where(mask, 0, self._t)
        self._pp = self._sample_perturb(mask, self._pp)

    def _sample_perturb(self, mask, old):
        W = self._W
        self._key, *k = jax.random.split(self._key, 6)
        ax = jax.random.normal(k[0], (W, 3)); ax = ax / jnp.linalg.norm(ax, axis=1, keepdims=True)
        new = dict(scale=jax.random.uniform(k[1], (W,), minval=1.0, maxval=self._scale_max),
                   amp=jax.random.uniform(k[2], (W,), minval=self._amp[0], maxval=self._amp[1]),
                   freq=jax.random.uniform(k[3], (W,), minval=1.0, maxval=3.0),
                   phase=jax.random.uniform(k[4], (W,), minval=0.0, maxval=2 * math.pi), axis=ax)
        if old is None:
            return new
        return {n: jnp.where(mask.reshape((W,) + (1,) * (v.ndim - 1)), v, old[n]) for n, v in new.items()}

    def _perturbed(self, m_act):
        """Wrist perturbation (mode r with FT_PERTURB=1) on M's (W, 7H) action."""
        if not self._perturb:
            return m_act
        W, H = self._W, self._H
        rho = min(1.0, self._env_steps / max(self._ramp, 1.0))
        p = self._pp
        a = m_act.reshape(W, H, 7)
        s = 1.0 + rho * (p["scale"] - 1.0)
        w = 2 * math.pi * p["freq"]
        osc = rho * p["amp"] * w * self._dt * jnp.cos(w * self._t * self._dt + p["phase"]) / self._pos_step    # (W,) action units
        wrist = a[..., :6] * s[:, None, None]
        wrist = wrist.at[..., :3].add((p["axis"] * osc[:, None])[:, None, :])
        if self._burst_p > 0:                                   # sudden pushes: random lifts, shoves and turns
            self._key, k1, k2 = jax.random.split(self._key, 3)
            start = (self._burst_t <= 0) & (jax.random.uniform(k1, (W,)) < self._burst_p)
            d = jax.random.normal(k2, (W, 6)); d = d / jnp.linalg.norm(d, axis=1, keepdims=True)
            self._burst_dir = jnp.where(start[:, None], d, self._burst_dir)
            self._burst_t = jnp.where(start, 3, self._burst_t)
            on = (self._burst_t > 0).astype(jnp.float32) * rho
            wrist = wrist + (self._burst_a * on[:, None] * self._burst_dir)[:, None, :]
            self._burst_t = jnp.maximum(self._burst_t - 1, 0)
        return jnp.concatenate([jnp.clip(wrist, -1.0, 1.0), a[..., 6:]], -1).reshape(W, 7 * H)

    def _m_action(self, mobs):
        st, pol, key = self._M
        a, key = pol(st, mobs, key)
        self._M[2] = key
        return self._perturbed(jnp.asarray(a, jnp.float32).reshape(self._W, -1))

    def _robs(self, mobs, m_act):
        return jnp.concatenate([mobs, self._gate, m_act], 1)

    def _external(self, mobs):
        self._mobs = mobs
        if self._mode != "r":
            return mobs
        self._m_next = self._m_action(mobs)
        return self._robs(mobs, self._m_next)

    # ---- env API ----
    def reset(self):
        mobs = super().reset()
        self._ft_reset(jnp.ones(self._W, bool))
        return self._external(mobs)

    def eval_obs(self):
        """External observation after an explicit `env.reset(frame_id=0)` of the underlying RLEnv (ft_eval.py)."""
        self._reset_state(jnp.ones(self._W, bool))
        self._ft_reset(jnp.ones(self._W, bool))
        _, _, _, ca, _ = self._live()
        base = jnp.asarray(self.env.observation.numpy().reshape(self._W, self._base_obs))
        return self._external(jnp.concatenate([base, self._extra(ca, self._eng)], 1))

    def ft_diag(self):
        g, rel, cap, on = [float(x) for x in self._diag]
        n = max(self._diag_n, 1)
        return {"gate_frac": g / n, "offset_rel_mean": rel / max(on, 1.0), "cap_frac": cap / max(on, 1.0)}

    def step(self, action):
        W, H, K, nf = self._W, self._H, self._K, self._nf
        bq, jq, jqd, ca, cf = self._live()
        if not hasattr(self, "_cnt"):
            self._cnt = jnp.zeros((W, H), jnp.int32)
        if self._mode == "r":
            m_act = self._m_next
            r = jnp.zeros((W, H, K)) if self._zero_r else action.reshape(W, H, K)
        else:
            m_act = action
            r = jnp.zeros((W, H, K))
            if self._R is not None:
                st, pol, key = self._R
                ra, self._R[2] = pol(st, self._robs(self._mobs, m_act), key)
                r = jnp.asarray(ra, jnp.float32).reshape(W, H, K)
        full, self._state, self._cnt = self._pre(bq, jq, jqd, ca, cf, m_act, self._state, self._cnt)
        g, eng = self._state[1], self._state[2]
        tf = to_jax(self.env.action.arrays["target_finger"]).reshape(W, self._total_f)
        tf = jnp.stack([tf[:, s:s + nf] for s in self._ffs], 1)
        fa = to_jax(self.env.action.filtered_action).reshape(W, self._inner_dim)[:, self._fing_idx]
        fq = jq[:, self._fq_ids]
        full, self._shadow, self._fG, self._gate, pen, diag = self._fix(
            full, g, eng, self._touch(ca), r, self._shadow, self._fG, self._gate, self._fresh, tf, fa, fq, self._lo, self._hi,
            self._D, self._fstep)
        self._diag = self._diag + diag; self._diag_n += 1
        self._fresh = jnp.zeros(W, bool); self._t = self._t + 1; self._env_steps += W
        gate_before = self._gate
        out = self._super(super(GEnv, self).step, full)
        _, _, _, ca2, _ = self._live()
        self._state, obs, term = self._post(self._state, out.observation, out.terminal_observation, out.terminated, out.truncated, ca2)
        done = jnp.logical_or(out.terminated.astype(bool), out.truncated.astype(bool)).reshape(-1)
        self._ft_reset(done)
        terms = out.objective_terms
        if terms is not None:
            terms = terms.at[:, self._term_idx].add(jnp.where(done, 0.0, self._pen * pen))
        if self._mode == "r" and term is not None:
            term = jnp.concatenate([term, gate_before, m_act], 1)
        return dataclasses.replace(out, observation=self._external(obs), terminal_observation=term, objective_terms=terms)
