"""JEnv: joint training of M and the fingers (gmd_ft, plan (a): "unfreeze G inside M's training").

FlashSAC's actor cannot hold G's own weights without editing FlashSAC, so the fingers are G's output plus a learned
deviation that the SAC actor outputs alongside M's wrists and engage:
    finger action = clip(G(x) + s(t) * u, -1, 1),   u in [-1, 1]^(H x 22) from the actor
s(t) is the unfreeze schedule: 0 until $FJ_UNFREEZE env steps (default 50M: the assist is gone and the hand-ray reward has
decayed, so M is trained exactly as in v4 until then), then ramps linearly to $FJ_SCALE (default 0.5, about twice G's mean
hold action) over $FJ_RAMP steps (default 5M). The ramp stands in for "a lower learning rate for G" (FlashSAC's learning-rate
multiplier applies to the whole actor). Penalty: $FJ_PEN * mean((finger action - G(x))^2) on the action_l2 term column
(weight -0.002 in every stage; default 2000, so a deviation of the full scale costs ~10% of the object-tracking reward).
Action = M's 14 (wrist 6 + engage per hand) + 44 finger deviations. Observation = GEnv's + G's inputs (gfeat features of
both hands, from the current state) + G's output for them, so the actor sees what G sees and what G would do.
Training starts from scratch (the larger action and observation cannot resume a v4 state); compare with the v4 curves.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/env/gmd")
from gfeat import feature_dim, features  # noqa: E402
from genv import GEnv, g_gated  # noqa: E402


class JEnv(GEnv):
    def __init__(self, env, **kw):
        super().__init__(env, **kw)
        W, H, nf = self._W, self._H, self._nf
        self._T0 = float(os.environ.get("FJ_UNFREEZE", "50000000"))
        self._ramp = float(os.environ.get("FJ_RAMP", "5000000"))
        self._smax = float(os.environ.get("FJ_SCALE", "0.5"))
        self._pen = float(os.environ.get("FJ_PEN", "2000"))
        gp = dict(np.load(os.environ["GMD_G"], allow_pickle=True))
        self._gn = int(gp.pop("n_hidden")); gp.pop("engage_timeline", None); gp.pop("off_n_hidden", None)
        self._gp = {k: jnp.asarray(v, jnp.float32) for k, v in gp.items()}
        self._gdim = feature_dim(self._meta)
        starts = np.asarray(env.action.layout.hand_action_starts)
        self._fing_idx = np.stack([np.arange(s + 6, s + 6 + nf) for s in starts])
        self._term_idx = list(env.objective.term_names).index("action_l2")
        self._mobs_dim = self.observation_dim
        self.action_dim = 7 * H + H * nf
        self.observation_dim = self._mobs_dim + H * self._gdim + H * nf
        self._env_steps = 0
        self._eval_step = None
        self._dev = jnp.zeros(2); self._dev_n = 0
        meta, gn, fing_idx = self._meta, self._gn, self._fing_idx

        def gobs(params, bq, jq, jqd, ca, cf, state):
            hist, prev_f, eng, prev_R, prev_p, target, fresh = state
            X, _, _, _ = features(jnp, meta, bq, jq, jqd, ca, cf, hist, prev_f, eng, prev_R, prev_p, target, fresh)
            g = g_gated(params, X.reshape(W * H, -1), gn, eng.reshape(-1)).reshape(W, H, nf)
            return jnp.concatenate([X.reshape(W, -1), g.reshape(W, -1)], 1)

        def mix(full, g, u, s):
            fing = jnp.clip(g + s * u, -1.0, 1.0)
            for h in range(fing_idx.shape[0]):
                full = full.at[:, fing_idx[h]].set(fing[:, h])
            d = fing - g
            return full, jnp.mean(d * d, (1, 2)), jnp.stack([jnp.abs(d).mean(), (jnp.abs(d) > 0.1).mean()])

        self._gobs = jax.jit(gobs)
        self._mix = jax.jit(mix)
        print(f"[fj] JEnv: unfreeze at {self._T0:.0f} env steps, ramp {self._ramp:.0f}, scale {self._smax}, penalty {self._pen}; "
              f"action {self.action_dim}, observation {self.observation_dim} (G inputs {H}x{self._gdim} + G output {H}x{nf})", flush=True)

    def set_eval_step(self, step):
        self._eval_step = step

    def scale(self):
        step = self._env_steps if self._eval_step is None else self._eval_step
        return self._smax * float(np.clip((step - self._T0) / max(self._ramp, 1.0), 0.0, 1.0))

    def ft_diag(self):
        d, frac = [float(x) for x in self._dev]
        n = max(self._dev_n, 1)
        return {"finger_dev_mean": d / n, "finger_dev_gt_0.1": frac / n, "unfreeze_scale": self.scale()}

    def _with_g(self, mobs):
        bq, jq, jqd, ca, cf = self._live()
        return jnp.concatenate([mobs, self._gobs(self._gp, bq, jq, jqd, ca, cf, self._state)], 1)

    def reset(self):
        return self._with_g(super().reset())

    def eval_obs(self):
        self._reset_state(jnp.ones(self._W, bool))
        _, _, _, ca, _ = self._live()
        base = jnp.asarray(self.env.observation.numpy().reshape(self._W, self._base_obs))
        return self._with_g(jnp.concatenate([base, self._extra(ca, self._eng)], 1))

    def step(self, action):
        W, H, nf = self._W, self._H, self._nf
        a = action.reshape(W, self.action_dim)
        m_act, u = a[:, :7 * H], jnp.clip(a[:, 7 * H:], -1.0, 1.0).reshape(W, H, nf)
        bq, jq, jqd, ca, cf = self._live()
        if not hasattr(self, "_cnt"):
            self._cnt = jnp.zeros((W, H), jnp.int32)
        full, self._state, self._cnt = self._pre(bq, jq, jqd, ca, cf, m_act, self._state, self._cnt)
        full, pen, dev = self._mix(full, self._state[1], u, self.scale())
        self._dev = self._dev + dev; self._dev_n += 1
        self._env_steps += W
        out = self._super(super(GEnv, self).step, full)
        _, _, _, ca2, _ = self._live()
        self._state, obs, term = self._post(self._state, out.observation, out.terminal_observation, out.terminated, out.truncated, ca2)
        done = jnp.logical_or(out.terminated.astype(bool), out.truncated.astype(bool)).reshape(-1)
        bq2, jq2, jqd2, _, cf2 = self._live()
        gx = self._gobs(self._gp, bq2, jq2, jqd2, ca2, cf2, self._state)
        terms = out.objective_terms
        if terms is not None:
            terms = terms.at[:, self._term_idx].add(jnp.where(done, 0.0, self._pen * pen))
        return dataclasses.replace(out, observation=jnp.concatenate([obs, gx], 1),
                                   terminal_observation=None if term is None else jnp.concatenate([term, gx], 1), objective_terms=terms)
