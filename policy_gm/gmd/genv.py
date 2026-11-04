"""GEnv: the simulator's JAX view with a frozen grasp policy G in the loop (used by train_m.py and eval_m.py).

M's action per hand = [wrist 6, engage 1] (14 numbers, sides right, left). Engage switches on above +0.3 and off below
-0.3. Each step G (jax, frozen, weights from $GMD_G) maps gfeat.features(...) to the 22 finger actions; the full 56-d S0
action [wrist from M, fingers from G] goes to the simulator. M observes the S0 observation plus, per hand, engaged and
in-contact (4 numbers). G's per-world state (wrist-command history, previous finger action, previous object pose,
engage) resets with the world.

Optional grip reward ($GMD_GRIP_LIB = grip_lib.py output, $GMD_GRIP_W = effective weight, default 10): per hand and object
with a known grip, r = window[t] * exp(-|p - p*| / 3 cm) * exp(-angle(R, R*) / 20 deg) on the object pose in the palm frame.
FlashSAC builds its reward from the objective terms and the curriculum weights (learner: frame_dt * terms . weights), so r is
added to the `hand_ray` term column scaled by W / 5: it gets hand_ray's schedule (weight 5 until 30M, ramped to 0 by 50M).
$GMD_GRIP_KERNEL=broad: r = window * [exp(-d / 12 cm) * (0.5 + 0.5 exp(-angle / 45 deg)) + exp(-d / 3 cm) exp(-angle / 20 deg)]
(dense guidance from far away plus the narrow bonus at the grip); default "narrow" is the product above.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import flash_chord.training.environment as fenv
from flash_chord.runtime.jax import to_jax

sys.path.insert(0, "/env/gmd")
from gfeat import features, frame_pose, quat_xyzw_to_mat, surface_normals  # noqa: E402
import meshmix  # noqa: E402

ON, OFF = 0.3, -0.3


def g_forward(p, x, n):
    h = (x - p["mu"]) / p["sd"]
    for i in range(n):
        h = h @ p[f"W{i}"] + p[f"b{i}"]
        mu, var = h.mean(-1, keepdims=True), h.var(-1, keepdims=True)
        h = jax.nn.silu((h - mu) / jnp.sqrt(var + 1e-6) * p[f"g{i}"] + p[f"be{i}"])
    return jnp.tanh(h @ p[f"W{n}"] + p[f"b{n}"])


def scene_meta(env):
    lay, cl, sl = env.action.layout, env.contact.layout, env.scene.layout
    fr = lambda f: dict(name=f.name, body_id=int(f.body_id), pos=list(f.body_to_frame_pos), quat=list(f.body_to_frame_quat_xyzw))  # noqa: E731
    ex = np.load(env.objective.config.extras_path, allow_pickle=True)
    sp = ex["object_surface_points"].astype(np.float64)
    meta = dict(sides=list(lay.sides), fps=float(env.reference.fps), object_body_ids=np.asarray(cl.object_body_ids),
                hand_slot_starts=np.asarray(cl.hand_slot_starts), link_counts=np.asarray(cl.link_counts),
                surface_points=sp, surface_normals=surface_normals(sp), bps=np.asarray(ex["bps"], np.float32),
                frames={s: [fr(sl.hand(s).palm_frame)] + [fr(f) for f in sl.hand(s).dp_frames] + [fr(f) for f in sl.hand(s).fingertip_frames] for s in lay.sides},
                finger_q_ids={s: np.asarray(sl.hand(s).finger_q_ids) for s in lay.sides},
                finger_dof_ids={s: np.asarray(sl.hand(s).finger_dof_ids) for s in lay.sides},
                hand_action_starts=np.asarray(lay.hand_action_starts), finger_counts=np.asarray(lay.finger_counts))
    meta.update(meshmix.meta_extras(env.world_count))      # per-world mesh variant (empty unless $GMD_MESHMIX)
    return meta


def make_g_step(meta, params, n_hidden, W, inner_dim, debounce=0):
    """Jitted (pre, post) functions around one simulator step. State: (hist, prev_f, eng, prev_R, prev_p, target, fresh).
    debounce K > 0: while a hand is engaged and touching an object, a release (engage < OFF) only takes effect after K
    consecutive release commands (pre's extra counter argument, per world and hand)."""
    H = len(meta["sides"]); nf = int(meta["finger_counts"][0])
    wrist_idx = np.stack([np.arange(s, s + 6) for s in meta["hand_action_starts"]])
    fing_idx = np.stack([np.arange(s + 6, s + 6 + nf) for s in meta["hand_action_starts"]])
    hs, L, B = [int(v) for v in meta["hand_slot_starts"]], int(meta["link_counts"][0]), len(meta["object_body_ids"])

    def pre(bq, jq, jqd, ca, cf, a, state, cnt):
        hist, prev_f, eng, prev_R, prev_p, target, fresh = state
        a = jnp.clip(a.reshape(W, H, 7), -1, 1)
        wrist, e = a[..., :6], a[..., 6]
        touch = jnp.stack([ca[:, hs[h]:hs[h] + B * L].max(1) for h in range(H)], 1) > 0
        hold_rel = (eng > 0.5) & (e < OFF) & touch
        cnt = jnp.where(hold_rel, cnt + 1, 0)
        want = jnp.where(e > ON, 1.0, jnp.where(e < OFF, 0.0, eng))
        eng = jnp.where(hold_rel & (cnt < debounce), 1.0, want)
        hist = jnp.stack([wrist, hist[0], hist[1]])
        R_now = jnp.stack([quat_xyzw_to_mat(bq[:, b, 3:], jnp) for b in meta["object_body_ids"]], 1)
        p_now = jnp.stack([bq[:, b, :3] for b in meta["object_body_ids"]], 1)
        prev_R = jnp.where(fresh[:, None, None, None], R_now, prev_R); prev_p = jnp.where(fresh[:, None, None], p_now, prev_p)
        X, R, p, target = features(jnp, meta, bq, jq, jqd, ca, cf, hist, prev_f, eng, prev_R, prev_p, target, fresh)
        g = g_gated(params, X.reshape(W * H, -1), n_hidden, eng.reshape(-1)).reshape(W, H, nf)
        full = jnp.zeros((W, inner_dim), jnp.float32)
        for h in range(H):
            full = full.at[:, wrist_idx[h]].set(wrist[:, h]).at[:, fing_idx[h]].set(g[:, h])
        return full, (hist, g, eng, R, p, target, jnp.zeros_like(fresh)), cnt

    def extra(ca, eng):
        contact = jnp.stack([ca[:, hs[h]:hs[h] + B * L].max(1) for h in range(H)], 1)
        return jnp.concatenate([eng, contact], 1).astype(jnp.float32)

    def post(state, obs, term_obs, terminated, truncated, ca):
        hist, prev_f, eng, prev_R, prev_p, target, fresh = state
        done = jnp.logical_or(terminated.astype(bool), truncated.astype(bool)).reshape(-1)
        k = ~done
        term_x = extra(ca, eng)
        state = (hist * k[None, :, None, None], prev_f * k[:, None, None], eng * k[:, None], prev_R, prev_p, target, fresh | done)
        obs = jnp.concatenate([obs, extra(ca, state[2])], 1)
        term_obs = None if term_obs is None else jnp.concatenate([term_obs, term_x], 1)
        return state, obs, term_obs

    return jax.jit(pre), jax.jit(extra), jax.jit(post)


def make_grip_reward(meta, lib, kernel="narrow"):
    """Jitted (body_q (W, bodies, 7), frame (W,)) -> grip reward (W,) for the grips in `lib` (grip_lib.py)."""
    valid, gp, gR, win = (np.asarray(lib[k]) for k in ("valid", "grip_p", "grip_R", "window"))
    T = win.shape[0]; win_j = jnp.asarray(win, jnp.float32)
    pairs = [(h, b) for h in range(valid.shape[0]) for b in range(valid.shape[1]) if valid[h, b]]
    obj_ids = [int(b) for b in meta["object_body_ids"]]

    @jax.jit
    def f(bq, frame):
        frame = jnp.clip(frame, 0, T - 1)
        r = jnp.zeros(bq.shape[0], jnp.float32)
        for h, b in pairs:
            Rw, pw = frame_pose(bq, meta["frames"][meta["sides"][h]][0], jnp)
            Ro = quat_xyzw_to_mat(bq[:, obj_ids[b], 3:], jnp); po = bq[:, obj_ids[b], :3]
            RwT = jnp.swapaxes(Rw, 1, 2)
            rel_R = RwT @ Ro; rel_p = jnp.einsum("wij,wj->wi", RwT, po - pw)
            d = jnp.linalg.norm(rel_p - gp[h, b], axis=1)
            c = (jnp.trace(jnp.einsum("ji,wjk->wik", gR[h, b], rel_R), axis1=1, axis2=2) - 1.0) / 2.0
            ang = jnp.arccos(jnp.clip(c, -1.0, 1.0))
            sharp = jnp.exp(-d / 0.03) * jnp.exp(-ang / 0.349)
            if kernel == "broad":
                sharp = jnp.exp(-d / 0.12) * (0.5 + 0.5 * jnp.exp(-ang / 0.785)) + sharp
            r = r + win_j[frame, h, b] * sharp
        return r
    return f, len(pairs)


def g_gated(p, x, n, eng):
    """Gated G (keys off_*: a second net used while engage is off): eng (N,) -> where(eng, G_on(x), G_off(x)).
    Plain G when there are no off_* keys."""
    on = g_forward(p, x, n)
    if "off_W0" not in p:
        return on
    q = {k[4:]: v for k, v in p.items() if k.startswith("off_")}
    return jnp.where(eng[:, None] > 0.5, on, g_forward(q, x, n))


class GEnv(fenv.WarpRLEnv):
    def __init__(self, env, **kw):
        super().__init__(env, **kw)
        gp = dict(np.load(os.environ["GMD_G"], allow_pickle=True))
        n = int(gp.pop("n_hidden")); gp.pop("engage_timeline", None); gp.pop("off_n_hidden", None)
        self._debounce = int(os.environ.get("GMD_DEBOUNCE", "0"))
        self._meta = scene_meta(env)
        W, H = self.world_count, len(self._meta["sides"])
        self._inner_dim = self.action_dim
        self.action_dim = 7 * H
        self._base_obs = self.observation_dim
        self.observation_dim = self.observation_dim + 2 * H
        self._pre, self._extra, self._post = make_g_step(self._meta, {k: jnp.asarray(v, jnp.float32) for k, v in gp.items()}, n, W, self._inner_dim,
                                                       self._debounce)
        self._W, self._H, self._nf = W, H, int(self._meta["finger_counts"][0])
        self._B = len(self._meta["object_body_ids"])
        self._Bw = env.model.body_count // W
        self._nq, self._nd = env.model.joint_coord_count // W, env.model.joint_dof_count // W
        self._reset_state(jnp.ones(W, bool))
        self._grip = None
        if os.environ.get("GMD_GRIP_LIB"):
            lib = dict(np.load(os.environ["GMD_GRIP_LIB"]))
            self._grip, npairs = make_grip_reward(self._meta, lib, os.environ.get("GMD_GRIP_KERNEL", "narrow"))
            self._ray_idx = list(env.objective.term_names).index("hand_ray")
            self._grip_k = float(os.environ.get("GMD_GRIP_W", "10")) / 5.0
            self._grip_n, self._grip_acc, self._grip_calls = 0, 0.0, 0
            print(f"[gmd] grip reward on: {npairs} hand-object grips from {os.environ['GMD_GRIP_LIB']}, added to term "
                  f"'hand_ray' (column {self._ray_idx}) x {self._grip_k:.2f}, kernel {os.environ.get('GMD_GRIP_KERNEL', 'narrow')}", flush=True)
        print(f"[gmd] GEnv: M action {self.action_dim} (wrist 6 + engage per hand), G fingers from {os.environ['GMD_G']}, "
              f"observation {self.observation_dim}, objects {self._B}, gated {'off_W0' in gp}, debounce {self._debounce}", flush=True)

    @property
    def _eng(self):
        return self._state[2]

    def _reset_state(self, mask):
        W, H, nf, B = self._W, self._H, self._nf, self._B
        if not hasattr(self, "_state"):
            self._state = (jnp.zeros((3, W, H, 6)), jnp.zeros((W, H, nf)), jnp.zeros((W, H)), jnp.tile(jnp.eye(3), (W, B, 1, 1)),
                           jnp.zeros((W, B, 3)), jnp.zeros((W, H), jnp.int32), jnp.ones(W, bool))
            return
        hist, prev_f, eng, R, p, tgt, fresh = self._state
        k = ~mask
        self._state = (hist * k[None, :, None, None], prev_f * k[:, None, None], eng * k[:, None], R, p, tgt, fresh | mask)

    def _live(self):
        e = self.env
        W = self._W
        return (to_jax(e.state_0.body_q).reshape(W, self._Bw, 7), to_jax(e.state_0.joint_q).reshape(W, self._nq),
                to_jax(e.state_0.joint_qd).reshape(W, self._nd), to_jax(e.contact.contact_active).reshape(W, -1).astype(jnp.float32),
                to_jax(e.contact.contact_force_w).reshape(W, -1, 3))

    def _super(self, fn, *a):
        """Call a WarpRLEnv method with the simulator's own observation and action sizes (it builds views and checks shapes with them)."""
        full = (self.observation_dim, self.action_dim)
        self.observation_dim, self.action_dim = self._base_obs, self._inner_dim
        try:
            return fn(*a)
        finally:
            self.observation_dim, self.action_dim = full

    def reset(self):
        obs = self._super(super().reset)
        self._reset_state(jnp.ones(self._W, bool))
        _, _, _, ca, _ = self._live()
        return jnp.concatenate([obs, self._extra(ca, self._eng)], 1)

    def step(self, action):
        bq, jq, jqd, ca, cf = self._live()
        if not hasattr(self, "_cnt"):
            self._cnt = jnp.zeros((self._W, self._H), jnp.int32)
        full, self._state, self._cnt = self._pre(bq, jq, jqd, ca, cf, action, self._state, self._cnt)
        if os.environ.get("GMD_SYNC"):                 # debugging aid: force G to finish before the simulator step
            full.block_until_ready()
        out = self._super(super().step, full)
        _, _, _, ca2, _ = self._live()
        self._state, obs, term = self._post(self._state, out.observation, out.terminal_observation, out.terminated, out.truncated, ca2)
        if self._grip is not None and out.objective_terms is not None:
            bq2 = to_jax(self.env.state_0.body_q).reshape(self._W, self._Bw, 7)
            done = jnp.logical_or(out.terminated.astype(bool), out.truncated.astype(bool)).reshape(-1)
            r = jnp.where(done, 0.0, self._grip(bq2, to_jax(self.env.timestep).reshape(-1) - 1))
            terms = out.objective_terms.at[:, self._ray_idx].add(self._grip_k * r)
            self._grip_acc = self._grip_acc + r.mean(); self._grip_calls += 1
            if self._grip_calls % 500 == 0:
                print(f"[gmd] grip reward: mean {float(self._grip_acc) / 500:.4f} per world-step over the last 500 steps "
                      f"({self._grip_calls * self._W:,} env steps)", flush=True)
                self._grip_acc = 0.0
            return dataclasses.replace(out, observation=obs, terminal_observation=term, objective_terms=terms)
        return dataclasses.replace(out, observation=obs, terminal_observation=term)
