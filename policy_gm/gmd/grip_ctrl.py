"""Hybrid M + scripted-wrist controller for the G grip test: M drives everything except one hand's wrist and engage, which a
tracking controller moves onto a given grip of a given object (object pose in the palm frame, e.g. measured in another
episode where M + G succeeded), then carries along the object's reference trajectory.

Plan for the scripted hand (palm-frame grip G_rel = (rel_R, rel_p): object pose in the palm frame):
  frame 0 .. t_g-14 : palm moves from its start pose to a pre-grasp pose `approach` m back along the palm->object direction
  t_g-12 .. t_g     : straight in to the grasp pose  P = O * G_rel^-1
  until t_g+6       : O = the object's actual pose (closed loop: the real object need not sit on its reference)
  after             : P(t) = O_ref(t+1) * G_rel^-1 (carry the object along its reference)
  engage            : on once t >= t_g-6 and the achieved grip is within 2.5 cm of G_rel, or from t_g+6 regardless
  lead              : the wrist target leads the goal by `lead` x the tracking error (the S0 wrist PD trails its target)
Trajectory mode (params rel_p_t / rel_R_t / eng_t / t_switch / t_follow0, e.g. the median approach of a successful
episode): palm = O_actual * rel(t)^-1 for t_follow0 <= t < t_switch (interpolated from the start pose before t_follow0),
engage = eng_t[t]; from t_switch on, carry along the reference with the final grip and engage on.
The wrist target is derived from the palm target through the hand's fixed wrist->palm transform (measured at frame 0) and
reached with the same EMA-inverting tracker as g_only.make_tracker (S0 relative action: |a| <= 1 = 3 cm / 0.1 rad per step).
"""
from __future__ import annotations

import numpy as np


def qmat(q):  # xyzw -> (..., 3, 3)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def mat_q(R):  # (..., 3, 3) -> xyzw
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(R.shape[:-2] + (4,))


def rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


def slerp_R(R0, R1, s):
    from scipy.spatial.transform import Rotation, Slerp
    out = []
    for a, b in zip(R0.reshape(-1, 3, 3), R1.reshape(-1, 3, 3)):
        out.append(Slerp([0, 1], Rotation.from_matrix([a, b]))([s]).as_matrix()[0])
    return np.stack(out).reshape(R0.shape)


def make_hybrid(env, genv, setup, m_policy, params):
    """Returns policy(actor, obs, key) -> (W, 14) action: M's action with the scripted hand's wrist and engage replaced."""
    import jax.numpy as jnp
    W = env.world_count; meta = genv._meta; sides = list(meta["sides"]); H = len(sides)
    h = int(params["hand"]); ob = int(params["obj"]); t_g = int(params["t_grasp"]); appr = float(params["approach"])
    rel_R, rel_p = np.asarray(params["rel_R"], np.float64), np.asarray(params["rel_p"], np.float64)
    A = env.action; arr = A.arrays; ema = float(A.ema); stp = arr["step"].numpy(); a0 = int(A.layout.hand_action_starts[h])
    pq_ids = arr["wrist_pos_q_ids"].numpy().reshape(H, 3)[h]; oq = int(arr["wrist_quat_q_ids"].numpy()[h])
    nq = env.model.joint_coord_count // W; Bw = env.model.body_count // W
    fr = meta["frames"][sides[h]][0]; f_R = qmat(np.asarray(fr["quat"], np.float64)); f_p = np.asarray(fr["pos"], np.float64)
    ref = setup.reference; T = int(ref.num_frames)
    rb = ob                                                                              # one reference body per object
    O_p = np.asarray(ref.object_body_pos_w())[:, rb]                                   # (T, 3)
    O_R = qmat(np.asarray(ref.object_body_quat_w())[:, rb][:, [1, 2, 3, 0]])           # (T, 3, 3)
    inv_R = rel_R.T; inv_p = -rel_R.T @ rel_p                                            # palm pose in the object frame
    st = {}

    def palm_world(bq):
        b = bq[:, int(fr["body_id"])]
        Rb = qmat(b[:, 3:]); return Rb @ f_R, b[:, :3] + np.einsum("wij,j->wi", Rb, f_p)

    def grasp_palm(t, obj=None):
        """Palm pose for the grip on the object: obj = (R (W,3,3), p (W,3)) actual pose, else the reference at t."""
        if obj is None:
            t = min(max(t, 0), T - 1)
            return O_R[t] @ inv_R, O_p[t] + O_R[t] @ inv_p
        Ro, po = obj
        return Ro @ inv_R, po + np.einsum("wij,j->wi", Ro, inv_p)

    traj = "rel_p_t" in params
    if traj:
        rp_t, rR_t = np.asarray(params["rel_p_t"]), np.asarray(params["rel_R_t"]); eng_t = np.asarray(params["eng_t"]).astype(bool)
        t_sw, t_f0 = int(params["t_switch"]), int(params["t_follow0"])

    def palm_from_rel(Rr, pr, obj):
        Ro, po = obj
        return Ro @ Rr.T, po - np.einsum("wij,j->wi", Ro @ Rr.T, pr)

    def desired_palm_traj(t, obj):
        if t >= t_sw:
            return grasp_palm(t + 1)
        if t >= t_f0:
            return palm_from_rel(rR_t[t], rp_t[t], obj)
        Rg, pg = palm_from_rel(rR_t[t_f0], rp_t[t_f0], obj)
        s = t / max(t_f0, 1); R0, p0 = st["palm0"]
        return slerp_R(R0, np.broadcast_to(Rg, R0.shape).copy(), s), p0 + s * (pg - p0)

    def desired_palm(t, obj):
        if traj:
            return desired_palm_traj(t, obj)
        if t >= t_g + 6:
            return grasp_palm(t + 1)
        Rg, pg = grasp_palm(t, obj)
        if t >= t_g:
            return Rg, pg
        d = np.einsum("wij,j->wi", Rg, rel_p); d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        pre = pg - appr * d
        if t >= t_g - 12:
            s = (t - (t_g - 12)) / 12.0
            return Rg, pre + s * (pg - pre)
        s = min(t / max(t_g - 14, 1), 1.0)
        R0, p0 = st["palm0"]
        Rs = slerp_R(R0, np.broadcast_to(Rg, R0.shape).copy(), s)
        return Rs, p0 + s * (pre - p0)

    def policy(actor, obs, key):
        a, key = m_policy(actor, obs, key)
        a = np.asarray(a, np.float32).reshape(W, H, 7).copy()
        t = int(env.timestep.numpy()[0])
        bq = env.state_0.body_q.numpy().reshape(W, Bw, 7); jq = env.state_0.joint_q.numpy().reshape(W, nq)
        if "palm0" not in st:                                          # fixed wrist -> palm transform, start palm pose
            Rp, pp = palm_world(bq)
            Rw = qmat(jq[:, oq:oq + 4]); pw = jq[:, pq_ids]
            st["wp_R"] = np.einsum("wji,wjk->wik", Rw, Rp); st["wp_p"] = np.einsum("wji,wj->wi", Rw, pp - pw)
            st["palm0"] = (Rp, pp)
        ob_q = bq[:, int(meta["object_body_ids"][ob])]
        ob_p = ob_q[:, :3]
        Rd, pd = desired_palm(t, (qmat(ob_q[:, 3:]), ob_p))
        Rd = np.broadcast_to(Rd, (W, 3, 3)); pd = np.broadcast_to(pd, (W, 3))
        Rp, pp = palm_world(bq)                                        # diagnostics: palm error, achieved grip
        st.setdefault("log", []).append((t, float(np.linalg.norm(pp - pd, axis=1).mean()),
                                         np.einsum("wji,wj->wi", Rp, ob_p - pp).mean(0)))
        Rw_d = np.einsum("wij,wkj->wik", Rd, st["wp_R"]); pw_d = pd - np.einsum("wij,wj->wi", Rw_d, st["wp_p"])
        lead = float(params.get("lead", 0.6))                          # aim past the goal by lead x the tracking error
        Rw_a = qmat(jq[:, oq:oq + 4]); pw_a = jq[:, pq_ids]
        err_rv = rotvec(np.einsum("wji,wjk->wik", Rw_a, Rw_d))
        pw_d = pw_d + lead * (pw_d - pw_a)
        from scipy.spatial.transform import Rotation as _Rot
        Rw_d = np.einsum("wij,wjk->wik", Rw_d, _Rot.from_rotvec(lead * err_rv).as_matrix())
        init = arr["needs_init"].numpy().astype(bool)
        tp = arr["target_pos"].numpy().reshape(W, H, 3)[:, h].copy(); tq = arr["target_quat"].numpy().reshape(W, H, 4)[:, h].copy()
        tp[init] = jq[init][:, pq_ids]; tq[init] = jq[init][:, oq:oq + 4]
        filt = A.filtered_action.numpy().reshape(W, -1)[:, a0:a0 + 6]
        want = np.concatenate([pw_d - tp, rotvec(np.einsum("wji,wjk->wik", qmat(tq), Rw_d))], 1)
        a[:, h, :6] = np.clip((want - ema * filt) / ((1 - ema) * stp[a0:a0 + 6]), -1, 1)
        grip_err = np.linalg.norm(np.einsum("wji,wj->wi", Rp, ob_p - pp) - rel_p, axis=1)   # achieved vs target grip
        st["eng"] = np.where(st.get("eng", np.zeros(W, bool)) | ((t >= t_g - 6) & (grip_err < 0.025)) | (t >= t_g + 6), True, False)
        a[:, h, 6] = np.where(st["eng"], 1.0, -1.0)
        if traj:
            a[:, h, 6] = 1.0 if (t >= t_sw or eng_t[min(t, t_sw - 1)]) else -1.0
        return jnp.asarray(a.reshape(W, 7 * H)), key

    return policy


def place_hand(env, genv, params, back=0.02):
    """Teleport the scripted hand onto the grip (params rel_R / rel_p) around the object's actual pose, `back` m short of
    it along the palm->object direction, with zero hand velocity; called right after env.reset(frame_id=F). The relative
    action re-initialises its targets from the new joint state on the next step."""
    import newton
    W = env.world_count; meta = genv._meta; sides = list(meta["sides"]); H = len(sides)
    h, ob = int(params["hand"]), int(params["obj"])
    rel_R, rel_p = np.asarray(params["rel_R"], np.float64), np.asarray(params["rel_p"], np.float64)
    arr = env.action.arrays; nq = env.model.joint_coord_count // W; nd = env.model.joint_dof_count // W; Bw = env.model.body_count // W
    pq_ids = arr["wrist_pos_q_ids"].numpy().reshape(H, 3)[h]; oq = int(arr["wrist_quat_q_ids"].numpy()[h])
    fr = meta["frames"][sides[h]][0]; f_R = qmat(np.asarray(fr["quat"], np.float64)); f_p = np.asarray(fr["pos"], np.float64)
    bq = env.state_0.body_q.numpy().reshape(W, Bw, 7); jq = env.state_0.joint_q.numpy().reshape(W, nq).copy()
    b = bq[:, int(fr["body_id"])]; Rb = qmat(b[:, 3:]); Rp = Rb @ f_R; pp = b[:, :3] + np.einsum("wij,j->wi", Rb, f_p)
    Rw = qmat(jq[:, oq:oq + 4]); pw = jq[:, pq_ids]
    wp_R = np.einsum("wji,wjk->wik", Rw, Rp); wp_p = np.einsum("wji,wj->wi", Rw, pp - pw)
    o = bq[:, int(meta["object_body_ids"][ob])]; Ro = qmat(o[:, 3:])
    Rd = Ro @ rel_R.T; pd = o[:, :3] - np.einsum("wij,j->wi", Rd, rel_p)
    d = np.einsum("wij,j->wi", Rd, rel_p); d /= np.linalg.norm(d, axis=1, keepdims=True); pd = pd - back * d
    Rw_d = np.einsum("wij,wkj->wik", Rd, wp_R); pw_d = pd - np.einsum("wij,wj->wi", Rw_d, wp_p)
    jq[:, pq_ids] = pw_d; jq[:, oq:oq + 4] = mat_q(Rw_d)
    env.state_0.joint_q.assign(jq.reshape(-1).astype(np.float32))
    qd = env.state_0.joint_qd.numpy().reshape(W, nd); qd[:] = 0.0; env.state_0.joint_qd.assign(qd.reshape(-1))
    newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)
    arr["needs_init"].assign(np.ones(W, np.int32))
