"""[v1, single object - used by the dust-pan mini experiment (G files g/*_r0.npz)]
G's per-hand input features from one step of simulator state. One pure function, `features(xp, ...)`, runs on numpy
(training set from recorded rollouts, closed-loop tests) and on jax.numpy (inside RL training), so the inputs can never
disagree between the three.

Per hand (wrist = palm frame; everything below is expressed in it):
  object pose (position 3, rotation 6-d)                         9
  object linear / angular velocity (finite difference, 20 Hz)    6
  11 hand points (palm, 5 distal phalanges, 5 tips) -> nearest object surface point, clipped to 0.3 m   33
  finger joint positions / velocities                           2 x 22
  hand-object contact flags per contact link (+ total force magnitude, log1p)   L + 1
  wrist command: raw wrist action, its first and second differences             18
  previous executed finger action                                22
  other hand: any contact, engaged; own engage; handedness (right = 1)          4
"""
from __future__ import annotations

import numpy as np

CLIP = 0.3


def quat_xyzw_to_mat(q, xp=np):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def frame_pose(body_q, frame, xp=np):
    """World pose (R (...,3,3), p (...,3)) of a semantic frame. body_q (..., bodies, 7) [p, q_xyzw]."""
    bq = body_q[..., frame["body_id"], :]
    Rb = quat_xyzw_to_mat(bq[..., 3:], xp)
    Rf = quat_xyzw_to_mat(np.asarray(frame["quat"], np.float64))
    return Rb @ Rf, bq[..., :3] + xp.einsum("...ij,j->...i", Rb, np.asarray(frame["pos"], np.float64))


def rotvec_from_mat(R, xp=np):
    c = xp.clip((xp.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    a = xp.arccos(c)
    v = xp.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = xp.sin(a)[..., None]
    return xp.where(s > 1e-6, v / xp.maximum(2 * s, 1e-9) * a[..., None], 0.5 * v)


def feature_dim(meta):
    return 9 + 6 + 33 + 44 + int(meta["link_counts"][0]) + 1 + 18 + 22 + 4


def features(xp, m, body_q, joint_q, joint_qd, contact_active, contact_force_w, raw_wrist_hist, prev_finger, engage,
             prev_Ro=None, prev_po=None):
    """All inputs for W worlds at one step (single-object scenes). raw_wrist_hist (3, W, H, 6) = raw wrist action at
    t, t-1, t-2; prev_finger (W, H, 22); engage (W, H); prev_Ro/prev_po = object pose at the previous step (None: zero
    velocity). Returns (X (W, H, D) float32, Ro, po)."""
    W, H = body_q.shape[0], len(m["sides"])
    ob = int(m["object_body_ids"][0])
    Ro = quat_xyzw_to_mat(body_q[:, ob, 3:], xp); po = body_q[:, ob, :3]
    if prev_Ro is None:
        prev_Ro, prev_po = Ro, po
    surf_w = xp.einsum("wij,kj->wki", Ro, np.asarray(m["surface_points"][0], np.float64)) + po[:, None]
    fps = float(m["fps"])
    v_w = (po - prev_po) * fps
    w_w = rotvec_from_mat(Ro @ xp.swapaxes(prev_Ro, 1, 2), xp) * fps
    hs, lc = [int(v) for v in m["hand_slot_starts"]], [int(v) for v in m["link_counts"]]
    any_contact = [contact_active[:, hs[h]:hs[h] + lc[h]].max(1) for h in range(H)]
    out = []
    for h, side in enumerate(m["sides"]):
        Rw, pw = frame_pose(body_q, m["frames"][side][0], xp)
        RwT = xp.swapaxes(Rw, 1, 2)
        rel_R = RwT @ Ro
        rel_p = xp.einsum("wij,wj->wi", RwT, po - pw)
        rot6 = xp.swapaxes(rel_R[:, :, :2], 1, 2).reshape(W, 6)
        vel = xp.concatenate([xp.einsum("wij,wj->wi", RwT, v_w), xp.einsum("wij,wj->wi", RwT, w_w)], 1)
        vecs = []
        for fr in m["frames"][side]:
            _, p = frame_pose(body_q, fr, xp)
            d = surf_w - p[:, None]
            k = xp.argmin(xp.sum(d * d, -1), 1)
            nn = xp.take_along_axis(d, k[:, None, None], 1)[:, 0]
            n = xp.sqrt(xp.sum(nn * nn, -1, keepdims=True))
            nn = nn * xp.minimum(1.0, CLIP / xp.maximum(n, 1e-9))
            vecs.append(xp.einsum("wij,wj->wi", RwT, nn))
        fq = joint_q[:, np.asarray(m["finger_q_ids"][side])]
        fqd = joint_qd[:, np.asarray(m["finger_dof_ids"][side])]
        ca = contact_active[:, hs[h]:hs[h] + lc[h]]
        cf = contact_force_w[:, hs[h]:hs[h] + lc[h]]
        fmag = xp.log1p(xp.sqrt(xp.sum(cf * cf, -1)).sum(1, keepdims=True))
        a0, a1, a2 = raw_wrist_hist[0][:, h], raw_wrist_hist[1][:, h], raw_wrist_hist[2][:, h]
        cmd = xp.concatenate([a0, a0 - a1, a0 - 2 * a1 + a2], 1)
        o = 1 - h
        flags = xp.stack([any_contact[o], engage[:, o], engage[:, h], xp.full((W,), float(side == "right"))], 1)
        out.append(xp.concatenate([rel_p, rot6, vel, *vecs, fq, fqd, ca, fmag, cmd, prev_finger[:, h], flags], 1))
    return xp.stack(out, 1).astype(xp.float32), Ro, po


class Featurizer:
    """numpy wrapper keeping the previous object pose between calls (rollout.py, label.py, view_g.py)."""

    def __init__(self, meta):
        self.m = meta
        self.prev = (None, None)

    def reset(self):
        self.prev = (None, None)

    def dim(self):
        return feature_dim(self.m)

    def __call__(self, body_q, joint_q, joint_qd, contact_active, contact_force_w, raw_wrist_hist, prev_finger, engage):
        X, Ro, po = features(np, self.m, body_q.astype(np.float64), joint_q, joint_qd, contact_active.astype(np.float64),
                             contact_force_w.astype(np.float64), raw_wrist_hist, prev_finger, engage, *self.prev)
        self.prev = (Ro, po)
        return X
