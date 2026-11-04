"""G's per-hand input features (v2: up to two objects, mesh shape code, surface normals). One pure function,
`features(xp, ...)`, runs on numpy (training set, closed-loop tests) and on jax.numpy (inside RL training), so the inputs
can never disagree. v1 (single object, dust-pan mini experiment) is gfeat_v1.py.

Target object per hand: the object whose surface is nearest the palm; it only switches when another object is closer by
more than 2 cm (hysteresis). The other object (if the scene has two) is kept as an obstacle.

Per hand (wrist = palm frame; everything below is expressed in it):
  target: pose (position 3, rotation 6-d) + linear/angular velocity (finite difference, 20 Hz)        15
  target: 11 hand points (palm, 5 distal phalanges, 5 tips) -> nearest surface point (clipped 0.3 m)  33
  target: surface normal at each of those points, oriented towards the hand point                     33
  target: BPS shape code                                                                              32
  other object: 11 nearest-surface vectors (zeros if absent) + present flag                           34
  finger joint positions / velocities                                                                 44
  contacts with the target per contact link (L) + log1p total force on it + any contact with the other  L + 2
  wrist command: raw wrist action, its first and second differences                                   18
  previous executed finger action                                                                     22
  other hand: any contact, engaged; own engage; handedness (right = 1)                                4
"""
from __future__ import annotations

import numpy as np

CLIP = 0.3
SWITCH = 0.02


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


def surface_normals(points, k=8):
    """Unoriented normals of sampled surface points (B, N, 3) from a plane fit to each point's k nearest neighbours."""
    out = np.zeros_like(points)
    for b in range(points.shape[0]):
        P = points[b]
        d = np.linalg.norm(P[:, None] - P[None], axis=-1)
        for i in range(len(P)):
            nb = P[np.argsort(d[i])[:k]]
            out[b, i] = np.linalg.svd(nb - nb.mean(0))[2][-1]
    return out


def feature_dim(meta):
    return 15 + 33 + 33 + 32 + 34 + 44 + int(meta["link_counts"][0]) + 2 + 18 + 22 + 4


def _take(a, idx, xp):
    """a (W, B, ...) -> a[w, idx[w]] (W, ...)."""
    i = idx.reshape((idx.shape[0], 1) + (1,) * (a.ndim - 2))
    return xp.take_along_axis(a, i, 1)[:, 0]


def _nearest(surf, p, xp):
    """surf (W, N, 3), p (W, 3) -> vector to the nearest point (W, 3) and its index (W,)."""
    d = surf - p[:, None]
    k = xp.argmin(xp.sum(d * d, -1), 1)
    return xp.take_along_axis(d, k[:, None, None], 1)[:, 0], k


def features(xp, m, body_q, joint_q, joint_qd, contact_active, contact_force_w, raw_wrist_hist, prev_finger, engage,
             prev_R=None, prev_p=None, prev_target=None, fresh=None):
    """All inputs for W worlds at one step. raw_wrist_hist (3, W, H, 6) = raw wrist action at t, t-1, t-2; prev_finger
    (W, H, 22); engage (W, H); prev_R (W, B, 3, 3) / prev_p (W, B, 3) = object poses at the previous step (None: zero
    velocity); prev_target (W, H) int (None: no hysteresis); fresh (W,) bool = worlds just reset (no hysteresis). Returns (X (W, H, D) float32, R, p, target (W, H))."""
    W, H = body_q.shape[0], len(m["sides"])
    obs_ids = [int(b) for b in m["object_body_ids"]]; B = len(obs_ids)
    R = xp.stack([quat_xyzw_to_mat(body_q[:, b, 3:], xp) for b in obs_ids], 1)              # (W,B,3,3)
    p = xp.stack([body_q[:, b, :3] for b in obs_ids], 1)                                     # (W,B,3)
    if prev_R is None:
        prev_R, prev_p = R, p
    fps = float(m["fps"])
    v = (p - prev_p) * fps
    w = rotvec_from_mat(R @ xp.swapaxes(prev_R, -1, -2), xp) * fps
    if "mesh_var" in m:                       # per-world mesh variant (meshmix.py): arrays (V, B, ...) picked per world
        var, ob = np.asarray(m["mesh_var"]), np.arange(B)[None]
        sp = np.asarray(m["surface_points_var"], np.float64)[var, ob]; sn = np.asarray(m["surface_normals_var"], np.float64)[var, ob]
        surf = xp.einsum("wbij,wbkj->wbki", R, sp) + p[:, :, None]
        nrm = xp.einsum("wbij,wbkj->wbki", R, sn)
        bps = np.asarray(m["bps_var"], np.float32)[var, ob]                                     # (W,B,32)
    else:
        sp = np.asarray(m["surface_points"], np.float64)[:B]; sn = np.asarray(m["surface_normals"], np.float64)[:B]
        surf = xp.einsum("wbij,bkj->wbki", R, sp) + p[:, :, None]                             # (W,B,N,3)
        nrm = xp.einsum("wbij,bkj->wbki", R, sn)
        bps = np.asarray(m["bps"], np.float32)[:B]
    hs, L = [int(x) for x in m["hand_slot_starts"]], int(m["link_counts"][0])
    ca_h = [contact_active[:, hs[h]:hs[h] + B * L].reshape(W, B, L) for h in range(H)]
    cf_h = [contact_force_w[:, hs[h]:hs[h] + B * L].reshape(W, B, L, 3) for h in range(H)]
    any_contact = [ca_h[h].reshape(W, -1).max(1) for h in range(H)]
    out, targets = [], []
    for h, side in enumerate(m["sides"]):
        frames = m["frames"][side]
        Rw, pw = frame_pose(body_q, frames[0], xp)
        RwT = xp.swapaxes(Rw, 1, 2)
        dist = xp.stack([xp.sqrt(xp.min(xp.sum((surf[:, b] - pw[:, None]) ** 2, -1), 1)) for b in range(B)], 1)  # (W,B)
        cand = xp.argmin(dist, 1)
        if prev_target is None or B == 1:
            tgt = cand
        else:
            prev = prev_target[:, h].astype(cand.dtype)
            better = _take(dist, cand, xp) < _take(dist, prev, xp) - SWITCH
            tgt = xp.where(better, cand, prev)
            if fresh is not None:
                tgt = xp.where(fresh, cand, tgt)
        targets.append(tgt)
        Rt, pt = _take(R, tgt, xp), _take(p, tgt, xp)
        rel_R = RwT @ Rt
        rel_p = xp.einsum("wij,wj->wi", RwT, pt - pw)
        rot6 = xp.swapaxes(rel_R[:, :, :2], 1, 2).reshape(W, 6)
        vel = xp.concatenate([xp.einsum("wij,wj->wi", RwT, _take(v, tgt, xp)), xp.einsum("wij,wj->wi", RwT, _take(w, tgt, xp))], 1)
        st, nt = _take(surf, tgt, xp), _take(nrm, tgt, xp)
        vt, nts, vo = [], [], []
        so = _take(surf, 1 - tgt, xp) if B == 2 else None
        for fr in frames:
            _, q = frame_pose(body_q, fr, xp)
            d, k = _nearest(st, q, xp)
            n = xp.take_along_axis(nt, k[:, None, None], 1)[:, 0]
            n = n * xp.where(xp.sum(n * -d, -1, keepdims=True) >= 0, 1.0, -1.0)              # face the hand point
            ln = xp.sqrt(xp.sum(d * d, -1, keepdims=True))
            vt.append(xp.einsum("wij,wj->wi", RwT, d * xp.minimum(1.0, CLIP / xp.maximum(ln, 1e-9))))
            nts.append(xp.einsum("wij,wj->wi", RwT, n))
            if so is not None:
                do, _ = _nearest(so, q, xp)
                lo = xp.sqrt(xp.sum(do * do, -1, keepdims=True))
                vo.append(xp.einsum("wij,wj->wi", RwT, do * xp.minimum(1.0, CLIP / xp.maximum(lo, 1e-9))))
            else:
                vo.append(xp.zeros((W, 3)))
        present = xp.full((W, 1), float(B == 2))
        bps_t = _take(xp.asarray(bps), tgt, xp) if bps.ndim == 3 else xp.asarray(bps)[tgt]
        fq = joint_q[:, np.asarray(m["finger_q_ids"][side])]
        fqd = joint_qd[:, np.asarray(m["finger_dof_ids"][side])]
        ca_t = _take(ca_h[h], tgt, xp)
        cf_t = _take(cf_h[h], tgt, xp)
        fmag = xp.log1p(xp.sqrt(xp.sum(cf_t * cf_t, -1)).sum(1, keepdims=True))
        ca_o = _take(ca_h[h], 1 - tgt, xp).max(1, keepdims=True) if B == 2 else xp.zeros((W, 1))
        a0, a1, a2 = raw_wrist_hist[0][:, h], raw_wrist_hist[1][:, h], raw_wrist_hist[2][:, h]
        cmd = xp.concatenate([a0, a0 - a1, a0 - 2 * a1 + a2], 1)
        o = 1 - h
        flags = xp.stack([any_contact[o], engage[:, o], engage[:, h], xp.full((W,), float(side == "right"))], 1)
        out.append(xp.concatenate([rel_p, rot6, vel, *vt, *nts, bps_t, *vo, present, fq, fqd, ca_t, fmag, ca_o, cmd,
                                   prev_finger[:, h], flags], 1))
    return xp.stack(out, 1).astype(xp.float32), R, p, xp.stack(targets, 1)


# column ranges used elsewhere
def blocks(meta):
    L = int(meta["link_counts"][0])
    o = 0; out = {}
    for name, n in (("pose", 15), ("vec_t", 33), ("nrm_t", 33), ("bps", 32), ("vec_o", 33), ("present", 1), ("fq", 22),
                    ("fqd", 22), ("ca_t", L), ("fmag", 1), ("ca_o", 1), ("cmd", 18), ("prev", 22), ("flags", 4)):
        out[name] = (o, o + n); o += n
    return out


class Featurizer:
    """numpy wrapper keeping the previous object poses and target objects between calls."""

    def __init__(self, meta):
        self.m = meta
        self.reset()

    def reset(self):
        self.prev = (None, None, None)

    def dim(self):
        return feature_dim(self.m)

    def __call__(self, body_q, joint_q, joint_qd, contact_active, contact_force_w, raw_wrist_hist, prev_finger, engage):
        X, R, p, tgt = features(np, self.m, body_q.astype(np.float64), joint_q, joint_qd, contact_active.astype(np.float64),
                                contact_force_w.astype(np.float64), raw_wrist_hist, prev_finger, engage, *self.prev)
        self.prev = (R, p, tgt)
        self.target = tgt
        return X
