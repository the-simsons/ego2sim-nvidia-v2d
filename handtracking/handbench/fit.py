"""Fitting helpers: similarity Procrustes and re-expressing a target mesh with MANO params."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from handbench.mano import Mano


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """dst ~ s R src + t. Returns s, R, t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    d = np.sign(np.linalg.det(U @ Vt))
    E = np.diag([1, 1, d])
    R = U @ E @ Vt
    s = (sig * np.diag(E)).sum() / (S ** 2).sum() * len(src) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def refit_global(mano: Mano, hand_pose, betas, is_right, verts_target, with_scale=True):
    """Given finger pose + betas, find (global_orient, transl, scale) so the MANO mesh matches verts_target (778,3)
    in the least-squares sense. Left hands: the model output is mirrored, so the rotation is fitted in the mirrored
    space and mapped back (mirror M: R_left_param = M R M)."""
    v0, _ = mano.forward(np.zeros(3), hand_pose, betas, None, True, 1.0)
    v0 = v0[0]
    M = np.diag([-1.0, 1, 1]) if not is_right else np.eye(3)
    tgt = verts_target @ M.T  # into right-model space
    # MANO global_orient rotates about the root joint, not the origin: fit in a frame that accounts for it
    _, j0 = mano.forward(np.zeros(3), hand_pose, betas, None, True, 1.0)
    root = j0[0, 0]
    s, R, t = umeyama(v0 - root, tgt, with_scale)
    go = Rot.from_matrix(R).as_rotvec()
    v1, _ = mano.forward(go, hand_pose, betas, None, True, 1.0)
    v1 = v1[0]
    # model: V = s (v1 - c) + c + transl  (scale about the vertex centroid, in mirrored space for left)
    c = v1.mean(0)
    pred = s * (v1 - c) + c
    transl_r = (tgt - pred).mean(0)
    transl = M @ transl_r
    return go, transl, s
