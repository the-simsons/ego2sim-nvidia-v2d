"""NumPy MANO forward pass (CPU / Mac friendly, no torch), numerically matching the manotorch layer the V2D
pipeline uses (axis-angle, use_pca=False, flat hand mean, no centering).

Side convention (same as upstream v2d HaMeR/WiLoR records): every hand is evaluated with the RIGHT model; a left
hand is the right-model output mirrored in x (and faces re-wound). `scale` rescales the mesh about its vertex
centroid (upstream `hand_scale`), then `transl` is added:

    V = mirror_if_left(MANO_R(global_orient, hand_pose, betas))
    V = scale * (V - mean(V)) + mean(V) + transl

Joint order (21): wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4 (OpenPose / MediaPipe order), with
fingertips taken from vertices 745, 317, 444, 556, 673.

The model file is a plain .npz converted from the licensed MANO_RIGHT.pkl (tools/convert_mano.py); it is never
committed. Path: $HANDBENCH_MANO or handtracking/assets/mano_right.npz.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

TIP_VERTS = [745, 317, 444, 556, 673]
# MANO 16 joints + 5 tips -> OpenPose order
JOINT_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
PARENTS21 = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
BONES = [(PARENTS21[j], j) for j in range(1, 21)]
FINGERTIPS = [4, 8, 12, 16, 20]

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "assets" / "mano_right.npz"


def rodrigues(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3)."""
    aa = np.asarray(aa, np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-12)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = np.zeros_like(kx)
    Kx = np.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], -1).reshape(aa.shape[:-1] + (3, 3))
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)


class Mano:
    def __init__(self, path: str | os.PathLike | None = None):
        path = Path(path or os.environ.get("HANDBENCH_MANO", DEFAULT_PATH))
        if not path.exists():
            raise FileNotFoundError(f"MANO npz not found at {path}; run handbench/tools/convert_mano.py")
        d = np.load(path)
        self.v_template = d["v_template"]
        self.shapedirs = d["shapedirs"]  # (778, 3, 10)
        self.posedirs = d["posedirs"]  # (778, 3, 135)
        self.J_regressor = d["J_regressor"]  # (16, 778)
        self.weights = d["weights"]  # (778, 16)
        self.faces = d["faces"].astype(np.int64)
        self.parents = d["kintree_table"][0].astype(np.int64)
        self.parents[0] = -1

    def faces_for(self, is_right: bool) -> np.ndarray:
        return self.faces if is_right else self.faces[:, [0, 2, 1]]

    def forward(self, global_orient, hand_pose, betas, transl=None, is_right=True, scale=1.0):
        """Batched: inputs (N,3), (N,45), (N,10), (N,3); is_right / scale scalar or (N,). Returns verts (N,778,3),
        joints (N,21,3)."""
        go = np.atleast_2d(np.asarray(global_orient, np.float64))
        N = go.shape[0]
        hp = np.asarray(hand_pose, np.float64).reshape(N, 15, 3)
        be = np.asarray(betas, np.float64).reshape(N, -1)
        R = rodrigues(np.concatenate([go[:, None], hp], 1))  # (N,16,3,3)
        v_shaped = self.v_template[None] + np.einsum("vck,nk->nvc", self.shapedirs[..., : be.shape[1]], be)
        J = np.einsum("jv,nvc->njc", self.J_regressor, v_shaped)
        pose_feat = (R[:, 1:] - np.eye(3)).reshape(N, 135)
        v_posed = v_shaped + np.einsum("vck,nk->nvc", self.posedirs, pose_feat)
        G = np.zeros((N, 16, 4, 4))
        G[:, 0, :3, :3], G[:, 0, :3, 3], G[:, 0, 3, 3] = R[:, 0], J[:, 0], 1
        for j in range(1, 16):
            p = self.parents[j]
            L = np.zeros((N, 4, 4))
            L[:, :3, :3], L[:, :3, 3], L[:, 3, 3] = R[:, j], J[:, j] - J[:, p], 1
            G[:, j] = G[:, p] @ L
        joints16 = G[:, :, :3, 3].copy()
        Gr = G.copy()
        Gr[:, :, :3, 3] -= np.einsum("njab,njb->nja", G[:, :, :3, :3], J)
        T = np.einsum("vj,njab->nvab", self.weights, Gr)
        verts = np.einsum("nvab,nvb->nva", T[..., :3, :3], v_posed) + T[..., :3, 3]
        joints = np.concatenate([joints16, verts[:, TIP_VERTS]], 1)[:, JOINT_ORDER]

        is_right = np.broadcast_to(np.asarray(is_right, bool), (N,))
        m = np.where(is_right, 1.0, -1.0)
        verts[..., 0] *= m[:, None]
        joints[..., 0] *= m[:, None]
        scale = np.broadcast_to(np.asarray(scale, np.float64), (N,))
        c = verts.mean(1, keepdims=True)
        verts = scale[:, None, None] * (verts - c) + c
        joints = scale[:, None, None] * (joints - c) + c
        if transl is not None:
            t = np.asarray(transl, np.float64).reshape(N, 1, 3)
            verts, joints = verts + t, joints + t
        return verts, joints
