"""Pose maths shared by reconstruction and scoring. Conventions: 4x4 homogeneous matrices, quaternions xyzw unless a
function name says wxyz, metres, seconds."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


def mat(R, t):
    """Stack rotation matrices (N,3,3) and translations (N,3) into (N,4,4)."""
    R, t = np.asarray(R, float), np.asarray(t, float)
    single = R.ndim == 2
    R, t = np.atleast_3d(R) if not single else R[None], t[None] if single else t
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = R, t
    return T[0] if single else T


def pose_from_xyzw(q, t):
    return mat(Rot.from_quat(np.asarray(q, float)).as_matrix(), t)


def pose_from_dict(d):
    """{"rotation": [w, x, y, z], "translation": [...]} (the bundle / FoundationPose JSON format) -> 4x4."""
    w, x, y, z = d["rotation"]
    return pose_from_xyzw([x, y, z, w], d["translation"])


def pose_to_dict(T):
    x, y, z, w = Rot.from_matrix(T[:3, :3]).as_quat()
    return {"rotation": [float(w), float(x), float(y), float(z)], "translation": [float(v) for v in T[:3, 3]],
            "scale": [1.0, 1.0, 1.0]}


def load_pose_dir(d, n=None):
    """Per-frame pose JSONs 000000.json ... -> (N,4,4); missing frames are NaN."""
    files = sorted(f for f in Path(d).glob("*.json") if f.stem.isdigit())
    count = n if n is not None else (int(files[-1].stem) + 1 if files else 0)
    out = np.full((count, 4, 4), np.nan)
    for f in files:
        i = int(f.stem)
        if i < count:
            out[i] = pose_from_dict(json.load(open(f)))
    return out


def interp_poses(t_src, R_src, p_src, t_query):
    """Rotations by slerp and positions linearly, sampled at t_query (clamped to the source range)."""
    t = np.clip(np.asarray(t_query, float), t_src[0], t_src[-1])
    R = Slerp(t_src, Rot.from_matrix(R_src))(t).as_matrix()
    p = np.stack([np.interp(t, t_src, p_src[:, k]) for k in range(3)], 1)
    return mat(R, p)


def rotation_angle_deg(Ra, Rb):
    """Angle of Ra^T Rb, per frame, in degrees."""
    tr = np.einsum("nij,nij->n", Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def mean_pose(T):
    """Average of rigid poses (N,4,4): quaternion mean for rotation, arithmetic mean for translation."""
    R = Rot.from_matrix(T[:, :3, :3]).mean().as_matrix()
    return mat(R, T[:, :3, 3].mean(0))


def kabsch(P, Q):
    """R, t minimising |R P + t - Q| for corresponding point sets (N,3)."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


def box_symmetries():
    """The 4 proper 180-degree turns that leave a cuboid (or any object with 3 distinct mirror planes) looking alike."""
    return [np.diag([sx, sy, sz, 1.0]) for sx, sy, sz in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]


def symmetry_rotations(sym):
    """Rotations (as 4x4) for a symmetry declaration {"revolution_axis": "z"|None, "flips_180": ["x", ...]}:
    the identity, each listed 180-degree flip and their products; a revolution axis adds turns every 10 degrees."""
    sym = sym or {}
    axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    rots = [np.eye(3)]
    for a in sym.get("flips_180") or []:
        rots += [r @ Rot.from_rotvec(np.pi * np.array(axes[a])).as_matrix() for r in rots]
    if sym.get("revolution_axis"):
        ax = np.array(axes[sym["revolution_axis"]])
        rots = [r @ Rot.from_rotvec(np.radians(d) * ax).as_matrix() for r in rots for d in range(0, 360, 10)]
    out = []
    for r in rots:
        if not any(np.allclose(r, o, atol=1e-6) for o in out):
            out.append(r)
    return [mat(r, np.zeros(3)) for r in out]


def still_mask(positions, fps, speed=0.012, window_s=0.2):
    """Frames where the smoothed speed of a trajectory is below `speed` m/s."""
    v = np.r_[0, np.linalg.norm(np.diff(positions, axis=0), axis=1)] * fps
    k = max(1, int(round(window_s * fps)))
    return np.convolve(v, np.ones(k) / k, mode="same") < speed


def runs(mask, min_len=1):
    """[(start, stop)] of consecutive True runs of at least min_len."""
    out, i, m = [], 0, np.asarray(mask, bool)
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out
