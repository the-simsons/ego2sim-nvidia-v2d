"""Rigid-body fitting from motion capture: still stretches, table height, and where a mesh sits in Motive's rigid-body
frame. Ground-truth side only."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from e2s.geometry import mat, runs, still_mask
from e2s.mocap.motive import AXES


def rest_stretches(body, fps, speed=0.012, min_s=0.25):
    """[(i0, i1, R_mean (3,3), p_mean (3,))] for each stretch where the body is still."""
    q, p = body["quat_xyzw"], body["pos"]
    ok = ~np.isnan(p).any(1)
    still = still_mask(np.where(ok[:, None], p, np.nanmean(p, 0)), fps, speed) & ok
    out = []
    for i0, i1 in runs(still, int(min_s * fps)):
        out.append((i0, i1, Rot.from_quat(q[i0:i1]).mean().as_matrix(), p[i0:i1].mean(0)))
    return out


def lowest_point_height(T_world_mesh, vertices, up):
    """Height of the lowest mesh vertex for one pose."""
    v = (T_world_mesh[:3, :3] @ vertices.T).T + T_world_mesh[:3, 3]
    return float(v[:, AXES[up]].min())


def table_height(rests, T_rb_mesh, vertices, up):
    """Median height of the lowest mesh point over the resting poses, and the spread (a large spread means the mesh
    placement in the rigid-body frame, or one of the rests, is wrong)."""
    h = [lowest_point_height(mat(R, p) @ T_rb_mesh, vertices, up) for _, _, R, p in rests]
    if not h:
        return np.nan, np.nan
    return float(np.median(h)), float(np.ptp(h))


def fit_box_centre(rests, extents, up, R_rb_box=np.eye(3)):
    """Least-squares centre of a box (given extents along its own axes) in the rigid-body frame, plus the table height,
    from the condition that the lowest corner touches the table in every rest. Returns (centre (3,), table_h, rank,
    residuals). The system needs rests in at least 3 independent orientations to be well determined; check the rank."""
    up_i = AXES[up]
    A, b = [], []
    for _, _, R, p in rests:
        r = R[up_i] @ R_rb_box  # world-up component of each box axis, in the box frame
        A.append(np.r_[R[up_i], -1.0])
        b.append(-p[up_i] + 0.5 * np.abs(r) @ np.asarray(extents))
    A, b = np.array(A), np.array(b)
    x, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3], float(x[3]), int(rank), A @ x - b
