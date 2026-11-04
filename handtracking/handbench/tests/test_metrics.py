"""Synthetic check of the metrics with the real rig calibration (spec 1.6): a known MANO hand rendered into cam_a and
cam_b must score ~0 against itself, and a copy shifted 2 cm along the viewing ray must give the cam_b pixel shift the
small-baseline formula of 1.3 predicts."""
import numpy as np
import pytest

from handbench import format as hf
from handbench import paths
from handbench.calib import load_factory, load_rig
from handbench.datasets.base import EpisodeRefs
from handbench.mano import Mano
from handbench.metrics import evaluate_episode
from handbench.render import depth_map, silhouette

pytestmark = pytest.mark.skipif(not (paths.ROOT / "assets/mano_right.npz").exists(), reason="MANO npz not available")

N = 6  # frames


class _Stereo:
    def __init__(self, crops, scale):
        self.crops, self.scale = crops, scale

    def get(self, i, side):
        return self.crops.get((i, side))


def _cams():
    return load_rig(paths.rig_json()) if paths.rig_json().exists() else load_factory(paths.calib_json())


def _gt_tracks(mano, shift=0.0):
    rng = np.random.default_rng(0)
    n = N
    a = hf.empty(n, np.arange(n) / 20.25, with_verts=True)
    for i in range(n):
        for side in ("L", "R"):
            r = hf.row_index(i, side)
            go = np.array([0.3, 2.5 if side == "R" else -2.5, 0.2]) + rng.normal(0, 0.05, 3)
            hp = rng.normal(0, 0.15, 45)
            be = np.zeros(10)
            tr = np.array([0.08 if side == "R" else -0.08, 0.12, 0.40])
            V, J = mano.forward(go, hp, be, tr, side == "R")
            ray = J[0].mean(0) / np.linalg.norm(J[0].mean(0))
            off = ray * shift / ray[2]  # move by `shift` in depth along the hand's viewing ray
            a["valid"][r], a["source"][r], a["conf"][r] = True, "observed", 1.0
            a["global_orient"][r], a["hand_pose"][r], a["betas"][r] = go, hp, be
            a["transl"][r] = tr + off
            a["joints_cam"][r] = J[0] + off
            a["verts_cam"][r] = V[0] + off
    meta = {k: None for k in hf.META_KEYS}
    meta.update(method="synthetic", fps=20.25, consumed_stereo_depth=False)
    return hf.HandTracks(a, meta)


def _refs(gt: hf.HandTracks, mano, cams):
    A, B = cams["ego_cam_a"], cams["ego_cam_b"]
    views = ("ego_cam_a", "ego_cam_b", "ego_cam_c")
    J = gt["joints_cam"].reshape(N, 2, 21, 3).astype(float)
    V = gt["verts_cam"].reshape(N, 2, 778, 3).astype(float)
    kp = np.stack([np.stack([np.stack([cams[v].project(J[i, s]) for v in views]) for s in range(2)]) for i in range(N)])
    masks, crops = {}, {}
    sc = 0.5
    for i in range(N):
        for s, side in enumerate("LR"):
            f = mano.faces_for(side == "R")
            uv = V[i, s, :, :2] / V[i, s, :, 2:] * np.diag(A.K)[:2] + A.K[:2, 2]
            masks[("ego_cam_a", i, side)] = silhouette(uv, f, A.size)
            masks[("ego_cam_b", i, side)] = silhouette(B.project(V[i, s]), f, B.size)
            K = A.K.copy()
            K[:2] *= sc
            K[:2, 2] = (A.K[:2, 2] + 0.5) * sc - 0.5
            uvh = V[i, s, :, :2] / V[i, s, :, 2:] * np.diag(K)[:2] + K[:2, 2]
            W, H = int(A.size[0] * sc), int(A.size[1] * sc)
            D = depth_map(uvh, V[i, s, :, 2], f, (W, H))
            crops[(i, side)] = (D, D > 0, np.array([0, 0, W - 1, H - 1]))
    return EpisodeRefs(name="synthetic", n_frames=N, fps=20.25, cams=cams, visible=np.ones((N, 2), bool),
                       kp2d=kp, kp2d_score=np.ones(kp.shape[:-1]), kp_views=views,
                       joints_gt={"tri_abc": J.copy(), "tri_ab": J.copy()},
                       mask_fn=lambda v, i, side: masks.get((v, i, side)), stereo=_Stereo(crops, sc))


def test_gt_scores_zero():
    mano, cams = Mano(), _cams()
    gt = _gt_tracks(mano)
    rows, epm = evaluate_episode(gt, _refs(gt, mano, cams), mano)
    r = {k: np.mean([x[k] for x in rows if k in x]) for k in rows[0] if k not in ("frame", "side", "scored")}
    assert r["mpjpe_tri_abc"] < 1e-3 and r["mpjpe_tri_ab"] < 1e-3
    assert r["kp_a"] < 1e-3 and r["kp_b"] < 1e-3 and r["kp_c"] < 1e-3
    assert r["pck10_b"] == 1.0
    assert r["iou_a"] > 0.99 and r["iou_b"] > 0.99
    assert r["stereo_abs"] < 1.0  # mm (painter's-algorithm rendering of the same mesh)
    assert epm["coverage_observed"] == 1.0 and epm["swaps_per_min"] == 0


def test_depth_shift_matches_formula():
    mano, cams = Mano(), _cams()
    gt = _gt_tracks(mano)
    ref = _refs(gt, mano, cams)
    dz = 0.02
    rows, _ = evaluate_episode(_gt_tracks(mano, shift=dz), ref, mano)
    depth = np.mean([x["depth_tri_abc"] for x in rows])
    lat = np.mean([x["lateral_tri_abc"] for x in rows])
    bias = np.mean([x["bias_tri_abc"] for x in rows])
    # the shift is 2 cm in z along the (off-axis) hand ray, so its length along the ray is 2 cm * |r| / r_z
    c = gt["joints_cam"].reshape(N, 2, 21, 3).mean(2).reshape(-1, 3)
    along = np.mean(1000 * dz * np.linalg.norm(c, axis=1) / c[:, 2])
    assert abs(bias - along) < 1.5 and abs(depth - along) < 1.5 and lat < 2.0, (bias, depth, along, lat)
    # the metric equals the per-joint reprojection displacement computed independently here
    B = cams["ego_cam_b"]
    sh = _gt_tracks(mano, shift=dz)
    J0 = gt["joints_cam"].reshape(N, 2, 21, 3).astype(float)
    J1 = sh["joints_cam"].reshape(N, 2, 21, 3).astype(float)
    disp = np.linalg.norm(B.project(J1) - B.project(J0), axis=-1)  # (N,2,21)
    kp_b = np.mean([x["kp_b"] for x in rows])
    assert abs(kp_b - disp.mean(-1).mean()) < 1e-3
    # parallax of the hand centroid in cam_b matches the 1.3 formula f_b * b * dZ / Z^2 (the per-joint error also
    # contains the hand's apparent size change, which cam_a sees too)
    # The formula is for a pinhole: check it on cam_b's undistorted projection. In the real (distorted, 127 deg)
    # image the shift is smaller where hands sit (periphery magnification < f), as the 1.3 exact numbers show.
    c0, c1 = J0.mean(2), J1.mean(2)
    par = np.linalg.norm(B.project(c1, distort=False) - B.project(c0, distort=False), axis=-1)
    pred = B.K[0, 0] * np.linalg.norm(B.center_in_a) * dz / c0[..., 2] ** 2
    assert np.all(np.abs(par - pred) / pred < 0.15), (par, pred)
    par_real = np.linalg.norm(B.project(c1) - B.project(c0), axis=-1)
    assert np.all(par_real < par * 1.05)
    ca1 = cams["ego_cam_a"].project(c1)
    ca0 = cams["ego_cam_a"].project(c0)
    assert np.linalg.norm(ca1 - ca0, axis=-1).max() < 0.3 * par.min()  # cam_a sees ~no centroid parallax
    assert abs(np.mean([x["stereo_signed"] for x in rows]) - 20.0) < 3.0
