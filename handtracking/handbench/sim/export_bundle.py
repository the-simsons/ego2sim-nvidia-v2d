"""Export a method's hand tracks as a hands-only reconstruction bundle for the team's sim chain
(ego_recon loader -> rigid URDF -> retarget to Sharpa -> FlashCHORD kinematic replay). Option A: no visual odometry.

World = the head camera held FIXED at its frame-0 pose, rotated so the table plane (RANSAC on frame-0 SGBM stereo
points in the lower image) is horizontal: +z up, table top at z = 0 (the loader's --ground_z_offset then lifts it).
Head motion therefore shows up as extra hand motion. The loader needs one object per bundle: a 2 cm placeholder cube
rests on the table, static, 0.45 m to the side of the hands.

Hands: the method's per-frame MANO params in the same convention as upstream records (right model mirrored for the
left hand, cam_t = MANO translation). The loader ignores `hand_scale`, so sim hands use the natural MANO size for the
betas (the method's per-side size factor is reported in the manifest).

Writes handtracking/sim/<clip>/bundle/<clip>/outputs/result/{result.npz, mesh.obj, manifest.json} (schema of
v2d_common.result_bundle.write_result_bundle, reproduced here so no teammate environment is needed).
Usage: python -m handbench.sim.export_bundle --episode 12 --method E1best-sgbm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from handbench import format as hf
from handbench import paths
from handbench.calib import load_rig
from handbench.refs_io import read_frames
from handbench.stereo import StereoDepth

SIM_ROOT = paths.ROOT / "sim"


def table_frame(ep, cams, rng=np.random.default_rng(0)):
    """Camera->world (4x4): +z = table normal pointing at the camera, table plane at z = 0, origin below the camera."""
    sd = StereoDepth(cams, out_scale=0.5)
    fr = read_frames(ep, 0, ("ego_cam_b", "ego_cam_c"))
    D = sd.compute(fr["ego_cam_c"], fr["ego_cam_b"])
    H, W = D.shape
    K = cams["ego_cam_a"].K.copy()
    K[:2] *= 0.5
    vv, uu = np.nonzero((D > 0.25) & (D < 1.5))
    keep = vv > H * 0.45  # table is in the lower part of the head camera's view
    uu, vv = uu[keep], vv[keep]
    z = D[vv, uu]
    P = np.c_[(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z]
    best, best_in = None, 0
    for _ in range(400):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - s[0]) @ n) < 0.01
        if inl.sum() > best_in:
            best, best_in = (n, s[0]), inl.sum()
    n, p0 = best
    inl = np.abs((P - p0) @ n) < 0.01
    c = P[inl].mean(0)
    _, _, Vt = np.linalg.svd(P[inl] - c)
    n = Vt[2]
    if n @ (-c) < 0:  # point the normal towards the camera (up)
        n = -n
    x = np.array([1.0, 0, 0]) - n[0] * n  # camera x projected on the table = world x
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R_w_c = np.stack([x, y, n])  # rows: world axes expressed in the camera frame
    origin_cam = -n * (n @ (np.zeros(3) - c))  # foot of the camera centre on the table plane (camera frame)
    T = np.eye(4)
    T[:3, :3] = R_w_c
    T[:3, 3] = -R_w_c @ origin_cam
    return T, {"table_inlier_points": int(inl.sum()), "camera_height_above_table_m": float(n @ (np.zeros(3) - c)),
               "tilt_deg": float(np.degrees(np.arccos(np.clip(-n[1], -1, 1))))}


def cube_obj(path: Path, s=0.01):
    v = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)])
    f = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    path.write_text("\n".join([f"v {a:.4f} {b:.4f} {c:.4f}" for a, b, c in v] +
                              [f"f {q[0]} {q[1]} {q[2]}\nf {q[0]} {q[2]} {q[3]}" for q in f]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--method", default="E1best-sgbm")
    a = ap.parse_args()
    ep = paths.ep_name(a.episode)
    clip = f"ep{paths.ep_index(ep)}_{a.method.replace('-', '_')}"
    out = SIM_ROOT / clip / "bundle" / clip / "outputs" / "result"
    out.mkdir(parents=True, exist_ok=True)
    cams = load_rig(paths.rig_json())
    A = cams["ego_cam_a"]
    tr = hf.load(paths.RESULTS / a.method / ep / "hands.npz")
    n = tr.n_frames
    T_w_c, tinfo = table_frame(ep, cams)
    c2w = T_w_c  # camera_to_world (X_world = R X_cam + t), constant: the head camera is held fixed
    hands = {}
    for side, name in (("L", "left"), ("R", "right")):
        rows = tr.rows(side)
        ok = tr["valid"][rows] & (tr["source"][rows] == "observed")
        go = np.where(ok[:, None], tr["global_orient"][rows], 0).astype(np.float32)
        hp = np.where(ok[:, None], tr["hand_pose"][rows], 0).astype(np.float32).reshape(n, 15, 3)
        ct = np.where(ok[:, None], tr["transl"][rows], 0).astype(np.float32)
        betas = np.median(tr["betas"][rows][ok], 0).astype(np.float32) if ok.any() else np.zeros(10, np.float32)
        from scipy.spatial.transform import Rotation as Rot

        Tw = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        Tw[:, :3, :3] = Rot.from_rotvec(go).as_matrix()
        Tw[:, :3, 3] = ct
        hands[name] = dict(betas=betas, wrist_orient=go, wrist_trans=ct, finger_pose=hp, valid=ok,
                           scale=np.where(ok, tr["scale"][rows], 1).astype(np.float32), wrist_to_camera=Tw,
                           size_factor_median=float(np.median(tr["scale"][rows][ok])) if ok.any() else 1.0)
    # placeholder object: on the table, 0.45 m to the side of the median hand position (world frame)
    wr = tr["joints_cam"].reshape(-1, 2, 21, 3)[:, :, 0].astype(float)
    ok_any = tr.scored_mask().reshape(-1, 2)
    hand_w = (np.nanmedian(wr[ok_any], 0) @ T_w_c[:3, :3].T) + T_w_c[:3, 3]
    obj_w = np.array([hand_w[0] + 0.45, hand_w[1], 0.01])
    T_o_w = np.eye(4, dtype=np.float32)
    T_o_w[:3, 3] = obj_w
    # object_to_world = camera_to_world @ object_to_camera  =>  object_to_camera = camera_to_world^-1 @ object_to_world
    o2c = (np.linalg.inv(c2w) @ T_o_w).astype(np.float32)
    cube_obj(out / "mesh.obj")
    K = A.K
    np.savez(out / "result.npz",
             camera_to_world_transform=np.tile(c2w.astype(np.float32), (n, 1, 1)),
             camera_is_valid=np.ones(n, bool),
             camera_intrinsics=np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], np.float32),
             object_to_camera_transform=np.tile(o2c, (n, 1, 1)),
             object_to_world_transform=np.tile(T_o_w, (n, 1, 1)),
             object_scale=np.float32(1.0), object_is_valid=np.ones(n, bool),
             **{f"hand_{s}_{k}": v for s, h in hands.items() for k, v in (
                 ("betas", h["betas"]), ("wrist_orient_in_camera", h["wrist_orient"]),
                 ("wrist_trans_in_camera", h["wrist_trans"]), ("wrist_to_camera_transform", h["wrist_to_camera"]),
                 ("wrist_to_world_transform", np.einsum("ij,njk->nik", c2w.astype(np.float32), h["wrist_to_camera"])),
                 ("finger_pose", h["finger_pose"]), ("scale", h["scale"]), ("is_valid", h["valid"]))})
    manifest = {"schema_version": 1, "n_frames": int(n), "result_npz": "result.npz", "camera_pose_convention":
                "camera_to_world", "mesh": {"path": "mesh.obj"},
                "sources": {"handbench_method": a.method, "episode": ep, "world": "fixed head camera (frame 0), "
                            "table-plane gravity alignment, no visual odometry", "table_fit": tinfo,
                            "placeholder_object": "2 cm cube, static", "hand_size_factor_ignored_by_loader":
                            {s: h["size_factor_median"] for s, h in hands.items()}}}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(json.dumps({"clip": clip, "out": str(out), "frames": int(n), **tinfo,
                      "valid_frames": {s: int(h["valid"].sum()) for s, h in hands.items()},
                      "size_factor": {s: round(h["size_factor_median"], 3) for s, h in hands.items()}}, indent=1))


if __name__ == "__main__":
    main()
