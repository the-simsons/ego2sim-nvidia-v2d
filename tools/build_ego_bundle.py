"""Build reconstruction bundles (result.npz + mesh.obj + manifest.json) for the task-library loader from OUR pieces:
stereo-VO camera motion (tools/ego_vo.py), FoundationPose object tracks, an object mesh, and (optionally) hand tracks.

Uses the official writer v2d.common.result_bundle.write_result_bundle, so the NPZ schema is exactly the pipeline's.
The loader handles ONE object per bundle, so one clip is written per object:
    OUT_ROOT/<clip>_<object>/outputs/result/{result.npz, mesh.obj, manifest.json}
Then the world (= camera frame 0) is gravity-aligned with the TABLE PLANE fitted to frame-0 stereo depth around the
objects (RANSAC; normal = +z, pointing at the camera). This replaces the loader's "thinnest OBB axis = up" heuristic,
which is wrong for upright objects such as cups -> run the loader with --no_ground_align.

Usage (recon_host env):
  python build_ego_bundle.py --stereo-dir S --cam-npz C --out-root OUT --clip ep12 \
      --object blue_cup=FP_DIR=MESH [--object NAME=FP_DIR=MESH ...] [--exclude-masks DIR,DIR] \
      [--left-hand DIR] [--right-hand DIR] [--no-still-snap]       --trajectory-source video [--mocap-used-for camera_path,sync] [--depth-source stereo]
Every bundle gets a provenance record (manifest.json "provenance" and provenance.json, e2s/provenance.py); policy
training refuses anything but trajectory_source=video for a final result.
Stationary stretches of each object are held at their mean pose (still_snap) unless --no-still-snap.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from v2d.common.result_bundle import write_result_bundle

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e2s import provenance  # noqa: E402

WORLD_KEYS = ("camera_to_world_transform", "object_to_world_transform",
              "hand_left_wrist_to_world_transform", "hand_right_wrist_to_world_transform")


def decode_depth(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = 65535.0 / np.maximum(raw, 1.0) - 1.0
    d[(raw >= 65535) | (raw <= 0)] = 0
    return d


def write_camera_jsons(c2w, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, M in enumerate(c2w):
        q = R.from_matrix(M[:3, :3]).as_quat()  # xyzw
        json.dump({"rotation": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
                   "translation": [float(v) for v in M[:3, 3]], "scale": [1.0, 1.0, 1.0]},
                  open(out_dir / f"{i:06d}.json", "w"))


def mesh_to_obj(mesh_path, out_obj):
    m = trimesh.load(str(mesh_path), force="mesh")
    if not hasattr(m.visual, "vertex_colors") or m.visual.kind is None:
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.tile([180, 180, 180, 255], (len(m.vertices), 1)))
    m.export(str(out_obj))
    return m


def fp_pose0(fp_dir):
    j = json.load(open(Path(fp_dir) / "000000.json"))
    w, x, y, z = j["rotation"]
    T = np.eye(4)
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = j["translation"]
    return T


def table_plane(stereo_dir, centres_cam0, exclude, radius=0.30, iters=2000, thresh=0.006, seed=0):
    """RANSAC plane on frame-0 stereo points within `radius` of the objects (objects/hands masked out)."""
    S = Path(stereo_dir)
    K = json.load(open(S / "intrinsics_undist.json"))
    d = decode_depth(S / "stereo_depth" / "000000.png")
    keep = d > 0.1
    for md in exclude:
        m = cv2.imread(str(Path(md) / "000000.png"), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            keep &= cv2.dilate((m > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
    v, u = np.nonzero(keep)
    z = d[v, u]
    P = np.stack([(u - K["cx"]) * z / K["fx"], (v - K["cy"]) * z / K["fy"], z], 1)
    near = np.min(np.linalg.norm(P[:, None, :] - centres_cam0[None], axis=2), axis=1) < radius
    P = P[near]
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < thresh
        if inl.sum() > best[0]:
            best = (inl.sum(), inl)
    Q = P[best[1]]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c, full_matrices=False)[2][-1]
    if n @ (-c) < 0:  # normal points from the table towards the camera (= up)
        n = -n
    return n, c, int(best[0]), len(P)


def still_snap(T, win=9, pos_tol=0.005, ang_tol_deg=2.0, min_run=8):
    """Hold the pose fixed through stationary stretches. A frame is still when every pose in the centred window lies
    within pos_tol / ang_tol of the window median; runs of >= min_run still frames get their mean pose.
    Our FP+VO tracks jitter ~mm/frame even at rest, which defeats support_recon's mocap-grade stillness test
    (1 mm, 0.01 rad per frame) and would also feed jitter to the policy as reference motion. Returns (T', runs)."""
    n = len(T)
    p = T[:, :3, 3]
    rot = R.from_matrix(T[:, :3, :3])
    still = np.zeros(n, bool)
    h = win // 2
    for t in range(n):
        a, b = max(0, t - h), min(n, t + h + 1)
        med = np.median(p[a:b], axis=0)
        mean_r = rot[a:b].mean()
        if (np.linalg.norm(p[a:b] - med, axis=1).max() < pos_tol
                and np.degrees((rot[a:b] * mean_r.inv()).magnitude()).max() < ang_tol_deg):
            still[t] = True
    out, runs, t = T.copy(), [], 0
    while t < n:
        if still[t]:
            s = t
            while t < n and still[t]:
                t += 1
            if t - s >= min_run:
                out[s:t, :3, 3] = p[s:t].mean(0)
                out[s:t, :3, :3] = rot[s:t].mean().as_matrix()
                runs.append((s, t - 1))
        else:
            t += 1
    return out, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stereo-dir", required=True)
    ap.add_argument("--cam-npz", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--object", action="append", required=True, help="NAME=FP_POSE_DIR=MESH")
    ap.add_argument("--exclude-masks", default="")
    ap.add_argument("--left-hand", default=None)
    ap.add_argument("--right-hand", default=None)
    ap.add_argument("--no-still-snap", action="store_true", help="keep raw object poses through stationary stretches")
    ap.add_argument("--trajectory-source", required=True, choices=provenance.SOURCES,
                    help="video unless motion capture supplied any input (see --mocap-used-for)")
    ap.add_argument("--mocap-used-for", default="", help=f"comma list from {provenance.USES}")
    ap.add_argument("--depth-source", default="stereo")
    a = ap.parse_args()
    prov = provenance.make(a.trajectory_source, [u for u in a.mocap_used_for.split(",") if u],
                           depth_source=a.depth_source, builder="tools/build_ego_bundle.py", clip=a.clip)
    out_root = Path(a.out_root)
    c2w = np.load(a.cam_npz)["camera_to_world"]
    work = out_root / f"{a.clip}_inputs"
    write_camera_jsons(c2w, work / "camera_to_world")
    objs = [o.split("=", 2) for o in a.object]

    # gravity from the table plane at frame 0, around all objects
    centres = []
    for name, fp, mesh in objs:
        m = trimesh.load(str(mesh), force="mesh")
        T0 = fp_pose0(fp)
        centres.append(T0[:3, :3] @ m.bounds.mean(0) + T0[:3, 3])
    n_up, c_tab, n_in, n_pts = table_plane(a.stereo_dir, np.array(centres),
                                           [m for m in a.exclude_masks.split(",") if m])
    Rg = trimesh.geometry.align_vectors(n_up, [0, 0, 1])[:3, :3]
    A = np.eye(4)
    A[:3, :3] = Rg
    A[:3, 3] = -Rg @ c_tab  # table point at frame 0 -> origin (loader re-centres xy and drops the bottom to z=0)
    print(f"table plane: normal (cam0) {np.round(n_up, 3)}, {n_in}/{n_pts} inliers within 30 cm of the objects")

    for name, fp, mesh in objs:
        rdir = out_root / f"{a.clip}_{name}" / "outputs" / "result"
        obj = work / f"{name}.obj"
        mesh_to_obj(mesh, obj)
        npz_path = write_result_bundle(
            result_dir=str(rdir), frames_dir=None,
            intrinsics_path=str(Path(a.stereo_dir) / "intrinsics_undist.json"), mesh_path=str(obj),
            object_poses_dir=fp, camera_to_world_dir=str(work / "camera_to_world"),
            left_hand_dir=a.left_hand, right_hand_dir=a.right_hand,
            source_manifest={"camera_source": f"stereo VO ({a.cam_npz})", "object_track_source": fp,
                             "object_mesh_source": str(mesh)})
        arr = dict(np.load(npz_path))
        for k in WORLD_KEYS:
            arr[k] = np.einsum("ij,njk->nik", A, arr[k].astype(np.float64)).astype(np.float32)
        runs = []
        if not a.no_still_snap:
            o2w, runs = still_snap(arr["object_to_world_transform"].astype(np.float64))
            arr["object_to_world_transform"] = o2w.astype(np.float32)
            arr["object_to_camera_transform"] = np.einsum(
                "nij,njk->nik", np.linalg.inv(arr["camera_to_world_transform"].astype(np.float64)), o2w).astype(np.float32)
            print(f"{name}: still-snap runs (frames) {runs}")
        np.savez(npz_path, **arr)
        man_p = rdir / "manifest.json"
        man = json.load(open(man_p))
        man["gravity_alignment"] = {"enabled": True, "target": "z_up", "method": "stereo_table_plane_frame0",
                                    "alignment_transform": A.tolist(), "table_normal_cam0": n_up.tolist(),
                                    "table_inliers": n_in}
        man["still_snap_runs"] = runs
        man["provenance"] = prov
        json.dump(man, open(man_p, "w"), indent=2)
        provenance.write(rdir.parent.parent, prov)
        valid = arr["object_is_valid"].mean()
        print(f"{name}: {npz_path} ({len(arr['camera_to_world_transform'])} frames, object valid {valid:.0%}, "
              f"hands valid L {arr['hand_left_is_valid'].mean():.0%} R {arr['hand_right_is_valid'].mean():.0%})")


if __name__ == "__main__":
    main()
