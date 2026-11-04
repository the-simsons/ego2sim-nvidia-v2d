"""Package Ego2Sim reconstructions of PUBLIC Track 3 episodes together with their motion-capture ground truth.

  build_share_public.py OUT_DIR EP:OBJ1,OBJ2 [EP:OBJ ...]

Same layout as build_share_package.py (sim_reference/, assets/, meshes/, trajectories/, raw/, quality.csv) plus
ground truth in trajectories/epEP.npz:
  gt_position, gt_wxyz, gt_visible     motion capture as released (dataset world frame, object frame = the scan's)
  gt_sim_position, gt_sim_wxyz         the same, moved into the simulator frame by ONE rigid transform taken from the
                                       anchor object's frame-0 pose (reference pose x inverse ground-truth pose), as the
                                       Kaggle scorer aligns a scene; anchor = first non-cup object (cups are symmetric)
The tracks used the public scan meshes, so reference and ground truth share each object's body frame.
Run with the kit env (pyarrow, numpy, scipy).
"""
import csv
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

H = Path.home() / "v2d"
W = H / "ego2sim"
ASSETS = H / "vtd_pinned/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed"
DATA = H / "v2d_data/track_3/public"
REFONLY_AUC = {"2": 0.8893, "11": 0.5666, "13": 0.9502, "18": 0.7856, "21": 0.8458, "23": 0.0358, "26": 0.9211,
               "40": 0.9608, "41": 0.9812}  # scripts/score_refonly.sh, kit metric code, 2026-10-16


def set_col(t, name, value):
    i = t.column_names.index(name)
    return t.set_column(i, t.schema.field(i), pa.array([value] * t.num_rows, type=t.schema.field(i).type))


def T_of(p, q_wxyz):
    T = np.eye(4); w, x, y, z = q_wxyz
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix(); T[:3, 3] = p
    return T


def main():
    out = Path(sys.argv[1]).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    eps = [(a.split(":")[0], a.split(":")[1].split(",")) for a in sys.argv[2:]]
    (out / "assets").mkdir(parents=True); (out / "meshes").mkdir()
    for o in sorted({o for _, ns in eps for o in ns}):
        for suf in (".obj", "_rigid.urdf", "_visual.stl"):
            shutil.copy2(ASSETS / f"{o}{suf}", out / "assets" / f"{o}{suf}")
        scan = DATA / f"mesh/{o}/{o}.glb"
        shutil.copy2(scan if scan.exists() else DATA / f"mesh/{o}/{o}_visual.glb", out / "meshes" / f"{o}.glb")
    qrows = []
    for ep, names in eps:
        sid = f"ep{ep}"
        t = pq.read_table(W / f"train/{sid}/processed/sequence_id={sid}/robot_name=sharpa_wave/{sid}-0.parquet")
        r = t.to_pylist()[0]
        if list(r["object_body_names"]) != names:
            raise SystemExit(f"{sid}: bodies {r['object_body_names']} != {names}")
        t = set_col(t, "object_mesh_paths", [str(out / "assets" / f"{o}.obj") for o in names])
        t = set_col(t, "object_urdf_paths", [str(out / "assets" / f"{o}_rigid.urdf") for o in names])
        t = set_col(t, "raw_motion_file", str(out / "raw" / f"{sid}_{names[0]}" / "result.npz"))
        dst = out / f"sim_reference/processed/sequence_id={sid}/robot_name=sharpa_wave/{sid}-0.parquet"
        dst.parent.mkdir(parents=True); pq.write_table(t, dst)
        stage = out / "sim_reference/reconstructed_stage"; stage.mkdir(parents=True, exist_ok=True)
        shutil.copy2(W / f"train/{sid}/reconstructed_stage/{sid}_sharpa_wave_support.usda", stage)
        P = np.asarray(r["object_body_position"], float); Q = np.asarray(r["object_body_wxyz"], float)
        arr = {"object_names": np.array(names), "fps": np.float32(r["fps"]),
               "object_position": P.astype(np.float32), "object_wxyz": Q.astype(np.float32)}
        for s in ("left", "right"):
            for k in ("trans", "global_orient", "finger_pose", "joints"):
                if r.get(f"mano_{s}_{k}"):
                    arr[f"mano_{s}_{k}"] = np.asarray(r[f"mano_{s}_{k}"], np.float32)
            if r.get(f"mano_{s}_betas"):
                arr[f"mano_{s}_betas"] = np.asarray(r[f"mano_{s}_betas"], np.float32)
            for k in ("wrist_position", "wrist_wxyz", "finger_joints"):
                if r.get(f"robot_{s}_{k}"):
                    arr[f"robot_{s}_{k}"] = np.asarray(r[f"robot_{s}_{k}"], np.float32)
            if r.get(f"{s}_robot_finger_joint_names"):
                arr[f"robot_{s}_finger_joint_names"] = np.array(r[f"{s}_robot_finger_joint_names"])
        # ground truth
        f = [p for p in glob.glob(str(DATA / "data/*/*.parquet")) if p.endswith(f"episode_{int(ep):06d}.parquet")][0]
        rows = pq.read_table(f, columns=["observation.objects"]).column(0).to_pylist()
        n = min(len(rows), len(P))
        gp = np.zeros((n, len(names), 3)); gq = np.zeros((n, len(names), 4)); gv = np.zeros((n, len(names)), bool)
        for i in range(n):
            by = {o["name"]: o for o in rows[i]}
            for k, nm in enumerate(names):
                o = by[nm]; gp[i, k] = o["pose"][:3]; gq[i, k] = o["pose"][3:]; gv[i, k] = bool(o["visible"])
        a = next((k for k, nm in enumerate(names) if "cup" not in nm), 0)
        A = T_of(P[0, a], Q[0, a]) @ np.linalg.inv(T_of(gp[0, a], gq[0, a]))
        sp = gp @ A[:3, :3].T + A[:3, 3]
        sq = np.zeros_like(gq)
        for k in range(len(names)):
            rr = R.from_matrix(A[:3, :3]) * R.from_quat(gq[:, k][:, [1, 2, 3, 0]])
            xyzw = rr.as_quat(); sq[:, k] = xyzw[:, [3, 0, 1, 2]]
        arr.update({"gt_position": gp.astype(np.float32), "gt_wxyz": gq.astype(np.float32), "gt_visible": gv,
                    "gt_sim_position": sp.astype(np.float32), "gt_sim_wxyz": sq.astype(np.float32),
                    "gt_sim_anchor": np.array(names[a])})
        (out / "trajectories").mkdir(exist_ok=True)
        np.savez_compressed(out / "trajectories" / f"{sid}.npz", **arr)
        for o in names:
            b = W / f"bundles/{sid}_{o}/outputs/result"; d = out / "raw" / f"{sid}_{o}"; d.mkdir(parents=True)
            for fn in ("result.npz", "manifest.json", "mesh.obj"):
                shutil.copy2(b / fn, d / fn)
        err = np.linalg.norm(P[:n] - sp, axis=2) * 100
        qrows.append({"episode": ep, "objects": " + ".join(names), "frames": len(P), "gt_frames": len(rows),
                      "reference_only_auc_vs_mocap": REFONLY_AUC.get(ep, ""),
                      "mean_pos_err_cm_per_object": " / ".join(f"{e:.1f}" for e in err.mean(0)),
                      "anchor": names[a]})
        print(f"{sid}: {names}, ref {len(P)} / gt {len(rows)} frames, mean pos err cm {np.round(err.mean(0), 1)}")
    with open(out / "quality.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qrows[0])); w.writeheader(); w.writerows(qrows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
