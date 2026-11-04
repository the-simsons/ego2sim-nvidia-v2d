"""Package the Ego2Sim reconstruction outputs of the 20 Track 3 evaluation episodes for another user of riddle.

  build_share_package.py OUT_DIR                                  (v1: train/epEP, ego2sim/meshes, policy MPPE)
  build_share_package.py OUT_DIR --tag _v2 --meshes meshes_final --summary ~/v2d/logs/eval_v2/summary.tsv
      (v2: train/epEP_v2; quality.csv gets FP silhouette IoU + frame-0 action from the rebuild summary instead)
The shared simulator assets must be the --meshes set (checked per .obj geometry), since the references point at them by name.

OUT_DIR/
  README.md, quality.csv, relocate.py
  assets/<obj>.obj, <obj>_rigid.urdf, <obj>_visual.stl     simulator assets (body frame of every pose below)
  meshes/<obj>.glb                                          the same meshes with vertex colours (viewing / CD-O)
  sim_reference/processed/sequence_id=epEP/robot_name=sharpa_wave/epEP-0.parquet   ManoSharpaData, paths -> assets/
  sim_reference/reconstructed_stage/epEP_sharpa_wave_support.usda                  table support surface
  trajectories/epEP.npz                                     plain arrays: objects, MANO hands, retargeted Sharpa hands
  raw/epEP_<obj>/{result.npz,manifest.json,mesh.obj}        upstream ego-recon bundles (camera + world frame)
Run with the kit env (pyarrow, numpy).
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

H = Path.home() / "v2d"
W = H / "ego2sim"
ASSETS = H / "vtd_pinned/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed"
ROSTER = H / "submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json"
RESULTS = H / "logs/e2s_results_eval.tsv"


def set_col(t, name, value):
    i = t.column_names.index(name)
    return t.set_column(i, t.schema.field(i), pa.array([value] * t.num_rows, type=t.schema.field(i).type))


def same_geometry(a, b):  # the loader re-exports assets (vertices split per face): compare unique positions, face count
    def read(p):
        v = [l.split()[1:4] for l in open(p) if l.startswith("v ")]
        return np.unique(np.round(np.array(v, float), 6), axis=0), sum(l.startswith("f ") for l in open(p))
    (va, fa), (vb, fb) = read(a), read(b)
    return fa == fb and va.shape == vb.shape and np.allclose(np.sort(va, 0), np.sort(vb, 0), atol=1e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tag", default="")
    ap.add_argument("--meshes", default="meshes")
    ap.add_argument("--summary")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    roster = json.load(open(ROSTER))["episodes"]
    objs = sorted({o for v in roster.values() for o in v})
    mesh_dir = W / a.meshes
    if a.meshes != "meshes":  # v1 assets were installed from meshes/ by name; later sets are checked byte for byte
        bad = [o for o in objs if not same_geometry(ASSETS / f"{o}.obj", mesh_dir / f"{o}.obj")]
        if bad:
            raise SystemExit(f"shared assets differ from {mesh_dir}: {bad}")
    (out / "assets").mkdir(parents=True)
    (out / "meshes").mkdir()
    for o in objs:
        for suf in (".obj", "_rigid.urdf", "_visual.stl"):
            shutil.copy2(ASSETS / f"{o}{suf}", out / "assets" / f"{o}{suf}")
        shutil.copy2(mesh_dir / f"{o}.glb", out / "meshes" / f"{o}.glb")
    best, summ = {}, {}
    if a.summary:
        for row in csv.DictReader(open(a.summary), delimiter="\t"):
            summ[row["episode"]] = row
    else:
        for row in csv.DictReader(open(RESULTS), delimiter="\t"):
            try:
                m = float(row["ref_mppe_cm"])
            except (TypeError, ValueError):
                continue
            ep = row["episode"]
            if ep not in best or m < best[ep]:
                best[ep] = m
    qrows = []
    for ep, names in sorted(roster.items(), key=lambda kv: int(kv[0])):
        sid = f"ep{ep}"
        src = W / f"train/{sid}{a.tag}/processed/sequence_id={sid}/robot_name=sharpa_wave/{sid}-0.parquet"
        t = pq.read_table(src)
        r = t.to_pylist()[0]
        if list(r["object_body_names"]) != list(names):
            raise SystemExit(f"{sid}: bodies {r['object_body_names']} != roster {names}")
        t = set_col(t, "object_mesh_paths", [str(out / "assets" / f"{o}.obj") for o in names])
        t = set_col(t, "object_urdf_paths", [str(out / "assets" / f"{o}_rigid.urdf") for o in names])
        t = set_col(t, "raw_motion_file", str(out / "raw" / f"{sid}_{names[0]}" / "result.npz"))
        dst = out / f"sim_reference/processed/sequence_id={sid}/robot_name=sharpa_wave/{sid}-0.parquet"
        dst.parent.mkdir(parents=True)
        pq.write_table(t, dst)
        stage = out / "sim_reference/reconstructed_stage"
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copy2(W / f"train/{sid}{a.tag}/reconstructed_stage/{sid}_sharpa_wave_support.usda", stage)
        arr = {"object_names": np.array(names), "fps": np.float32(r["fps"]),
               "object_position": np.asarray(r["object_body_position"], np.float32),
               "object_wxyz": np.asarray(r["object_body_wxyz"], np.float32)}
        for s in ("left", "right"):
            for k in ("trans", "global_orient", "finger_pose", "joints"):
                v = r.get(f"mano_{s}_{k}")
                if v:
                    arr[f"mano_{s}_{k}"] = np.asarray(v, np.float32)
            if r.get(f"mano_{s}_betas"):
                arr[f"mano_{s}_betas"] = np.asarray(r[f"mano_{s}_betas"], np.float32)
            for k in ("wrist_position", "wrist_wxyz", "finger_joints"):
                v = r.get(f"robot_{s}_{k}")
                if v:
                    arr[f"robot_{s}_{k}"] = np.asarray(v, np.float32)
            if r.get(f"{s}_robot_finger_joint_names"):
                arr[f"robot_{s}_finger_joint_names"] = np.array(r[f"{s}_robot_finger_joint_names"])
        (out / "trajectories").mkdir(exist_ok=True)
        np.savez_compressed(out / "trajectories" / f"{sid}.npz", **arr)
        for o in names:
            b = W / f"bundles/{sid}_{o}/outputs/result"
            d = out / "raw" / f"{sid}_{o}"
            d.mkdir(parents=True)
            for f in ("result.npz", "manifest.json", "mesh.obj"):
                shutil.copy2(b / f, d / f)
        n = len(arr["object_position"])
        hands = [s for s in ("left", "right") if f"mano_{s}_trans" in arr]
        q = {"episode": ep, "objects": " + ".join(names), "frames": n, "hands": " + ".join(hands)}
        if a.summary:
            s = summ[ep]
            q["fp_silhouette_iou_median"] = s["iou_median"].replace(",", " + ")
            q["frame0"] = " | ".join("smoothed" if f.startswith("frame 0") else "kept (moving)"
                                     for f in s["frame0"].strip(";").split(";"))
        else:
            q["our_policy_mppe_vs_this_reference_cm"] = f"{best[ep]:.2f}" if ep in best else ""
        qrows.append(q)
        print(f"{sid}: {names}, {n} frames, hands {hands}")
    with open(out / "quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(qrows[0]))
        w.writeheader(); w.writerows(qrows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
