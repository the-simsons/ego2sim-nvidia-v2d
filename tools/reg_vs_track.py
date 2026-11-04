"""Why does a reference built with OUR mesh score worse than one built with the scan? Splits the gap into
(a) the scorer's mesh->scan registration (AUC.py _register_object_body, same as Kaggle) and (b) tracking.

  python reg_vs_track.py KIT REF_SCAN_PARQUET REF_OURS_PARQUET OBJ OUR_MESH SCAN_MESH [EPISODE SLOT]
Both parquets are FlashCHORD references (ManoSharpaData) of the same episode: one tracked with the scan, one with our
mesh. If both tracks are right, world_T_scan(t) = world_T_ours(t) @ X for one constant X (ours <- scan body frames).
Prints the spread of X over frames (tracking disagreement) and the angle / anchor offset between X and the scorer's
registration (registration error).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation


def poses(p, obj):
    r = pq.read_table(p, columns=["object_body_names", "object_body_position", "object_body_wxyz"]).to_pylist()[0]
    b = r["object_body_names"].index(obj)
    P = np.asarray(r["object_body_position"], float)[:, b]
    Q = np.asarray(r["object_body_wxyz"], float)[:, b]
    return P, Rotation.from_quat(Q[:, [1, 2, 3, 0]])


def main():
    kit, ref_s, ref_o, obj, mesh_o, mesh_s = sys.argv[1:7]
    ep, slot = (int(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) > 8 else (0, 0)
    sys.path.insert(0, kit)
    from v2dlb.mesh_budget import budget_mesh
    spec = importlib.util.spec_from_file_location("auc", Path(kit) / "metric_code/track_3/AUC.py")
    auc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auc)

    Ps, Rs = poses(ref_s, obj)
    Po, Ro = poses(ref_o, obj)
    T = min(len(Ps), len(Po))
    Ps, Rs, Po, Ro = Ps[:T], Rs[:T], Po[:T], Ro[:T]
    Xr = Ro.inv() * Rs  # ours_R_scan per frame
    Xt = Ro.inv().apply(Ps - Po)  # scan origin in ours frame per frame
    Xr_med = Xr.mean()
    spread = np.degrees((Xr_med.inv() * Xr).magnitude())
    Xt_med = np.median(Xt, axis=0)
    tspread = np.linalg.norm(Xt - Xt_med, axis=1) * 100
    # how much does the object rotate in this clip (lever for a body-frame error)
    rot = np.degrees((Rs[0].inv() * Rs).magnitude())
    print(f"{obj}: {T} frames, object rotates up to {rot.max():.0f} deg (p90 {np.percentile(rot, 90):.0f})")
    print(f"  tracking: X rotation spread median {np.median(spread):.1f} p90 {np.percentile(spread, 90):.1f} deg; "
          f"translation spread median {np.median(tspread):.1f} p90 {np.percentile(tspread, 90):.1f} cm")

    # scorer registration, exactly as AUC.py does it (reduced scan, budgeted candidate)
    m = trimesh.load(mesh_s, force="mesh")
    if len(m.faces) > 20000:
        m = m.simplify_quadric_decimation(face_count=20000)
    anchor = np.array(m.center_mass if m.is_watertight else m.centroid, float)
    v, f = budget_mesh(mesh_o, faces=4096, vertices=4096)
    cand = (np.asarray(v, float), np.asarray(f, np.int64))
    ref = (np.asarray(m.vertices, float), np.asarray(m.faces, np.int64))
    cand_anchor, R = auc._register_object_body(cand, ref, anchor.copy(), 1_000_003 * ep + 101 * slot)
    # scorer: world_R_scan = world_R_ours @ R^T -> ours_R_scan = R^T; scan anchor in ours frame = cand_anchor
    Rreg = Rotation.from_matrix(R.T)
    err = np.degrees((Xr_med.inv() * Rreg).magnitude())
    anchor_track = Xt_med + Xr_med.apply(anchor)  # scan anchor in ours frame, from the tracks
    # the scorer rescales ours by RMS radius before ICP, so compare anchors in ours units
    print(f"  registration vs tracks: rotation off by {err:.1f} deg, anchor off by "
          f"{np.linalg.norm(cand_anchor - anchor_track) * 100:.1f} cm")
    # rotation error decomposed: is it a flip about a principal axis?
    print(f"  registration rotvec (deg) relative to tracks: {np.round(np.degrees((Xr_med.inv() * Rreg).as_rotvec()), 1)}")


if __name__ == "__main__":
    main()
