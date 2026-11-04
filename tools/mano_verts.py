"""MANO hand meshes for every frame of a reference parquet (flat hand mean, no centring, per-frame shift onto the stored
joints) -> NPZ {left_V, left_F, right_V, right_F}. Run in the v2d_hamer env.   mano_verts.py PARQUET OUT.npz"""
import os, sys
import numpy as np, pyarrow.parquet as pq, torch
from manotorch.manolayer import ManoLayer
r = pq.read_table(sys.argv[1]).to_pylist()[0]
out = {}
for side in ("left", "right"):
    if not r.get(f"mano_{side}_trans"):
        continue
    m = ManoLayer(rot_mode="axisang", use_pca=False, side=side, center_idx=None, flat_hand_mean=True,
                  mano_assets_root=os.path.expanduser("~/v2d/mano"))
    go = torch.tensor(r[f"mano_{side}_global_orient"], dtype=torch.float32)
    fp = torch.tensor(r[f"mano_{side}_finger_pose"], dtype=torch.float32)
    b = torch.tensor(r[f"mano_{side}_betas"], dtype=torch.float32)[None].repeat(len(go), 1)
    with torch.no_grad():
        o = m(torch.cat([go, fp], 1), b)
    tr = np.asarray(r[f"mano_{side}_trans"], float)[:, None]
    V, J = o.verts.numpy() + tr, o.joints.numpy() + tr
    V = V + (np.asarray(r[f"mano_{side}_joints"], float) - J).mean(1, keepdims=True)
    F = m.th_faces.numpy()
    if side == "left":
        F = F[:, [0, 2, 1]]  # keep outward winding for the mirrored model's back-face culling
    out[f"{side}_V"], out[f"{side}_F"] = V.astype(np.float32), F.astype(np.int32)
np.savez_compressed(sys.argv[2], **out)
print(sys.argv[2], {k: v.shape for k, v in out.items()})