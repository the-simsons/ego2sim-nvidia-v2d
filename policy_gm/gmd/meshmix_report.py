"""Teacher rollouts split by mesh variant (episodes recorded with $GMD_MESHMIX): per variant, the share of worlds label.py keeps
(no object beyond 25 cm / 1.5 rad, mean error < 5 cm) and the mean object position error.
  meshmix_report.py <ep_dir> [setting ...]      (default: nominal perturb)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gfeat import quat_xyzw_to_mat  # noqa: E402


def main(ep_dir, settings):
    m = np.load(ep_dir / "meta.npz", allow_pickle=True)["meta"].item()
    if "mesh_var" not in m:
        print(f"{ep_dir.name}: recorded without mesh mixing"); return
    var = np.asarray(m["mesh_var"]).max(1)                                  # world uses v1 for some object
    for s in settings:
        d = np.load(ep_dir / f"{s}.npz")
        ts = np.clip(d["timestep"], 0, m["num_frames"] - 1); pe = 0; rot = 0
        for j, ob in enumerate(int(b) for b in m["object_body_ids"]):
            po = d["body_q"][:, :, ob, :3].astype(np.float64)
            qa = d["body_q"][:, :, ob, 3:].astype(np.float64); qr = m["ref_obj_quat_wxyz"][ts, j][..., [1, 2, 3, 0]]
            Ra, Rr = quat_xyzw_to_mat(qa), quat_xyzw_to_mat(qr)
            rot = np.maximum(rot, np.arccos(np.clip((np.einsum("twij,twij->tw", Ra, Rr) - 1) / 2, -1, 1)))
            pe = np.maximum(pe, np.linalg.norm(po - m["ref_obj_pos"][ts, j], axis=-1))
        keep = ~((pe > 0.25) | (rot > 1.5)).any(0) & (pe.mean(0) < 0.05)
        for v, name in ((0, "v2"), (1, "v1")):
            w = var == v
            print(f"{ep_dir.name} {s:10s} {name} worlds {w.sum():3d}: kept {keep[w].mean():.2f}, mean object error "
                  f"{100 * pe[:, w].mean():.2f} cm", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:] or ["nominal", "perturb"])
