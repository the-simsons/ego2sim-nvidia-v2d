"""Table plane from motion capture, for the mocap-assisted package (ported from the 2026-11-08 AIMS mocap_table.py).

The table height is the lowest box corner while the box rests at the start of the take (mean pose over the first
0.5 s). The plane is written as a frame-0 depth image (v2d encoding) seen from the mocap-assisted camera, at
<outputs_dir>/stereo/stereo_depth_table/000000.png, which build_ego_bundle.py fits its table plane to. Also reports
monocular depth (MoGe) against the mocap box distance.

    python -m e2s.mocap.assist.table --clip BlueBox_Take1_000 [--root DATA] [--link]
--link points <outputs_dir>/stereo/stereo_depth at the table depth (the 2026-11-08 package layout)."""
import argparse
import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from bench import manifest as mf
from e2s.geometry import load_pose_dir
from e2s.mocap.assist.common import Clip


def decode(p):
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = 65535.0 / np.maximum(raw, 1.0) - 1.0
    d[(raw >= 65535) | (raw <= 0)] = 0
    return d


def table_height(c, rest_s=0.5):
    n = max(1, int(round(rest_s * c.mocap_fps)))
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat(c.qm[:n]).mean().as_matrix()
    T[:3, 3] = c.pm[:n].mean(0)
    Twb = T @ c.T_rb_mesh
    return float(((Twb[:3, :3] @ c.corners.T).T + Twb[:3, 3])[:, c.up].min())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--root")
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args(argv)
    c = Clip(mf.load(a.clip), mf.data_root(a.root))
    table = table_height(c)
    c2w = np.load(c.out / "camera_vo.npz")["camera_to_world"]
    u, v = np.meshgrid(np.arange(c.W), np.arange(c.H))
    rays = np.dstack([(u - c.cx) / c.fx, (v - c.cy) / c.fy, np.ones_like(u, float)])
    C, R = c2w[0, :3, 3], c2w[0, :3, :3]
    denom = (rays @ R[c.up, :].reshape(3, 1))[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (table - C[c.up]) / denom
    s[~np.isfinite(s) | (s <= 0) | (s > 20)] = 0
    out = c.out / "stereo" / "stereo_depth_table"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "000000.png"), np.clip(65535.0 / (s + 1.0), 0, 65535).astype(np.uint16))
    if a.link:
        link = c.out / "stereo" / "stereo_depth"
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(out)
    ratios = []
    depth_dir = c.root / c.m["inputs"]["depth"]
    Tcb = load_pose_dir(c.out / "fp_mocap")
    for i in range(0, len(c2w), 10):
        m = c.mask(i) > 0
        m = cv2.erode(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        dm = decode(depth_dir / f"{i:06d}.png")
        if m.sum() < 500 or (dm[m] <= 0).all():
            continue
        cz = ((Tcb[i, :3, :3] @ c.corners.T).T + Tcb[i, :3, 3])[:, 2]
        ratios.append(np.median(dm[m][dm[m] > 0]) / (0.5 * (cz.min() + np.median(cz))))
    r = np.array(ratios)
    rec = {"table_height_m": table, "camera_height_above_table_m": float(C[c.up] - table),
           "monocular_to_mocap_depth_ratio_median": float(np.median(r)) if len(r) else None}
    json.dump(rec, open(out / "table.json", "w"), indent=1)
    print(f"{c.name}: table height {table:.4f} m (manifest {c.m['ground_truth'].get('table_z')}); camera {C[c.up] - table:.3f} m "
          f"above it; plane valid {(s > 0).mean() * 100:.0f}%; MoGe/mocap depth ratio median {rec['monocular_to_mocap_depth_ratio_median']}")


if __name__ == "__main__":
    main()
