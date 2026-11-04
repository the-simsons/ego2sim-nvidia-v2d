"""Per-world v1/v2 mesh mixing, step 1 (host): register each v1 mesh into the v2 body frame and write one spec per episode.

Only 4 objects have different v1 (AgentSTAR, rian_eval_compare/_cleaned) and v2 (meshes_final) meshes: mini_dust_pan,
plastic_dish_rack, water_pitcher, wooden_piece_1/2. Their body frames differ (the rack's axes are swapped), so the v1 mesh is
registered onto the v2 mesh: rigid ICP on surface samples from the 24 axis-aligned starting rotations, best symmetric
chamfer wins, no scaling (each mesh keeps its own size estimate). Then it is shifted along the episode's frame-0 world "up"
(in the body frame) so its lowest point matches the v2 mesh's: the object rests on the table the same way at the spawn.

  meshmix_prep.py <out_dir> <episode>:<recon> ...      recon: v2recon | v2clean | v2clean2 (the episode's M reconstruction)
Writes <out_dir>/ep<N>_<obj>_v1.obj (registered v1 mesh, per episode), <out_dir>/ep<N>.npz (spec read by meshmix.py) and
<out_dir>/ep<N>_<obj>_reg.png (overlay, three views), and prints the registration report.
Spec: names (B,), alt (B,) bool, v1_mesh (B,) container paths ('' if not alt), bps_var (2, B, 32), surf_var (2, B, 96, 3)
(variant 0 = the episode's own extras, variant 1 = v1 mesh where alt, else a copy of variant 0), p_v1, seed.
Episodes with fused objects (1, 28) are not passed: they stay on v2 only.
"""
from __future__ import annotations

import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

O = Path("/usr1/e2s/v2d_challenge_data/outputs")
V1 = O / "rian_eval_compare/_cleaned/package"
PKG = {"v2recon": O / "ego2sim_recon_v2_pkg", "v2clean": O / "ego2sim_recon_v2clean_pkg", "v2clean2": O / "ego2sim_recon_v2clean2_pkg"}
FEAT = {"v2recon": O / "rl_train_req/s0_features_v2", "v2clean": O / "rl_train_req/s0_features_v2clean",
        "v2clean2": O / "rl_train_req/s0_features_v2clean2"}
DIFFER = {"mini_dust_pan", "plastic_dish_rack", "water_pitcher", "wooden_piece_1", "wooden_piece_2"}
P_V1, SEED = 0.5, 0


def bps_from(paths):
    """Same basis / projection / sampling as s0_features_v2clean2.bps_from (copied: that module needs the S0 env)."""
    rng = np.random.default_rng(7)
    basis = rng.normal(size=(512, 3)); basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    basis *= 0.2 * rng.random((512, 1)) ** (1 / 3)
    proj = np.random.default_rng(11).normal(size=(512, 32)) / np.sqrt(512)
    codes, surf = [], []
    for p in paths:
        m = trimesh.load(p, process=False, force="mesh")
        pts = trimesh.sample.sample_surface(m, 20000, seed=0)[0]
        codes.append((cKDTree(pts - (m.bounds[0] + m.bounds[1]) / 2).query(basis)[0] @ proj) / 0.05)
        surf.append(trimesh.sample.sample_surface(m, 96, seed=0)[0].astype(np.float32))
    return np.asarray(codes, np.float32), np.stack(surf)


def icp(src, dst, R, t, iters=60):
    tree = cKDTree(dst)
    for _ in range(iters):
        cur = src @ R.T + t
        _, j = tree.query(cur)
        m = dst[j]
        cs, cm = cur.mean(0), m.mean(0)
        U, _, Vt = np.linalg.svd((m - cm).T @ (cur - cs))
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        dR = U @ D @ Vt
        R, t = dR @ R, dR @ (t - cs) + cm
    return R, t


def chamfer(a, b):
    return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def register(v1_mesh, v2_mesh):
    src = trimesh.sample.sample_surface(v1_mesh, 4000, seed=1)[0]
    dst = trimesh.sample.sample_surface(v2_mesh, 4000, seed=2)[0]
    best = None
    for R0 in Rotation.create_group("O").as_matrix():
        R, t = icp(src, dst, R0, dst.mean(0) - src.mean(0) @ R0.T)
        c = chamfer(src @ R.T + t, dst)
        if best is None or c < best[0]:
            best = (c, R, t)
    return best


def frame0_pose(recon, ep, name):
    import pyarrow.parquet as pq
    f = glob.glob(str(PKG[recon] / f"sim_reference/processed/sequence_id=ep{ep}/robot_name=sharpa_wave/*.parquet"))[0]
    t = pq.read_table(f, columns=["object_body_names", "object_body_wxyz"])
    names = [str(n) for n in t.column("object_body_names")[0].as_py()]
    q = np.asarray(t.column("object_body_wxyz")[0].as_py(), np.float64)[0]           # (B, 4) at frame 0
    return names, q


def plot(path, title, a, b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        ax[k].scatter(b[:, i], b[:, j], s=1, c="#2a78d6", alpha=0.4, label="v2")
        ax[k].scatter(a[:, i], a[:, j], s=1, c="#e34948", alpha=0.4, label="v1 registered")
        ax[k].set_aspect("equal"); ax[k].set_title("xyz"[i] + "xyz"[j])
    ax[0].legend(markerscale=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def main(out, specs):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cont = "/out/" + str(out.relative_to(O / "gmd"))
    for spec in specs:
        ep, recon = spec.split(":"); ep = int(ep)
        ex = np.load(FEAT[recon] / f"episode_{ep:06d}.npz", allow_pickle=True)
        names = [str(n) for n in ex["object_names"]]
        body_names, q0 = frame0_pose(recon, ep, None)
        assert len(body_names) == len(names), (body_names, names)
        bps0, surf0 = ex["bps"].astype(np.float32), ex["object_surface_points"].astype(np.float32)
        chk_b, chk_s = bps_from([PKG[recon] / "assets" / f"{n}.obj" for n in names])
        assert np.allclose(chk_b, bps0, atol=1e-4) and np.allclose(chk_s, surf0, atol=1e-6), f"ep{ep}: recomputed v2 features differ from the extras"
        B = len(names)
        alt = np.array([n in DIFFER for n in names])
        bps1, surf1, paths = bps0.copy(), surf0.copy(), [""] * B
        for b, n in enumerate(names):
            if not alt[b]:
                continue
            m1 = trimesh.load(V1 / "assets" / f"{n}.obj", process=False, force="mesh")
            m2 = trimesh.load(PKG[recon] / "assets" / f"{n}.obj", process=False, force="mesh")
            c, R, t = register(m1, m2)
            v = np.asarray(m1.vertices) @ R.T + t
            up = Rotation.from_quat(q0[b][[1, 2, 3, 0]]).inv().apply([0, 0, 1.0])     # world up in the body frame at frame 0
            shift = (np.asarray(m2.vertices) @ up).min() - (v @ up).min()
            v = v + shift * up
            reg = trimesh.Trimesh(v, np.asarray(m1.faces), process=False)
            p = out / f"ep{ep}_{n}_v1.obj"; reg.export(p)
            paths[b] = f"{cont}/{p.name}"
            bb, ss = bps_from([p]); bps1[b], surf1[b] = bb[0], ss[0]
            a = trimesh.sample.sample_surface(reg, 3000, seed=3)[0]; d = trimesh.sample.sample_surface(m2, 3000, seed=4)[0]
            ang = np.degrees(Rotation.from_matrix(R).magnitude())
            print(f"ep{ep} {n}: chamfer {100 * chamfer(a, d):.2f} cm, rotation {ang:.0f} deg, bottom shift {100 * shift:+.2f} cm, "
                  f"extent v1 {np.round(np.ptp(v, 0), 3).tolist()} v2 {np.round(np.ptp(np.asarray(m2.vertices), 0), 3).tolist()}, "
                  f"BPS distance v1-v2 {np.linalg.norm(bps1[b] - bps0[b]):.2f}", flush=True)
            plot(out / f"ep{ep}_{n}_reg.png", f"ep{ep} {n}: chamfer {100 * chamfer(a, d):.2f} cm", a, d)
        np.savez(out / f"ep{ep}.npz", names=np.asarray(names), alt=alt, v1_mesh=np.asarray(paths), bps_var=np.stack([bps0, bps1]),
                 surf_var=np.stack([surf0, surf1]), p_v1=P_V1, seed=SEED, recon=recon)
        print(f"ep{ep} ({recon}): objects {names}, mixed {[n for n, a in zip(names, alt) if a]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
