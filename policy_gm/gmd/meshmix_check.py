"""Smoke test of per-world mesh mixing (meshmix.py) on one episode, static part: in the solver's per-world MuJoCo geoms the
hulls of the unused variant are parked and the used ones are not; the S0 observation's shape block carries each world's
variant; the drop test (zero action for 2 s) keeps every object near its spawn in both variants.
  GMD_MESHMIX=<spec> python meshmix_check.py <checkpoint> [--worlds 16]
The teacher's tracking per variant comes from rollout.py nominal + meshmix_report.py.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/env/gmd")
import meshmix  # noqa: E402
import rollout  # noqa: E402  (installs meshmix)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("checkpoint"); ap.add_argument("--worlds", type=int, default=16)
    args = ap.parse_args()
    assert meshmix.active(), "set GMD_MESHMIX"
    W = args.worlds
    cfg, metadata, training, setup, env, jax_env, thr = rollout.build(args.checkpoint, W)
    var = meshmix.variants(W)
    gpos = env.solver.mjw_model.geom_pos.numpy()                          # (W, ngeom, 3)
    g2s = env.solver.mjc_geom_to_newton_shape.numpy()                     # (W, ngeom) Newton shape id
    nshape = env.model.shape_count // W
    ok = True
    for b, binding in enumerate(setup.scene.objects):
        if binding.name not in meshmix.HULLS:
            continue
        for k, hulls in enumerate(meshmix.HULLS[binding.name]):
            ids = {binding.shapes.start + s for s in hulls}
            for w in range(W):
                g = [j for j in range(g2s.shape[1]) if g2s[w, j] >= 0 and g2s[w, j] - w * nshape in ids]
                d = np.linalg.norm(gpos[w, g], axis=-1)
                parked = d.min() > 50
                want = var[w, b] != (1 - k)                                   # k = 0: v2 hulls, used when var == 0
                if parked == want or len(g) != len(hulls):
                    ok = False
                    print(f"FAIL {binding.name} world {w} variant {var[w, b]} {'v1' if k else 'v2'} hulls: {len(g)} geoms, "
                          f"min |pos| {d.min():.2f}", flush=True)
        print(f"{binding.name}: v1 in worlds {np.flatnonzero(var[:, b]).tolist()}", flush=True)
    print(f"geom parking per world: {'OK' if ok else 'FAILED'}", flush=True)

    strat = env.observation_strategy
    obs = strat.compute(env.state_0).numpy().reshape(W, -1)
    s, e = strat.block_ranges[list(strat.block_names).index("object_shape_bps32")]
    want = meshmix.SPEC["bps_var"][var, np.arange(var.shape[1])[None]].reshape(W, -1)
    err = np.abs(obs[:, s:e] - want).max()
    print(f"observation BPS per world: max error {err:.2e} ({'OK' if err < 1e-5 else 'FAILED'}), "
          f"v1/v2 rows differ: {bool(np.abs(want[var[:, 0] + var[:, -1] > 0] - want[0]).max() > 0.1) if var.any() else 'no v1 world'}", flush=True)

    ids = [int(bd.root.body_id) for bd in setup.scene.objects]
    Bw = env.model.body_count // W
    p0 = env.state_0.body_q.numpy().reshape(W, Bw, 7)[:, ids, :3].copy()
    for _ in range(40):
        env.step()
    p1 = env.state_0.body_q.numpy().reshape(W, Bw, 7)[:, ids, :3]
    drift = np.linalg.norm(p1 - p0, axis=-1)
    for b, bd in enumerate(setup.scene.objects):
        for v in (0, 1):
            m = var[:, b] == v
            if m.any():
                print(f"drop test {bd.name} {'v1' if v else 'v2'} worlds: drift over 2 s mean {100 * drift[m, b].mean():.2f} cm, "
                      f"max {100 * drift[m, b].max():.2f} cm, finite {bool(np.isfinite(p1[m]).all())}", flush=True)


if __name__ == "__main__":
    main()
