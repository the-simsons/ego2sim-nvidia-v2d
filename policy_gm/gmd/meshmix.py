"""Per-world v1/v2 mesh mixing, step 2 (inside the FlashCHORD container): install() when $GMD_MESHMIX = a meshmix_prep.py spec.

FlashCHORD (read-only) is patched at three points, nothing is copied:
  objects._decompose_object_collision  a mixed object gets the CoACD hulls of BOTH meshes in the one scene copy (the v1 mesh
                                       is already registered into the v2 body frame); all its hulls are filtered pairwise.
  builder._apply_object_scales         after replication, each world parks the hulls of the variant it does not use 100 m
                                       (+ 50 m per object index) from the body origin, in the body frame. Newton's MuJoCo
                                       solver builds the mjModel from world 0 only, but copies each world's shape transforms
                                       into geom_pos per world, so this is the per-world switch. Needs collision.ground=false
                                       (a parked hull would otherwise hit the infinite floor plane once the object tilts).
  S0Observation.compute_into           the object_shape_bps32 block is overwritten per world with that world's variant.
G's features read the per-world variant through meta_extras() (genv.scene_meta, rollout.static_meta).

Variant per world and object: v1 with probability p_v1 for mixed objects, world 0 always v2; fixed for the run (seeded, one
draw per world count, so a second scene in the same process gets the same assignment).
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np

SPEC = None
VARS: dict[int, np.ndarray] = {}      # world count -> (W, B) int, 0 = v2 (the episode's own mesh), 1 = v1
HULLS: dict[str, tuple] = {}          # object name -> (v2 hull ids, v1 hull ids), component-local
_CUR = [None]
PARK, PARK_STEP = 100.0, 50.0


def active():
    return SPEC is not None


def install():
    global SPEC
    path = os.environ.get("GMD_MESHMIX")
    if not path or SPEC is not None:
        return
    SPEC = {k: v for k, v in np.load(path, allow_pickle=True).items()}
    import flash_chord.scene.builder as fb
    import flash_chord.scene.objects as fo

    spawn, decompose, scales = fo._spawn_component, fo._decompose_object_collision, fb._apply_object_scales

    def spawn_named(destination, asset, *args, **kwargs):
        _CUR[0] = asset.name
        try:
            return spawn(destination, asset, *args, **kwargs)
        finally:
            _CUR[0] = None

    def decompose_both(builder, imported_shapes, **coacd_params):
        start = builder.shape_count
        n = decompose(builder, imported_shapes, **coacd_params)
        name = _CUR[0]
        names = [str(x) for x in SPEC["names"]]
        if name not in names or not SPEC["alt"][names.index(name)]:
            return n
        import newton
        import trimesh
        from flash_chord.assets.mesh import convex_decompose
        v2 = list(range(start, builder.shape_count))
        h0 = v2[0]
        mat = {f: float(getattr(builder, f"shape_material_{f}")[h0]) for f in ("ke", "kd", "kf", "ka", "mu", "restitution")
               if hasattr(builder, f"shape_material_{f}")}
        cfg = dataclasses.replace(builder.default_shape_cfg, is_visible=False, has_shape_collision=True, density=0.0, **mat)
        m = trimesh.load(str(SPEC["v1_mesh"][names.index(name)]), process=False, force="mesh")
        v1 = []
        for hull in convex_decompose(np.asarray(m.vertices), np.asarray(m.faces), **coacd_params):
            mesh = newton.Mesh(hull.vertices.astype(np.float32), hull.faces.reshape(-1).astype(np.int32), compute_inertia=False)
            v1.append(builder.add_shape_mesh(body=builder.shape_body[h0], xform=builder.shape_transform[h0], mesh=mesh,
                                             scale=builder.shape_scale[h0], cfg=cfg))
        existing = {tuple(sorted(p)) for p in builder.shape_collision_filter_pairs}
        allh = v2 + v1
        for i, a in enumerate(allh):
            for b in allh[i + 1:]:
                if (min(a, b), max(a, b)) not in existing:
                    builder.add_shape_collision_filter_pair(min(a, b), max(a, b))
        HULLS[name] = (tuple(v2), tuple(v1))
        print(f"[meshmix] {name}: {len(v2)} v2 hulls + {len(v1)} v1 hulls", flush=True)
        return n + len(v1)

    def scales_then_park(builder, copy, objects, reference, object_scales, **kwargs):
        out = scales(builder, copy, objects, reference, object_scales, **kwargs)
        W = object_scales.shape[0]
        var = variants(W)
        nshape = copy.shape_count
        for b, binding in enumerate(objects):
            if binding.name not in HULLS:
                continue
            v2, v1 = HULLS[binding.name]
            for w in range(W):
                for s in (v1 if var[w, b] == 0 else v2):
                    sid = w * nshape + binding.shapes.start + s
                    t = builder.shape_transform[sid]
                    vals = [float(t[i]) for i in range(7)]
                    vals[:3] = [0.0, 0.0, PARK + PARK_STEP * b]
                    builder.shape_transform[sid] = type(t)(*vals)
            print(f"[meshmix] {binding.name}: v1 in {int(var[:, b].sum())}/{W} worlds", flush=True)
        return out

    fo._spawn_component = spawn_named
    fo._decompose_object_collision = decompose_both
    fb._apply_object_scales = scales_then_park
    _patch_observation()
    print(f"[meshmix] installed from {path}: mixed {[str(n) for n, a in zip(SPEC['names'], SPEC['alt']) if a]}, "
          f"p_v1 {float(SPEC['p_v1'])}", flush=True)


def variants(W):
    if W not in VARS:
        rng = np.random.default_rng(int(SPEC["seed"]))
        var = ((rng.random((W, len(SPEC["names"]))) < float(SPEC["p_v1"])) & SPEC["alt"][None]).astype(np.int64)
        var[0] = 0
        VARS[W] = var
    return VARS[W]


def _patch_observation():
    try:
        import s0_observation as so
    except ImportError:
        return
    import warp as wp

    @wp.kernel
    def _set_bps(out: wp.array(dtype=wp.float32), out_dim: int, start: int, bps: wp.array(dtype=wp.float32), bps_dim: int):
        w, i = wp.tid()
        out[w * out_dim + start + i] = bps[w * bps_dim + i]

    compute_into = so.S0Observation.compute_into

    def compute_into_mixed(self, state, output):
        output = compute_into(self, state, output)
        a = self.arrays
        if "bps_w" not in a:
            names = list(self.block_names)
            if "object_shape_bps32" not in names:
                a["bps_w"] = None
            else:
                start = self.block_ranges[names.index("object_shape_bps32")][0]
                var = variants(self.world_count)
                B = var.shape[1]
                bps = SPEC["bps_var"][var, np.arange(B)[None]].reshape(self.world_count, -1).astype(np.float32)   # (W, B*32)
                a["bps_w"] = (wp.array(bps.reshape(-1), dtype=wp.float32, device=output.device), start, bps.shape[1])
        if a["bps_w"] is not None:
            arr, start, dim = a["bps_w"]
            wp.launch(_set_bps, dim=(self.world_count, dim), inputs=[output, self.observation_dim, start, arr, dim], device=output.device)
        return output

    so.S0Observation.compute_into = compute_into_mixed


def meta_extras(W):
    """Per-world variant arrays for G's features (gfeat.features): empty when mixing is off."""
    if not active():
        return {}
    from gfeat import surface_normals
    sv = SPEC["surf_var"].astype(np.float64)
    return dict(mesh_var=variants(W), surface_points_var=sv,
                surface_normals_var=np.stack([surface_normals(s) for s in sv]), bps_var=SPEC["bps_var"].astype(np.float32))
