"""Simplify a mesh ourselves before packing (the packer's fast_simplification can wreck thin parts), several ways.

  python presimplify.py IN OUT_DIR [--faces 4000]
Writes OUT_DIR/<stem>_<variant>.obj for: pm_qec (MeshLab quadric edge collapse, topology + boundary preserved, planar
quadrics), pm_qec_q (same, higher quality threshold, normal check), pm_remesh (isotropic remesh to the face budget, then
qec), fs_agg3 (fast_simplification, aggressiveness 3 instead of the default 7). Run with envs/meshclean.
"""
import argparse
from pathlib import Path

import numpy as np
import pymeshlab as ml
import trimesh


def qec(ms, faces, **kw):
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=faces, preservetopology=True, preserveboundary=True,
                                                 planarquadric=True, autoclean=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--faces", type=int, default=4000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.inp).stem
    m = trimesh.load(a.inp, force="mesh", process=True)
    tmp = out / f"{stem}_in.obj"
    m.export(tmp)

    def run(name, fn):
        ms = ml.MeshSet()
        ms.load_new_mesh(str(tmp))
        fn(ms)
        p = out / f"{stem}_{name}.obj"
        ms.save_current_mesh(str(p))
        r = trimesh.load(p, force="mesh", process=False)
        print(f"  {name:10s} {len(r.faces):6d} faces, {len(r.vertices):6d} verts -> {p}")

    run("pm_qec", lambda ms: qec(ms, a.faces))
    run("pm_qec_q", lambda ms: qec(ms, a.faces, qualitythr=0.5, preservenormal=True, optimalplacement=True))
    edge = np.sqrt(4 * m.area / a.faces / np.sqrt(3)) * 0.8  # edge length giving ~1.5x the budget, then decimate

    def remesh(ms):
        ms.meshing_isotropic_explicit_remeshing(targetlen=ml.PureValue(edge), iterations=4)
        qec(ms, a.faces)
    run("pm_remesh", remesh)
    import fast_simplification
    v, f = fast_simplification.simplify(m.vertices, m.faces, target_count=a.faces, agg=3)
    p = out / f"{stem}_fs_agg3.obj"
    trimesh.Trimesh(v, f).export(p)
    print(f"  fs_agg3    {len(f):6d} faces -> {p}")
    tmp.unlink()


if __name__ == "__main__":
    main()
