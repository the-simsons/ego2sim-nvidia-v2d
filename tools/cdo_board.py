"""Local CD-O scoreboard: every eval (episode, object) pair scored with the kit's own CD-O code at Kaggle's seeds, against
the PUBLIC scan (proxy for the hidden reference), averaged over all pairs like Kaggle (mean over the 36 pairs).

  python cdo_board.py KIT MESH_DIR [--set NAME=PATH ...] [--ref packed|raw] [--spoon CM] [--json OUT]
MESH_DIR holds <obj>.glb or <obj>.obj; --set overrides single objects (e.g. to try a cleaned mesh). wooden_spoon has no
public scan: it enters the total as the constant --spoon (default 0.177 cm, implied by the first Kaggle CD-O 0.320).
Reference = the scan packed to 4096/4096 like a submission (default, the setting of all earlier local numbers) or raw.
Run with envs/kit.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

SCANS = Path.home() / "v2d/v2d_data/track_3/public/mesh"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find(d, n):
    for c in (Path(d) / f"{n}.glb", Path(d) / f"{n}.obj", Path(d) / f"{n}_visual.glb"):
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kit")
    ap.add_argument("mesh_dir")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--ref", choices=["packed", "raw"], default="packed")
    ap.add_argument("--spoon", type=float, default=0.177)
    ap.add_argument("--json")
    ap.add_argument("--only", help="comma-separated objects to score (no total)")
    a = ap.parse_args()
    kit = Path(a.kit)
    sys.path.insert(0, str(kit))
    from v2dlb.mesh_budget import budget_mesh
    cdo = load(kit / "metric_code/track_3/CD-O.py", "cdo")
    roster = json.load(open(kit / "data/track_3_evaluation_objects.json"))["episodes"]
    over = dict(s.split("=", 1) for s in a.set)
    pairs = {}
    for e, names in roster.items():
        for s, n in enumerate(names):
            pairs.setdefault(n, []).append((int(e), s))

    def packed(p):
        v, f = budget_mesh(str(p), faces=4096, vertices=4096)
        return np.asarray(v, float), np.asarray(f, np.int64)

    rows, allv = {}, []
    for n in sorted(pairs):
        if a.only and n not in a.only.split(","):
            continue
        slots = pairs[n]
        scan = find(SCANS / n, n)
        if scan is None:
            rows[n] = {"pairs": len(slots), "cd_o": a.spoon, "source": "constant (no scan)"}
            allv += [a.spoon] * len(slots)
            continue
        cand_p = Path(over[n]) if n in over else find(a.mesh_dir, n)
        if a.ref == "packed":
            ref = packed(scan)
        else:
            m = trimesh.load(scan, force="mesh", process=False)
            ref = (np.asarray(m.vertices, float), np.asarray(m.faces, np.int64))
        cand = packed(cand_p)
        vals = [cdo._object_chamfer_cm(ref, cand, cdo._object_seed(e, s)) for e, s in slots]
        rows[n] = {"pairs": len(slots), "cd_o": float(np.mean(vals)), "per_pair": [round(v, 4) for v in vals],
                   "source": str(cand_p)}
        allv += vals
    if a.only:
        for n, r in rows.items():
            print(f"{n:18s} {r['pairs']:5d} {r['cd_o']:8.3f}  {r['source']}")
        return
    total = float(np.mean(allv))
    no_spoon = float(np.mean([v for n, r in rows.items() if "per_pair" in r for v in r["per_pair"]]))
    print(f"{'object':18s} {'pairs':>5s} {'CD-O cm':>8s}  source")
    for n, r in rows.items():
        print(f"{n:18s} {r['pairs']:5d} {r['cd_o']:8.3f}  {r['source']}")
    print(f"TOTAL (mean over {len(allv)} pairs, spoon {a.spoon}): {total:.4f} cm   without spoon: {no_spoon:.4f} cm")
    if a.json:
        json.dump({"total": total, "total_no_spoon": no_spoon, "ref": a.ref, "objects": rows}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
