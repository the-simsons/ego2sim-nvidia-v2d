"""Freeze the mesh set for the full run: copy / rescale each object's chosen mesh into one folder, with a manifest.

  python build_mesh_final.py OUT_DIR
Each entry: source folder, optional target longest box edge (rescale about the body origin, CD-O unchanged). Writes
OUT_DIR/<name>.{obj,glb} and OUT_DIR/manifest.json (source, scale, box edges, watertight, sha256).
"""
import hashlib
import json
import sys
from pathlib import Path

import trimesh

E2 = Path.home() / "v2d/ego2sim"
# name: (source folder, source name, target longest box edge in m or None, why)
SET = {
    "beige_cup": ("meshes", "beige_cup", None, "harness 2026-10-16"),
    "blue_cup": ("meshes", "blue_cup", None, "harness ep12"),
    "mini_sweeper": ("meshes", "mini_sweeper", None, "harness 2026-10-16"),
    "white_pot": ("meshes", "white_pot", None, "harness 2026-10-16"),
    "white_pot_lid": ("meshes", "white_pot_lid", None, "harness ep2"),
    "wooden_spoon": ("meshes", "wooden_spoon", None, "harness eval ep35 (no public scan)"),
    "mini_dust_pan": ("meshes_fs", "mini_dust_pan", 0.178, "harness FoundationStereo run, size: FS hand-free p90"),
    "water_pitcher": ("meshes_fs", "water_pitcher", 0.380, "harness FS run, rescaled to 38.0 cm (FS hand-free p90); "
                      "v2 broke FP tracking (ep21 AUC 0.000 -> 0.784, NOTES Phase 8c)"),
    "plastic_dish_rack": ("meshes_fusion", "plastic_dish_rack", None, "depth fusion ep13 (FS depth + VO, all frames)"),
    "wooden_piece_1": ("meshes", "wooden_piece_1", 0.229, "harness ep23, size: FS hand-free p90"),
    "wooden_piece_2": ("meshes", "wooden_piece_1", 0.229, "same mesh as wooden_piece_1 (near-identical scans)"),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    man = {}
    for name, (folder, src, target, why) in SET.items():
        entry = {"source": f"ego2sim/{folder}/{src}", "why": why}
        for ext in ("obj", "glb"):
            m = trimesh.load(E2 / folder / f"{src}.{ext}", force="mesh")
            s = 1.0 if target is None else target / m.extents.max()
            if s != 1.0:
                m.apply_scale(s)
            dst = out / f"{name}.{ext}"
            m.export(dst)
            if ext == "obj":
                entry.update(scale=round(s, 5), box_cm=[round(x * 100, 2) for x in m.extents],
                             watertight=bool(m.is_watertight), faces=len(m.faces))
            entry[f"sha256_{ext}"] = sha(dst)
        man[name] = entry
        print(f"{name:18s} x{entry['scale']:.4f} box cm {entry['box_cm']} watertight {entry['watertight']}  <- {entry['source']}")
    json.dump(man, open(out / "manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
