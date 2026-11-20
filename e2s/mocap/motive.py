"""OptiTrack Motive CSV export loader (format version 1.2x, rotation type quaternion).

Header line 1 holds key, value pairs (Capture Frame Rate, Rotation Type, Length Units, ...). Lines 3-7 name each
column: kind (Rigid Body, Rigid Body Marker, Marker), asset name, id, Rotation/Position, X/Y/Z/W. Data from line 8.
The up axis is not in the file: it is an export setting (Motive defaults to Y up; our exports are Z up), so it is
given by the clip manifest and checked against the data by `check_up_axis`.
"""
import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}


def load(path):
    """-> {"meta": {...}, "fps": float, "frame": (N,), "time": (N,) s, "bodies": {name: {"quat_xyzw": (N,4), "pos": (N,3)}},
           "markers": {name: (N,3)}}. Missing samples are NaN."""
    with open(path, errors="replace") as f:
        lines = [next(f) for _ in range(7)]
    head = lines[0].strip().split(",")
    meta = dict(zip(head[0::2], head[1::2]))
    if meta.get("Rotation Type", "Quaternion") != "Quaternion":
        raise ValueError(f"{path}: rotation type {meta.get('Rotation Type')!r}; export quaternions")
    scale = {"Meters": 1.0, "Centimeters": 0.01, "Millimeters": 0.001}.get(meta.get("Length Units", "Meters"))
    if scale is None:
        raise ValueError(f"{path}: unknown length units {meta.get('Length Units')!r}")
    kinds = lines[2].rstrip("\n").split(",")
    names = lines[3].rstrip("\n").split(",")
    quantity = lines[5].rstrip("\n").split(",")
    comps = lines[6].rstrip("\n").split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=7)
    if data.ndim == 1:
        data = data[None]
    fps = float(meta.get("Export Frame Rate") or meta.get("Capture Frame Rate"))
    out = {"meta": meta, "fps": fps, "frame": data[:, 0].astype(int), "time": data[:, 0] / fps, "bodies": {}, "markers": {}}
    cols = {}
    for c in range(2, len(names)):
        if c >= data.shape[1]:
            break
        cols.setdefault((kinds[c], names[c], quantity[c]), {})[comps[c]] = c
    for (kind, name, qty), cc in cols.items():
        if kind == "Rigid Body" and qty == "Rotation":
            out["bodies"].setdefault(name, {})["quat_xyzw"] = data[:, [cc["X"], cc["Y"], cc["Z"], cc["W"]]]
        elif kind == "Rigid Body" and qty == "Position":
            out["bodies"].setdefault(name, {})["pos"] = data[:, [cc["X"], cc["Y"], cc["Z"]]] * scale
        elif kind in ("Rigid Body Marker", "Marker") and qty == "Position":
            key = name if kind == "Rigid Body Marker" else f"marker:{name}"
            if key not in out["markers"]:
                out["markers"][key] = data[:, [cc["X"], cc["Y"], cc["Z"]]] * scale
    for name, b in out["bodies"].items():
        if "quat_xyzw" not in b or "pos" not in b:
            raise ValueError(f"{path}: rigid body {name!r} lacks rotation or position columns")
    return out


def check_up_axis(positions, up):
    """The vertical axis should vary least while an object rests on a table: return (ok, message)."""
    p = positions[~np.isnan(positions).any(1)]
    spread = np.ptp(p[: max(10, len(p) // 10)], axis=0)
    up_i = AXES[up]
    heights = p[:, up_i]
    ok = heights.min() > 0.2 and heights.max() < 2.5
    return ok, f"{up}-axis height range {heights.min():.3f}..{heights.max():.3f} m (expect a table-top object, 0.2-2.5 m)"
