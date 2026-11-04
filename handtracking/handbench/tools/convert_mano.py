"""Convert a licensed MANO_RIGHT.pkl (needs chumpy to unpickle) into the plain .npz handbench.mano reads.

Usage: python convert_mano.py MANO_RIGHT.pkl [OUT.npz]   (default OUT: handtracking/assets/mano_right.npz, gitignored)
"""
import pickle
import sys
from pathlib import Path

import numpy as np


def arr(x):
    x = x.r if hasattr(x, "r") else x
    x = x.toarray() if hasattr(x, "toarray") else x
    return np.asarray(x, dtype=np.float64)


def main():
    src = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "assets/mano_right.npz"
    d = pickle.load(open(src, "rb"), encoding="latin1")
    keep = {k: arr(d[k]) for k in ["v_template", "shapedirs", "posedirs", "J_regressor", "weights", "hands_mean",
                                   "hands_components"]}
    keep["faces"] = np.asarray(d["f"], np.int64)
    keep["kintree_table"] = np.asarray(d["kintree_table"], np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **keep)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
