"""Figure: the 21 scored hand keypoints on the MANO right hand (rest pose, palm view), numbered in scoring order.

  ../.venv-bench/bin/python tools/draw_hand_joints.py [--out results/egofound3r/figures/hand_joints_21.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from handbench.mano import Mano  # noqa: E402

NAMES = ["wrist", "thumb CMC", "thumb MCP", "thumb IP", "thumb tip", "index MCP", "index PIP", "index DIP", "index tip",
         "middle MCP", "middle PIP", "middle DIP", "middle tip", "ring MCP", "ring PIP", "ring DIP", "ring tip",
         "little MCP", "little PIP", "little DIP", "little tip"]
COLOURS = {"thumb": "#d62728", "index": "#ff7f0e", "middle": "#2ca02c", "ring": "#1f77b4", "little": "#9467bd",
           "wrist": "#333333"}
TITLE = "Scoring handtracking using 21 keypoints (right hand, palm view, MANO model)"
LEGEND = "Base Knuckle (MCP/CMC), Middle Knuckle (PIP/IP), Last Knuckle (DIP), Fingertip (tip)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "egofound3r" / "figures" / "hand_joints_21.png"))
    a = ap.parse_args()
    m = Mano()
    V, J = m.forward(np.zeros(3), np.zeros(45), np.zeros(10), np.zeros(3), True, 1.0)
    V, J = V[0] * 1000, J[0] * 1000  # mm
    F = m.faces_for(True)
    uvV, dV = np.c_[-V[:, 2], V[:, 0]], V[:, 1]
    uvJ = np.c_[-J[:, 2], J[:, 0]]
    if uvJ[12, 1] < uvJ[0, 1]:  # fingers up
        uvV[:, 1] *= -1
        uvJ[:, 1] *= -1
    if uvJ[4, 0] > uvJ[20, 0]:  # thumb on the left
        uvV[:, 0] *= -1
        uvJ[:, 0] *= -1
    cols = [COLOURS[n.split()[0]] for n in NAMES]
    bones = [(0, 1), (1, 2), (2, 3), (3, 4)] + [(a_, b) for f in range(1, 5) for a_, b in
                                                ((0, 4 * f + 1), (4 * f + 1, 4 * f + 2), (4 * f + 2, 4 * f + 3),
                                                 (4 * f + 3, 4 * f + 4))]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.add_collection(PolyCollection(uvV[F][np.argsort(dV[F].mean(1))], facecolor="#f3d9c6", edgecolor="#e5c6b1",
                                     linewidth=0.25))
    for p, c in bones:
        ax.plot(uvJ[[p, c], 0], uvJ[[p, c], 1], "-", color=cols[c], lw=2.5, zorder=3, solid_capstyle="round")
    ax.scatter(uvJ[:, 0], uvJ[:, 1], s=70, c=cols, zorder=4, edgecolors="white", linewidths=1.3)
    xl, xr = uvV[:, 0].min() - 20, uvV[:, 0].max() + 20
    ytop, ybot = uvV[:, 1].max(), uvJ[0, 1] + 5
    left = [i for i in range(1, 21) if NAMES[i].split()[0] in ("thumb", "index")]
    right = [i for i in range(1, 21) if i not in left]
    for idx, x, ha in ((left, xl, "right"), (right, xr, "left")):
        idx = sorted(idx, key=lambda i: -uvJ[i, 1])
        for i, y in zip(idx, np.linspace(ytop, ybot, len(idx))):
            ax.annotate(f"{i}  {NAMES[i]}", uvJ[i], xytext=(x, y), fontsize=10, color=cols[i], ha=ha, va="center",
                        zorder=5, arrowprops=dict(arrowstyle="-", color=cols[i], lw=0.7, alpha=0.7))
    ax.annotate("0  wrist", uvJ[0], xytext=(uvJ[0, 0], uvJ[0, 1] - 16), fontsize=10, color=cols[0], ha="center",
                va="center")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(xl - 85, xr + 85)
    ax.set_ylim(uvV[:, 1].min() - 30, ytop + 10)
    ax.set_title(TITLE, fontsize=12)
    ax.text((xl + xr) / 2, uvV[:, 1].min() - 18, LEGEND, ha="center", va="top", fontsize=8.5, color="#555555")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(a.out)


if __name__ == "__main__":
    main()
