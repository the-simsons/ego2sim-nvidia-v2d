"""Summary of a GMD run: S0 vs M+G per episode (table + curves) -> outputs/gmd/<run>/SUMMARY.md, curves_all.png.
   summarize_full.py [s0only]   (s0only: just write summary_s0.json with each episode's S0 best 64-world AUC)
env (second run): GMD_ROOT (output folder), GMD_RUNS ("A:seen:41 B:unseen:37 ..."), GMD_GNAME (G file name), GMD_NOTE (extra
header line), GMD_SET (episode table), GMD_TEACH (G's teacher episodes, for the header)."""
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from episodes import S0  # noqa: E402

G = Path(os.environ.get("GMD_ROOT", "/usr1/e2s/v2d_challenge_data/outputs/gmd/full"))
GNAME = os.environ.get("GMD_GNAME", "g9")
T = Path("/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0")
RUNS = [("A", "seen", 41), ("A", "seen", 26), ("B", "unseen, S0 good", 38), ("B", "unseen, S0 good", 6), ("B", "unseen, S0 good", 37),
        ("C", "unseen, S0 failed", 28), ("C", "unseen, S0 failed", 19), ("C", "unseen, S0 failed", 5)]
if os.environ.get("GMD_RUNS"):
    RUNS = [(g, d.replace("_", " "), int(e)) for g, d, e in (x.split(":") for x in os.environ["GMD_RUNS"].split())]


def s0_curve(ep):
    pts = []
    phase = S0[ep][4] if len(S0[ep]) > 4 else "p1_96M"
    for f in glob.glob(str(T / S0[ep][0] / phase / "eval/eval_*_w64.log")):
        m = re.findall(r'"evaluation/add_auc": ([0-9.]+)', open(f).read())
        if m:
            pts.append((int(re.findall(r"eval_(\d+)_w64", f)[0]), float(m[-1])))
    return sorted(pts)


def first_cross(pts, thr=0.9):
    return next((s for s, a in pts if a >= thr), None)


def at(pts, step):
    """Score of the last checkpoint at or before `step`."""
    c = [a for s, a in pts if s <= step]
    return c[-1] if c else None


s0 = {ep: s0_curve(ep) for _, _, ep in RUNS}
json.dump({str(ep): max(a for _, a in p) for ep, p in s0.items() if p}, open(G / "summary_s0.json", "w"), indent=1)
if len(sys.argv) > 1:
    sys.exit()

gate = {}
for _, _, ep in RUNS:
    try:
        gate[ep] = json.load(open(G / f"ep{ep}/fingers_g_{GNAME}.json"))["add_auc"] / json.load(open(G / f"ep{ep}/nominal.json"))["add_auc"]
    except FileNotFoundError:
        gate[ep] = None
rows = ["| group | episode | S0 best | M+G best (at) | stopped | M+G @ 25M / S0 @ 25M | first >= 0.9: M+G / S0 | Gate 1 (teacher wrists + G) |",
        "|---|---|---|---|---|---|---|---|"]
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
nc = 4 if len(RUNS) <= 8 else 5
nr = max(2, -(-len(RUNS) // nc))
fig, axs = plt.subplots(nr, nc, figsize=(4.5 * nc, 3.75 * nr), sharey=True, squeeze=False)
for ax, (grp, desc, ep) in zip(axs.ravel(), RUNS):
    name = f"{grp}_ep{ep}"
    f = G / "m" / name / "evals.jsonl"
    m = sorted((r["step"], r["add_auc"]) for r in map(json.loads, open(f))) if f.exists() else []
    stop = (G / "m" / name / "STOP").read_text().split(" ", 1)[1].split(":")[0] if (G / "m" / name / "STOP").exists() else "cap"
    p0 = s0[ep]
    if m:
        bs, ba = max(m, key=lambda x: x[1])
        fm, fs = first_cross(m), first_cross(p0)
        g = f"{100 * gate[ep]:.0f}%" if gate.get(ep) else "-"
        a25, s25 = at(m, 25.2e6), at(p0, 25.2e6)
        rows.append(f"| {grp} ({desc}) | {ep} | {max(a for _, a in p0):.3f} | {ba:.3f} ({bs / 1e6:.1f}M) | {stop} at {m[-1][0] / 1e6:.1f}M | "
                    f"{a25 if a25 is None else round(a25, 3)} / {s25 if s25 is None else round(s25, 3)} | "
                    f"{'-' if fm is None else f'{fm / 1e6:.1f}M'} / {'-' if fs is None else f'{fs / 1e6:.1f}M'} | {g} |")
    ax.plot([s / 1e6 for s, _ in p0], [a for _, a in p0], "o-", c="gray", ms=3, label="S0")
    ax.plot([s / 1e6 for s, _ in m], [a for _, a in m], "o-", c="C3", ms=3, label=f"M + frozen {GNAME}")
    ax.axhline(0.9, ls=":", c="k", lw=0.8); ax.set_title(f"{grp}: ep{ep} ({desc})", fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)
for ax in axs.ravel()[len(RUNS):]:
    ax.axis("off")
axs[0, 0].legend(loc="lower right")
for ax in axs[:, 0]:
    ax.set_ylabel("64-world ADD-AUC")
for ax in axs[-1]:
    ax.set_xlabel("environment steps (M)")
fig.tight_layout(); fig.savefig(G / "curves_all.png", dpi=100)
out = [f"# GMD run ({G.name}): movement policy M with a frozen grasp policy {GNAME}", "",
       f"{GNAME}: behaviour-cloned from the S0 teachers of {os.environ.get('GMD_TEACH', '41, 40, 26, 18, 2, 35, 8, 3, 36')}.",
       os.environ.get("GMD_NOTE", ""),
       "M: FlashSAC, S0 recipe/curriculum/world count per episode; stop at AUC >= 0.9 for 3 checkpoints, or converged",
       f"({os.environ.get('GMD_STOP', 'within 0.05 for 5 checkpoints after 33.6M')}), cap 96.5M. All scores: 64-world ADD-AUC, FlashCHORD evaluation composition.",
       "Evaluation episodes are scored against the cleaned reconstruction (not the hidden mocap).", "", *rows, "",
       "Curves: curves_all.png. Videos (real time, one episode): videos/ep<N>_M_vs_S0.mp4.", ""]
(G / "SUMMARY.md").write_text("\n".join(out))
print("\n".join(out))
