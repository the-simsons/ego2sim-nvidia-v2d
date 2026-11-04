"""All gmd_ft experiments vs S0 and v4: 64-world ADD-AUC per checkpoint, one panel per episode -> outputs/gmd_ft/plots/curves_all.png"""
import glob, json, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = "/usr1/e2s/v2d_challenge_data/outputs"; FO = f"{O}/gmd_ft"; S0D = f"{O}/rl_train_req/training_s0/eval"
EPS = [("B_ep6", 6, "ep6 (pot + spoon, unseen)"), ("B_ep15", 15, "ep15 (dish rack + spoon, unseen)"), ("A_ep3", 3, "ep3 (pot + spoon, G teacher)")]
C = {"S0": "#2a78d6", "v4": "#6b6a64", "R": "#eda100", "Mres": "#e87ba4", "joint": "#eb6834", "jo": "#1baf7a"}


def jl(p, key="step"):
    try: r = [json.loads(l) for l in open(p)]
    except FileNotFoundError: return [], []
    r.sort(key=lambda d: d[key]); return [d[key] / 1e6 for d in r], [d["add_auc"] for d in r]


def s0(ep):
    pts = []
    for f in glob.glob(f"{S0D}/ep{ep}_v2recon_ray5decay/*/eval/metrics_*.json"):
        pts.append((int(re.findall(r"metrics_(\d+)", f)[0]) / 1e6, json.load(open(f))["metrics"]["add_auc"]))
    pts.sort(); return [p[0] for p in pts], [p[1] for p in pts]


fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)
for ax, (run, ep, title) in zip(axes, EPS):
    v4 = [json.loads(l) for l in open(f"{O}/gmd/v4/m/{run}/evals.jsonl")]
    vb = max(v4, key=lambda d: d["add_auc"])["step"] / 1e6
    ax.axvspan(0, 35, color="#9a998f", alpha=0.10, lw=0); ax.text(1, 0.03, "assist (VOC) on", fontsize=9, color="#6b6a64")
    ax.axvline(50, color="#9a998f", lw=1, ls=":"); ax.text(50.5, 0.03, "Exp 3 unfreeze", fontsize=9, color="#6b6a64")
    x, y = s0(ep); ax.plot(x, y, color=C["S0"], lw=2, label="S0 (rl_train_req/training_s0)")
    x, y = jl(f"{O}/gmd/v4/m/{run}/evals.jsonl"); ax.plot(x, y, color=C["v4"], lw=2, label="v4: M + frozen G")
    x, y = jl(f"{FO}/r_{run}_k3/evals.jsonl"); ax.plot([vb + s for s in x], y, color=C["R"], lw=1.6, ls="--", label="Exp 1: R on frozen v4 (from v4 best)")
    x, y = jl(f"{FO}/m_{run}_k3/evals.jsonl")
    if x: ax.plot(x, y, color=C["Mres"], lw=1.6, ls="--", label="Exp 1: M resumed with R")
    x, y = jl(f"{FO}/j_{run}/evals.jsonl"); ax.plot(x, y, color=C["joint"], lw=2, label="Exp 3: joint training")
    x, y = jl(f"{FO}/jo_{run}/evals.jsonl"); ax.plot(x, y, color=C["jo"], lw=2, label="Exp 4: joint setup, G frozen")
    h = {k: json.load(open(f"{FO}/harsh/{run}_harsh_zero_r{z}.json"))["add_auc"] for k, z in (("off", 1), ("on", 0))}
    ax.scatter([vb], [h["off"]], marker="x", s=70, color="#e34948", zorder=5, label="Exp 2: v4 best under harsh shaking (R off)")
    ax.set_title(title, fontsize=12, loc="left"); ax.set_xlabel("environment steps (M)")
    ax.grid(color="#e4e3dc", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_ylim(0, 1); ax.set_xlim(0, 112)
axes[0].set_ylabel("64-world ADD-AUC (no assist)")
hd, lb = axes[0].get_legend_handles_labels(); hd2, lb2 = axes[1].get_legend_handles_labels()
for a, b in zip(hd2, lb2):
    if b not in lb: hd.append(a); lb.append(b)
fig.legend(hd, lb, loc="lower center", ncol=4, frameon=False, fontsize=10)
fig.suptitle("All experiments vs S0 and v4 (Exp 1 R curve drawn from v4's best checkpoint; M frozen there)", x=0.01, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig(f"{FO}/plots/curves_all.png", dpi=130)
print("saved")
