"""Plots + summary for the VOC fix check (vocfix_check.sh): object position / orientation error over time for the old VOC
replay, the feedforward VOC replay and the S0 policy (16-world nominal rollout, no VOC), with the reference object speed."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = Path("/usr1/e2s/v2d_challenge_data/outputs/gmd/full/debug/vocfix")
EPS = {26: "ep26: dish rack + white pot (2 objects), public recon", 37: "ep37: wooden spoon (1 object), v2recon"}


def qang(a, b):
    """Angle between xyzw quaternion arrays."""
    d = np.abs(np.sum(a * b, -1)).clip(0, 1)
    return 2 * np.arccos(d)


fig, axs = plt.subplots(2, 2, figsize=(15, 8))
rows = []
for r, (ep, title) in enumerate(EPS.items()):
    voc = {m: json.load(open(O / f"ep{ep}_{m}.json")) for m in ("old", "ff")}
    meta = np.load(O / f"ro_ep{ep}/meta.npz", allow_pickle=True)["meta"].item()
    ro = np.load(O / f"ro_ep{ep}/nominal.npz")
    fps = meta["fps"]; bq = ro["body_q"]                                    # (T, W, bodies, 7) before each step
    ids = meta["object_body_ids"]; T = bq.shape[0]
    rp = meta["ref_obj_pos"]; rq = meta["ref_obj_quat_wxyz"][..., [1, 2, 3, 0]]
    f = np.maximum(np.arange(T) - 1, 0)                                     # state before step s = result of step s-1
    pe = np.linalg.norm(bq[:, :, ids, :3] - rp[f][:, None], axis=-1).max(-1)  # (T, W) worse object
    oe = qang(bq[:, :, ids, 3:], rq[f][:, None]).max(-1)
    t_pol = np.arange(T) / fps
    speed = np.r_[0, np.linalg.norm(np.diff(rp, axis=0), axis=-1).max(-1) * fps]
    for c, (key, scale, unit) in enumerate([("object_pos_m", 100, "cm"), ("object_ori_rad", np.degrees(1), "deg")]):
        ax = axs[r, c]
        for m, col, lab in (("old", "C3", "old VOC replay"), ("ff", "C2", "new VOC replay (feedforward)")):
            e = np.array([x[key] for x in voc[m]]) * scale
            ax.plot(np.arange(len(e)) / 50.0, e, c=col, lw=1.2, label=lab)
        pol = (pe if c == 0 else oe) * scale
        ax.plot(t_pol, pol.mean(1), c="C0", lw=1.2, label="S0 policy, no VOC (mean of 16)")
        ax.fill_between(t_pol, np.percentile(pol, 10, 1), np.percentile(pol, 90, 1), color="C0", alpha=0.15)
        ax2 = ax.twinx(); ax2.fill_between(np.arange(len(speed)) / fps, speed, color="gray", alpha=0.15, lw=0)
        ax2.set_ylabel("reference object speed (m/s)", color="gray"); ax2.tick_params(axis="y", colors="gray")
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.set_title(f"{title}: object {'position' if c == 0 else 'orientation'} error", fontsize=10)
        ax.set_xlabel("time (s)"); ax.set_ylabel(unit); ax.grid(alpha=0.3)
        if r == 0 and c == 0:
            ax.legend(loc="upper left", fontsize=8)
    row = {"ep": ep}
    for m in ("old", "ff"):
        e = np.array([x["object_pos_m"] for x in voc[m]]) * 100; a = np.degrees([x["object_ori_rad"] for x in voc[m]])
        row[m] = (e.mean(), np.percentile(e, 95), a.mean())
    row["policy"] = (100 * pe.mean(), 100 * np.percentile(pe, 95), np.degrees(oe.mean()))
    row["auc"] = json.load(open(O / f"ro_ep{ep}/nominal.json"))["add_auc"]
    rows.append(row)
fig.tight_layout(); fig.savefig(O / "voc_fix_errors.png", dpi=100)
print("| episode | old VOC replay | new VOC replay | S0 policy (no VOC, 16 worlds) |\n|---|---|---|---|")
for r in rows:
    fmt = lambda v: f"{v[0]:.2f} cm mean / {v[1]:.2f} cm p95 / {v[2]:.1f} deg"  # noqa: E731
    print(f"| {r['ep']} | {fmt(r['old'])} | {fmt(r['ff'])} | {fmt(r['policy'])} (AUC {r['auc']:.3f}) |")
