"""gmd_ft pilot summary -> outputs/gmd_ft/SUMMARY.md (run by ft_queue.sh at the end; safe to rerun any time)."""
import json
from pathlib import Path

FO = Path("/usr1/e2s/v2d_challenge_data/outputs/gmd_ft"); V4 = Path("/usr1/e2s/v2d_challenge_data/outputs/gmd/v4")
S0 = json.load(open(V4 / "summary_s0.json"))


def rows(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def one(p):
    return json.loads(p.read_text()) if p.exists() else {}


def best(r):
    return max(r, key=lambda x: x["add_auc"]) if r else {}


f = lambda x, k="add_auc": "-" if not x or k not in x else f"{x[k]:.3f}"  # noqa: E731
out = ["# gmd_ft pilots: bounded finger correction R on frozen G, then M resumed from v4",
       "",
       "All AUCs: 64-world ADD-AUC, evaluation composition (no assist, frame-0 reset, deterministic). R: offset on finger targets,",
       "|offset| <= delta per joint, only while engaged and touching; trained with M frozen and wrist perturbation.",
       "",
       "| run | K | S0 best | v4 best | parity (R = 0) | best R (at) | perturbed: with R / without | cap frac / offset (best R) | M resumed best (at) | resumed M, R off |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for run in ["B_ep6", "B_ep15", "A_ep3"]:
    ep = run.split("_ep")[1]
    v4 = best(rows(V4 / "m" / run / "evals.jsonl")); par = one(FO / "parity" / f"{run}.json")
    for k in (3, 1):
        rd = FO / f"r_{run}_k{k}"
        if not rd.exists() and k == 1:
            continue
        rb = best(rows(rd / "evals.jsonl")); md = FO / f"m_{run}_k{k}"; mb = best(rows(md / "evals.jsonl"))
        out.append(f"| {run} | {k} | {S0[ep]:.3f} | {f(v4)} ({v4.get('step', 0) / 1e6:.1f}M) | {f(par)} | {f(rb)} ({rb.get('step', 0) / 1e6:.1f}M) | "
                   f"{f(one(rd / 'perturbed_zero_r0.json'))} / {f(one(rd / 'perturbed_zero_r1.json'))} | "
                   f"{f(rb, 'cap_frac')} / {f(rb, 'offset_rel_mean')} | {f(mb)} ({mb.get('step', 0) / 1e6:.1f}M) | {f(one(md / 'best_r_off.json'))} |")
out += ["", "Warning signs: cap frac > 0.2 (R wants more than delta), resumed M collapsing with R off (M depends on R).",
        "Queue log: logs/queue.log; per-run config.txt, train.log, eval.log, evals.jsonl."]
(FO / "SUMMARY.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
