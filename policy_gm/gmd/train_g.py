"""Behaviour cloning of the grasp policy G from labelled teacher rollouts (JAX + optax).

  train_g.py <name> --train ep40[:settings] [ep38 ...] [--extra <dir>/<file>_data.npz ...] [--steps 30000]
Data: <ep>/<setting>_data.npz from label.py, kept worlds only; worlds >= 56 of every file are validation.
Model: MLP 3 x 512, LayerNorm, SiLU, tanh output (22 finger actions). Inputs normalised with training-set statistics (frozen).
Batches: phase-balanced (acquire 40 / release 15 / hold 35 / approach+idle 10 %, renormalised over phases present);
loss: MSE, x2 on acquire and release frames.
Writes <root>/g/<name>.npz (weights + normalisation) and <name>_report.json (per-phase validation MSE per training episode,
all-worlds MSE per --heldout episode). BPS shape code dropped (set to the dataset mean) on --bps_dropout of samples.
Engage made causal (defaults off): --neg_frac of each batch are counterfactual copies of frames with the hand within
--neg_dist of the object or touching it and engage on, with own-engage set to 0 and the label "open toward the approach
shape": clip((q_open - q) / 0.10, -1, 1), q = finger angles, q_open = that file's median finger pose in approach frames
(per hand). --contact_dropout of samples get their contact inputs (per-link contacts, force, other-object contact,
other hand's contact) set to the dataset mean, so the grasp decision cannot come from contact alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path("/out/full")
SETTINGS = ["nominal", "perturb", "dart0.1", "dart0.2", "perturb_dart0.1", "perturb_dart0.2"]
PHASE_P = {0: 0.05, 1: 0.05, 2: 0.40, 3: 0.35, 4: 0.15}
PHASES = ["idle", "approach", "acquire", "hold", "release"]


def load(files, with_ids=False):
    X, Y, P, V, F, Hh = [], [], [], [], [], []
    for i, f in enumerate(files):
        d = np.load(f)
        keep = d["keep"]; W = keep.shape[0]; Xa, Ya, Pa = d["X"], d["Y"], d["phase"]   # each key decompresses once
        val = np.arange(W) >= 56
        nh = Pa.shape[2]
        for w in np.flatnonzero(keep):
            X.append(Xa[:, w].reshape(-1, Xa.shape[-1])); Y.append(Ya[:, w].reshape(-1, Ya.shape[-1]))
            P.append(Pa[:, w].reshape(-1)); V.append(np.full(Pa[:, w].size, val[w]))
            F.append(np.full(Pa[:, w].size, i)); Hh.append(np.tile(np.arange(nh), Pa.shape[0]))
    out = (np.concatenate(X), np.concatenate(Y), np.concatenate(P), np.concatenate(V))
    return out + (np.concatenate(F), np.concatenate(Hh)) if with_ids else out


def files_for(spec):
    ep, _, s = spec.partition(":")
    return [ROOT / ep / f"{x}_data.npz" for x in (s.split(",") if s else SETTINGS)]


def init(key, sizes):
    p = {}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        p[f"W{i}"] = jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a) * (0.01 if i == len(sizes) - 2 else 1.0)
        p[f"b{i}"] = jnp.zeros(b)
        if i < len(sizes) - 2:
            p[f"g{i}"] = jnp.ones(b); p[f"be{i}"] = jnp.zeros(b)
    return p


def forward(p, x, n):
    h = x
    for i in range(n):
        h = h @ p[f"W{i}"] + p[f"b{i}"]
        mu, var = h.mean(-1, keepdims=True), h.var(-1, keepdims=True)
        h = jax.nn.silu((h - mu) / jnp.sqrt(var + 1e-6) * p[f"g{i}"] + p[f"be{i}"])
    return jnp.tanh(h @ p[f"W{n}"] + p[f"b{n}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--train", nargs="+", required=True); ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--width", type=int, default=512); ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout", nargs="*", default=[]); ap.add_argument("--bps_dropout", type=float, default=0.3)
    ap.add_argument("--root", default="/out/full")
    ap.add_argument("--neg_frac", type=float, default=0.0); ap.add_argument("--neg_dist", type=float, default=0.10)
    ap.add_argument("--contact_dropout", type=float, default=0.0)
    ap.add_argument("--off_net", action="store_true")
    ap.add_argument("--boost", nargs="*", default=[]); ap.add_argument("--boost_k", type=float, default=1.0)
    args = ap.parse_args()
    global ROOT
    ROOT = Path(args.root)
    import sys; sys.path.insert(0, "/env/gmd")
    from gfeat import blocks
    meta = np.load(ROOT / args.train[0].partition(":")[0] / "meta.npz", allow_pickle=True)["meta"].item()
    b0, b1 = blocks(meta)["bps"]
    files = [f for s in args.train for f in files_for(s)] + [ROOT / e for e in args.extra]
    file_ep = [Path(f).parent.name for f in files]                      # episode of each file (for --boost)
    X, Y, P, V, FID, HID = load(files, with_ids=True)
    tr = ~V
    BLK = blocks(meta); c0, c1 = BLK["ca_t"][0], BLK["ca_o"][1]; f0 = BLK["flags"][0]
    e_own = f0 + 2; q0, q1 = BLK["fq"]; v0 = BLK["vec_t"][0]
    if args.off_net:
        # Engage-off half of a gated G: the teachers' own engage-off frames (idle / approach / release, label = teacher action)
        # plus counterfactual copies of engage-on frames with engage set to 0: not touching -> the teacher's own action
        # (pre-shaping), touching -> a gentle open toward the approach shape (0.5 x clip((q_open - q) / 0.10)).
        dist = np.linalg.norm(X[:, v0:v0 + 3], axis=1); touching = X[:, c0:c1 - 1].max(1) > 0
        base = X[:, e_own] < 0.5; cf = ~base
        q_open = {}
        for k in np.unique(FID):
            for hh in np.unique(HID):
                m = (FID == k) & (HID == hh) & (P == 1)
                q_open[(k, hh)] = np.median(X[m, q0:q1], 0) if m.any() else np.median(X[(FID == k) & (HID == hh), q0:q1], 0)
        Xc = X[cf].copy(); Xc[:, e_own] = 0.0; Yc = Y[cf].copy()
        t = touching[cf]
        qo = np.stack([q_open[(k, hh)] for k, hh in zip(FID[cf][t], HID[cf][t])])
        Yc[t] = 0.5 * np.clip((qo - X[cf][t, q0:q1]) / 0.10, -1, 1)
        print(f"[{args.name}] off-net: {base.sum()} engage-off frames + {cf.sum()} counterfactuals ({t.sum()} touching -> gentle open)", flush=True)
        X = np.concatenate([X[base], Xc]); Y = np.concatenate([Y[base], Yc]).astype(np.float32)
        P = np.concatenate([P[base], P[cf]]); V = np.concatenate([V[base], V[cf]])
        FID = np.concatenate([FID[base], FID[cf]]); HID = np.concatenate([HID[base], HID[cf]]); tr = ~V
    neg_X = neg_Y = None
    if args.neg_frac > 0:
        dist = np.linalg.norm(X[:, v0:v0 + 3], axis=1); touching = X[:, c0:c1 - 1].max(1) > 0
        pool = np.flatnonzero(tr & (X[:, e_own] > 0.5) & ((dist < args.neg_dist) | touching))
        q_open = {}
        for k in np.unique(FID):
            for hh in np.unique(HID):
                m = (FID == k) & (HID == hh) & (P == 1)
                q_open[(k, hh)] = np.median(X[m, q0:q1], 0) if m.any() else np.median(X[(FID == k) & (HID == hh), q0:q1], 0)
        qo = np.stack([q_open[(k, hh)] for k, hh in zip(FID[pool], HID[pool])])
        neg_X = X[pool].copy(); neg_X[:, e_own] = 0.0
        neg_Y = np.clip((qo - X[pool, q0:q1]) / 0.10, -1, 1).astype(np.float32)
        print(f"[{args.name}] negatives: {len(pool)} counterfactual frames (engage off, near / touching, label = open "
              f"toward the approach shape; mean |label| {np.abs(neg_Y).mean():.3f})", flush=True)
    mu = X[tr].mean(0); sd = np.maximum(X[tr].std(0), 1e-3)
    Xn = ((X - mu) / sd).astype(np.float32)
    if neg_X is not None:
        neg_Xn = ((neg_X - mu) / sd).astype(np.float32)
    present = [k for k in PHASE_P if (P[tr] == k).any()]
    z = sum(PHASE_P[k] for k in present)
    idx = {k: np.flatnonzero(tr & (P == k)) for k in present}
    boost = np.isin(np.asarray(file_ep)[FID], args.boost) if args.boost else np.zeros(len(P), bool)
    pw = {k: np.where(boost[idx[k]], args.boost_k, 1.0) for k in present}
    pw = {k: v / v.sum() for k, v in pw.items()}
    if args.boost:
        print(f"[{args.name}] oversampling {args.boost} x{args.boost_k}: their share of training frames {boost[tr].mean():.2f}", flush=True)
    print(f"[{args.name}] train {tr.sum()} val {V.sum()} dim {X.shape[1]}; phase counts " +
          " ".join(f"{PHASES[k]}={len(idx[k])}" for k in present), flush=True)
    n = args.depth
    p = init(jax.random.PRNGKey(args.seed), [X.shape[1]] + [args.width] * n + [Y.shape[1]])
    sched = optax.cosine_decay_schedule(args.lr, args.steps, alpha=0.05)
    opt = optax.adamw(sched, weight_decay=1e-4); st = opt.init(p)

    @jax.jit
    def step(p, st, x, y, w, drop, cdrop):
        x = x.at[:, b0:b1].multiply(1.0 - drop[:, None])        # BPS dropout (normalised space: 0 = dataset mean)
        x = x.at[:, c0:c1].multiply(1.0 - cdrop[:, None]).at[:, f0].multiply(1.0 - cdrop)   # contact dropout
        def loss(p):
            return (w[:, None] * (forward(p, x, n) - y) ** 2).mean()
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(args.seed)
    Xj, Yj = jnp.asarray(Xn), jnp.asarray(Y)
    n_neg = int(round(args.batch * args.neg_frac)) if neg_X is not None else 0
    counts = rng.multinomial(args.batch - n_neg, [PHASE_P[k] / z for k in present])
    if n_neg:
        NXj, NYj = jnp.asarray(neg_Xn), jnp.asarray(neg_Y)
    for it in range(args.steps):
        b = np.concatenate([rng.choice(idx[k], c, p=pw[k]) for k, c in zip(present, counts)])
        w = np.where(np.isin(P[b], [2, 4]), 2.0, 1.0).astype(np.float32)
        xb, yb = Xj[b], Yj[b]
        if n_neg:
            nb = rng.integers(0, len(neg_Y), n_neg)
            xb = jnp.concatenate([xb, NXj[nb]]); yb = jnp.concatenate([yb, NYj[nb]]); w = np.concatenate([w, np.ones(n_neg, np.float32)])
        drop = (rng.random(len(w)) < args.bps_dropout).astype(np.float32)
        cdrop = (rng.random(len(w)) < args.contact_dropout).astype(np.float32)
        p, st, l = step(p, st, xb, yb, jnp.asarray(w), jnp.asarray(drop), jnp.asarray(cdrop))
        if it % 5000 == 0 or it == args.steps - 1:
            print(f"[{args.name}] step {it} loss {float(l):.4f}", flush=True)

    fwd = jax.jit(lambda x: forward(p, x, n))
    report = {"name": args.name, "train": args.train, "extra": args.extra, "neg_frac": args.neg_frac,
              "contact_dropout": args.contact_dropout}
    for ep in [s.partition(":")[0] for s in args.train] + args.heldout:
        fs = files_for(ep)
        if not all(f.exists() for f in fs):
            continue
        Xe, Ye, Pe, Ve = load(fs)
        held_in = any(s.partition(":")[0] == ep for s in args.train)
        m = Ve if held_in else np.ones_like(Ve)                # unseen episode: all worlds are test data
        pred = np.asarray(fwd(jnp.asarray(((Xe[m] - mu) / sd).astype(np.float32))))
        err = ((pred - Ye[m]) ** 2).mean(1); base = (Ye[m] ** 2).mean(1)
        report[ep] = {"split": "validation worlds" if held_in else "unseen episode (all worlds)"}
        for k in range(5):
            s = Pe[m] == k
            if s.any():
                report[ep][PHASES[k]] = {"mse": round(float(err[s].mean()), 4), "zero_action_mse": round(float(base[s].mean()), 4),
                                         "frames": int(s.sum())}
        print(f"[{args.name}] {ep} ({report[ep]['split']}): " + "  ".join(
            f"{k} {v['mse']:.4f} (zero {v['zero_action_mse']:.4f})" for k, v in report[ep].items() if isinstance(v, dict)), flush=True)
    out = ROOT / "g"; out.mkdir(exist_ok=True)
    np.savez(out / f"{args.name}.npz", mu=mu, sd=sd, n_hidden=n, **{k: np.asarray(v) for k, v in p.items()})
    json.dump(report, open(out / f"{args.name}_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
