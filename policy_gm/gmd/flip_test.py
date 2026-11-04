"""Engage-flip test on the teachers' labelled data: G's mean finger action (+ = closing) with own-engage forced on vs off, on
frames with the hand near (< 10 cm) or touching the object; and the mean change. A causal engage bit gives a large gap.
  flip_test.py <g.npz> [<g.npz> ...]     (host numpy; data: outputs/gmd/v2/ep*/nominal_data.npz)"""
import sys

import numpy as np

sys.path.insert(0, "/usr1/e2s/retarget_env/gmd")
from gfeat import blocks  # noqa: E402

V = "/usr1/e2s/v2d_challenge_data/outputs/gmd/v2"
EPS = [35, 36, 41, 40, 18, 2, 3, 26, 8]
BL = blocks(np.load(f"{V}/ep18/meta.npz", allow_pickle=True)["meta"].item())
e_own = BL["flags"][0] + 2; v0 = BL["vec_t"][0]; c0, c1 = BL["ca_t"][0], BL["ca_o"][1]


def G(p, x):
    n = int(p["n_hidden"]); h = (x - p["mu"]) / p["sd"]
    for i in range(n):
        h = h @ p[f"W{i}"] + p[f"b{i}"]; h = (h - h.mean(-1, keepdims=True)) / np.sqrt(h.var(-1, keepdims=True) + 1e-6) * p[f"g{i}"] + p[f"be{i}"]
        h = h / (1 + np.exp(-h))
    return np.tanh(h @ p[f"W{n}"] + p[f"b{n}"])


rng = np.random.default_rng(0)
xs = []
for e in EPS:
    D = np.load(f"{V}/ep{e}/nominal_data.npz"); X = D["X"][:, D["keep"]].reshape(-1, D["X"].shape[-1]).astype(np.float32)
    near = (np.linalg.norm(X[:, v0:v0 + 3], axis=1) < 0.10) | (X[:, c0:c1 - 1].max(1) > 0)
    x = X[near]; xs.append(x[rng.choice(len(x), min(len(x), 4000), replace=False)])
x = np.concatenate(xs)
for path in sys.argv[1:]:
    p = dict(np.load(path))
    on = x.copy(); on[:, e_own] = 1; off = x.copy(); off[:, e_own] = 0
    g1, g0 = G(p, on), G(p, off)
    print(f"{path.split('/')[-1]}: near/touching frames {len(x)} | mean finger action engage on {g1.mean():+.3f}, off {g0.mean():+.3f} | "
          f"mean |on - off| {np.abs(g1 - g0).mean():.3f} (|G| {np.abs(g1).mean():.3f})")
