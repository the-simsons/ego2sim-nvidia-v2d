"""Label teacher rollouts (per hand, per step) and build G's training set.   label.py <episode_dir> [settings...]

Per hand h: wrench share s_h = the S0 required-wrench support computed from hand h's contacts only (combined = both hands,
identical to the training reward); with two objects, per frame the object with the most support. Holds by hysteresis on the 3-frame median of s_h:
  start: combined > 0.5 and s_h > 0.2 for >= 4 frames;  end: combined < 0.3 or s_h < 0.1 for >= 2 frames;
  gaps < 3 frames merged, holds < 6 frames dropped.
contact onset = start of the contiguous run (contact or hand-object distance < 1 cm) that leads into the hold.
engage = [onset - 6, hold end).  phase: 0 idle, 1 approach (< 15 cm, before acquire), 2 acquire [onset - 6, start + 10),
3 hold, 4 release (end until no contact and > 2 cm, at most 10 frames).
World kept if the object never passes 25 cm / 1.5 rad and its mean position error is < 5 cm.
Writes <setting>_data.npz (X, Y, phase, engage, keep, world, t, hand) and timeline_<setting>.png.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).parent))
from gfeat import Featurizer, blocks, quat_xyzw_to_mat  # noqa: E402


warnings.filterwarnings("ignore", category=RuntimeWarning)


def wrench_shares(d, m):
    """Per-hand share (T, W, H) and combined support (T, W), as the S0 required-wrench term, taking per frame the object
    with the most support (scenes with one or two objects)."""
    T, W, S = d["contact_active"].shape
    nd = int(m["n_dirs"]); mu = m["wrench_friction"]; rc_all = np.asarray(m["obj_rc"], np.float64)
    B = len(m["object_body_ids"]); L = int(m["link_counts"][0]); Hn = len(m["sides"])
    act = m["wrench_active"].reshape(-1, B)
    dirs = m["wrench_dirs"].reshape(-1, B, nd, 6)
    ts = np.clip(d["timestep"], 0, len(act) - 1)
    share = np.zeros((T, W, Hn), np.float32); comb = np.zeros((T, W), np.float32)
    ang = np.arange(8) * np.pi / 4
    for b in range(B):
        per_h = []
        for h in range(Hn):
            sl = slice(int(m["hand_slot_starts"][h]) + b * L, int(m["hand_slot_starts"][h]) + (b + 1) * L)
            ca = d["contact_active"][..., sl].astype(bool)
            pc = d["contact_pos_o"][..., sl, :].astype(np.float32); nc = d["contact_dir_o"][..., sl, :].astype(np.float32)
            nc = nc / np.maximum(np.linalg.norm(nc, axis=-1, keepdims=True), 1e-9)
            t1 = np.cross(nc, np.array([1.0, 0, 0], np.float32)); small = np.linalg.norm(t1, axis=-1) < 0.1
            t1[small] = np.cross(nc[small], np.array([0, 1.0, 0], np.float32)); t1 /= np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-9)
            t2 = np.cross(nc, t1)
            F = nc[..., None, :] + mu * (np.cos(ang)[:, None] * t1[..., None, :] + np.sin(ang)[:, None] * t2[..., None, :])
            F = np.concatenate([F / np.linalg.norm(F, axis=-1, keepdims=True), nc[..., None, :]], -2)   # (T,W,L,9,3)
            Tq = np.cross(pc[..., None, :], F) / rc_all[b]
            D = dirs[ts, b].astype(np.float32)                                                       # (T,W,nd,6)
            val = np.einsum("twdk,twsek->twsde", D[..., :3], F) + np.einsum("twdk,twsek->twsde", D[..., 3:], Tq)
            val = np.nan_to_num(val, nan=0.0)               # zero force direction -> no support (Warp's normalize(0) = 0)
            per_h.append(np.maximum(np.where(ca[..., None, None], val, 0.0).max(-1).max(2), 0.0))  # (T,W,nd)
        on = act[ts, b] > 0.5
        for h in range(Hn):
            share[..., h] = np.maximum(share[..., h], np.clip(per_h[h].min(-1) / 0.5, 0, 1) * on)
        comb = np.maximum(comb, np.clip(np.maximum.reduce(per_h).min(-1) / 0.5, 0, 1) * on)
    return share, comb


def runs(mask):
    """[start, end) of True runs in a 1-D bool array."""
    x = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(x == 1), np.flatnonzero(x == -1)))


def holds_for(s_h, comb):
    T = len(s_h); out = []; i = 0
    start_ok = (comb > 0.5) & (s_h > 0.2); end_ok = (comb < 0.3) | (s_h < 0.1)
    while i < T:
        if start_ok[i:i + 4].all() and i + 4 <= T:
            j = i + 4
            while j < T and not end_ok[j:j + 2].all():
                j += 1
            out.append([i, j]); i = j + 2
        else:
            i += 1
    merged = []
    for a, b in out:
        if merged and a - merged[-1][1] < 3:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= 6]


def label_world(s_h, comb, contact, dist):
    """Per hand series (T,) -> engage (T,), phase (T,), holds."""
    T = len(s_h)
    eng = np.zeros(T, np.float32); ph = np.zeros(T, np.int8)
    near = contact | (dist < 0.01)
    hl = holds_for(s_h, comb)
    for a, b in hl:
        on = a
        while on > 0 and near[on - 1]:
            on -= 1
        ph[max(on - 6, 0):min(a + 10, T)] = 2
        ph[a:b][ph[a:b] == 0] = 3
        ph[min(a + 10, b):b] = 3
        r = b
        while r < min(b + 10, T) and (contact[r] or dist[r] < 0.02):
            r += 1
        ph[b:max(r, min(b + 1, T))] = 4
        eng[max(on - 6, 0):b] = 1
        pre = np.arange(0, max(on - 6, 0))
        appr = pre[(dist[pre] < 0.15) & (ph[pre] == 0)]
        ph[appr] = 1
    return eng, ph, hl


def main():
    ep_dir = Path(sys.argv[1])
    settings = sys.argv[2:] or ["nominal", "perturb", "dart0.1", "dart0.2", "perturb_dart0.1", "perturb_dart0.2"]
    m = np.load(ep_dir / "meta.npz", allow_pickle=True)["meta"].item()
    sides = m["sides"]; Hn = len(sides)
    starts, nf = m["hand_action_starts"], m["finger_counts"]
    wrist_idx = np.stack([np.arange(s, s + 6) for s in starts]); fing_idx = np.stack([np.arange(s + 6, s + 6 + n) for s, n in zip(starts, nf)])
    summary = []
    for setting in settings:
        d = dict(np.load(ep_dir / f"{setting}.npz"))
        T, W = d["timestep"].shape
        share, comb = wrench_shares(d, m)
        share = median_filter(share, size=(3, 1, 1), mode="nearest"); comb = median_filter(comb, size=(3, 1), mode="nearest")
        # object error vs reference -> world filter
        ts = np.clip(d["timestep"], 0, m["num_frames"] - 1); pe = 0; rot_err = 0
        for j, ob in enumerate(int(b) for b in m["object_body_ids"]):
            po = d["body_q"][:, :, ob, :3].astype(np.float64); ref = m["ref_obj_pos"][ts, j]
            qa = d["body_q"][:, :, ob, 3:].astype(np.float64); qr = m["ref_obj_quat_wxyz"][ts, j][..., [1, 2, 3, 0]]
            Ra, Rr = quat_xyzw_to_mat(qa), quat_xyzw_to_mat(qr)
            rot_err = np.maximum(rot_err, np.arccos(np.clip((np.einsum("twij,twij->tw", Ra, Rr) - 1) / 2, -1, 1)))
            pe = np.maximum(pe, np.linalg.norm(po - ref, axis=-1))
        keep = ~((pe > 0.25) | (rot_err > 1.5)).any(0) & (pe.mean(0) < 0.05)
        # features (engage from labels, so label first: distances need features -> compute features with engage=0, then patch)
        feat = Featurizer(m); feat.reset()
        hist = np.zeros((3, W, Hn, 6), np.float32); X = []
        raw = np.clip(d["raw"], -1, 1); exe = d["exe"]
        for t in range(T):
            hist[2], hist[1] = hist[1], hist[0]; hist[0] = raw[t][:, wrist_idx]
            prev = exe[t - 1][:, fing_idx] if t > 0 else np.zeros((W, Hn, int(nf[0])), np.float32)
            X.append(feat(d["body_q"][t], d["joint_q"][t], d["joint_qd"][t], d["contact_active"][t].astype(np.float64),
                          d["contact_force_w"][t].astype(np.float64), hist, prev, np.zeros((W, Hn))))
        X = np.stack(X)                                                          # (T,W,H,D)
        b0, b1 = blocks(m)["vec_t"]
        dist = np.linalg.norm(X[..., b0:b1].reshape(T, W, Hn, 11, 3), axis=-1).min(-1)
        contact = np.zeros((T, W, Hn), bool)
        for h in range(Hn):
            s0, L = int(m["hand_slot_starts"][h]), int(m["link_counts"][h])
            contact[..., h] = d["contact_active"][..., s0:s0 + L * len(m["object_body_ids"])].any(-1)
        eng = np.zeros((T, W, Hn), np.float32); ph = np.zeros((T, W, Hn), np.int8); nholds = np.zeros((W, Hn), int)
        two_hand = 0; hold_frames = 0
        for w in range(W):
            for h in range(Hn):
                eng[:, w, h], ph[:, w, h], hl = label_world(share[:, w, h], comb[:, w], contact[:, w, h], dist[:, w, h])
                nholds[w, h] = len(hl)
            both = (ph[:, w, 0] == 3) & (ph[:, w, 1] == 3); two_hand += both.sum(); hold_frames += ((ph[:, w] == 3).any(-1)).sum()
        X[..., -3] = eng[..., ::-1]                                              # other hand engaged
        X[..., -2] = eng                                                         # own engage
        Y = raw[:, :, fing_idx]                                                  # (T,W,H,22) clean teacher finger actions
        tw = d.get("teacher_worlds", np.zeros(W, bool))
        np.savez_compressed(ep_dir / f"{setting}_data.npz", X=X.astype(np.float32), Y=Y.astype(np.float32), phase=ph, engage=eng,
                            keep=keep, share=share.astype(np.float32), comb=comb.astype(np.float32), dist=dist.astype(np.float32),
                            contact=contact, t=ts, teacher_worlds=tw)
        line = (f"{ep_dir.name} {setting:16s} kept {keep.sum()}/{W}  holds/world R {nholds[:, 0].mean():.2f} L {nholds[:, 1].mean():.2f}  "
                f"two-hand fraction of hold frames {two_hand / max(hold_frames, 1):.2f}  phase share (idle/appr/acq/hold/rel) "
                + "/".join(f"{(ph[:, keep] == k).mean():.2f}" for k in range(5)))
        print(line); summary.append(line)
        if setting == "nominal":
            np.save(ep_dir / "engage_timeline.npy", (eng[:, keep].mean(1) > 0.5).astype(np.float32))   # (T, H) majority over worlds
            plot(ep_dir, setting, share, comb, eng, ph, contact, dist, pe, keep, sides)
    (ep_dir / "labels_summary.txt").write_text("\n".join(summary) + "\n")


def plot(ep_dir, setting, share, comb, eng, ph, contact, dist, pe, keep, sides):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    w = int(np.flatnonzero(keep)[0]) if keep.any() else 0
    t = np.arange(len(comb)); fig, ax = plt.subplots(len(sides) + 1, 1, figsize=(13, 7), sharex=True)
    names = ["idle", "approach", "acquire", "hold", "release"]; cols = ["white", "#d9e8f5", "#fde2b0", "#bfe3bf", "#f5c6c6"]
    for h, side in enumerate(sides):
        a = ax[h]
        for k in range(1, 5):
            for s, e in runs(ph[:, w, h] == k):
                a.axvspan(s, e, color=cols[k], label=names[k] if s == runs(ph[:, w, h] == k)[0][0] else None)
        a.plot(t, share[:, w, h], label=f"{side} share", c="k"); a.plot(t, comb[:, w], label="combined", c="gray", ls="--")
        a.plot(t, eng[:, w, h] * 1.05, label="engage", c="C3", lw=2); a.plot(t, contact[:, w, h] * 0.5, label="contact", c="C0", lw=0.8)
        a.plot(t, np.minimum(dist[:, w, h] / 0.15, 1.2), label="dist/15cm", c="C2", lw=0.8)
        a.set_ylabel(side); a.legend(loc="upper right", fontsize=7, ncol=4)
    ax[-1].plot(t, pe[:, w] * 100); ax[-1].set_ylabel("obj err cm"); ax[-1].set_xlabel("frame (20 Hz)")
    fig.suptitle(f"{ep_dir.name} {setting} world {w}"); fig.tight_layout(); fig.savefig(ep_dir / f"timeline_{setting}.png", dpi=110)


if __name__ == "__main__":
    main()
