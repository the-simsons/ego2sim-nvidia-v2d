"""EgoFound3R inference_output.pt -> export.npz with MANO parameters fitted to its 778-vertex hands.

Runs in EgoFound3R's env (needs torch to read the output and to fit). Fitting uses a torch copy of handbench.mano's
forward pass (same model file, same left = mirrored right convention, scale fixed at 1): per hand-frame, global_orient,
hand_pose, betas and transl are fitted to EgoFound3R's vertices by Adam on the mean squared vertex error with small
pose / shape priors. The fit residual is stored per hand-frame.

  ../methods/.venv-egofound3r/bin/python tools/egofound3r_export.py data/egofound3r/track3/episode_000041
Writes <dir>/export.npz:
  joints_native (T,2,21,3), verts778 (T,2,778,3)   EgoFound3R hands, camera frame, metres; side 0 = L, 1 = R
  valid (T,2), predicted (T,2)    predicted = valid, on an H-rate frame (not interpolated to the input rate), not filled
  global_orient (T,2,3), hand_pose (T,2,45), betas (T,2,10), transl (T,2,3), fit_err_mean_mm / fit_err_max_mm (T,2)
  K_pred (T,3,3) in input-video pixels, high_frame_indices, h_fps, input_fps, meta (JSON)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from handbench.fit import refit_global  # noqa: E402
from handbench.mano import JOINT_ORDER, TIP_VERTS, Mano  # noqa: E402

MIRROR = np.diag([-1.0, 1.0, 1.0])


class TorchMano:
    """handbench.mano.Mano.forward for the right model, unscaled, in torch."""

    def __init__(self, m: Mano, device="cpu"):
        f = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)  # noqa: E731
        self.v_template, self.shapedirs, self.posedirs = f(m.v_template), f(m.shapedirs), f(m.posedirs)
        self.J_regressor, self.weights, self.parents = f(m.J_regressor), f(m.weights), m.parents

    @staticmethod
    def rodrigues(aa):
        th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        k = aa / th
        kx, ky, kz = k.unbind(-1)
        z = torch.zeros_like(kx)
        K = torch.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], -1).reshape(aa.shape[:-1] + (3, 3))
        s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
        return torch.eye(3, dtype=aa.dtype, device=aa.device) + s * K + (1 - c) * (K @ K)

    def forward(self, go, hp, be, transl):
        N = go.shape[0]
        R = self.rodrigues(torch.cat([go[:, None], hp.reshape(N, 15, 3)], 1))
        v_shaped = self.v_template[None] + torch.einsum("vck,nk->nvc", self.shapedirs, be)
        J = torch.einsum("jv,nvc->njc", self.J_regressor, v_shaped)
        eye = torch.eye(3, dtype=go.dtype, device=go.device)
        v_posed = v_shaped + torch.einsum("vck,nk->nvc", self.posedirs, (R[:, 1:] - eye).reshape(N, 135))
        Gs = [torch.cat([torch.cat([R[:, 0], J[:, 0, :, None]], 2), self._row(N, go)], 1)]
        for j in range(1, 16):
            p = int(self.parents[j])
            L = torch.cat([torch.cat([R[:, j], (J[:, j] - J[:, p])[:, :, None]], 2), self._row(N, go)], 1)
            Gs.append(Gs[p] @ L)
        G = torch.stack(Gs, 1)
        joints16 = G[:, :, :3, 3]
        t = G[:, :, :3, 3] - torch.einsum("njab,njb->nja", G[:, :, :3, :3], J)
        T_R = torch.einsum("vj,njab->nvab", self.weights, G[:, :, :3, :3])
        T_t = torch.einsum("vj,nja->nva", self.weights, t)
        verts = torch.einsum("nvab,nvb->nva", T_R, v_posed) + T_t
        joints = torch.cat([joints16, verts[:, TIP_VERTS]], 1)[:, JOINT_ORDER]
        return verts + transl[:, None], joints + transl[:, None]

    @staticmethod
    def _row(N, ref):
        r = torch.zeros(N, 1, 4, dtype=ref.dtype, device=ref.device)
        r[..., 3] = 1
        return r


def fit_side(tm: TorchMano, mano: Mano, V: np.ndarray, is_right: bool, iters=(600, 400)):
    """V (n,778,3) camera frame -> (go, hp, be, transl, err_mean, err_max) in handbench.mano convention."""
    n = len(V)
    M = np.eye(3) if is_right else MIRROR
    tgt = V @ M.T  # right-model space
    go0, tr0 = np.zeros((n, 3)), np.zeros((n, 3))
    for i in range(n):
        go0[i], tr_c, _ = refit_global(mano, np.zeros(45), np.zeros(10), True, tgt[i], with_scale=False)
        tr0[i] = tr_c
    d = lambda x: torch.tensor(x, dtype=torch.float64, requires_grad=True)  # noqa: E731
    go, hp, be, tr = d(go0), d(np.zeros((n, 45))), d(np.zeros((n, 10))), d(tr0)
    T = torch.as_tensor(tgt, dtype=torch.float64)
    for n_it, lr in zip(iters, (0.03, 0.005)):
        opt = torch.optim.Adam([go, hp, be, tr], lr=lr)
        for _ in range(n_it):
            v, _ = tm.forward(go, hp, be, tr)
            loss = ((v - T) ** 2).sum(-1).mean() + 1e-6 * (hp ** 2).sum(-1).mean() + 1e-6 * (be ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        v, _ = tm.forward(go, hp, be, tr)
        e = (v - T).norm(dim=-1)
    go, hp, be, tr = (x.detach().numpy() for x in (go, hp, be, tr))
    # back to handbench convention: left = mirror of right-model output, transl added after mirroring
    tr = tr @ M.T
    # check against the numpy model (exact convention used by the evaluator)
    vn, _ = mano.forward(go, hp, be, tr, is_right, 1.0)
    assert np.abs(vn - V).max(-1).max() < 1.0, "fit diverged"
    return go, hp, be, tr, e.mean(-1).numpy() * 1000, e.max(-1).values.numpy() * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="EgoFound3R output dir (with inference_output.pt)")
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(0)
    run = Path(a.run_dir)
    d = torch.load(run / "inference_output.pt", map_location="cpu", weights_only=False)
    mp = d["metric_predictions"]
    J = mp["hand"]["joint_xyz_metric"].numpy().astype(np.float64)
    V = mp["hand_778"]["vertex_xyz_metric"].numpy().astype(np.float64)
    valid = mp["hand_778"]["valid"].numpy() & mp["hand"]["valid"].numpy()
    T = J.shape[0]
    hi = d["high_frame_indices"].numpy()
    on_h = np.zeros(T, bool)
    on_h[hi] = True
    predicted = valid & on_h[:, None] & ~d["hand_missing_fill_full"].numpy()
    # model-input pixels -> input-video pixels
    A = np.array(d["provenance"]["input_affine_source_to_model"], np.float64)
    K_pred = np.linalg.inv(A)[None] @ d["hand_projection_K_full"].numpy().astype(np.float64)

    mano = Mano()
    tm = TorchMano(mano)
    out = {k: np.zeros((T, 2) + s) for k, s in
           (("global_orient", (3,)), ("hand_pose", (45,)), ("betas", (10,)), ("transl", (3,)))}
    out["fit_err_mean_mm"], out["fit_err_max_mm"] = np.full((T, 2), np.nan), np.full((T, 2), np.nan)
    for s, is_right in ((0, False), (1, True)):
        idx = np.nonzero(valid[:, s])[0]
        go, hp, be, tr, em, ex = fit_side(tm, mano, V[idx, s], is_right)
        for k, x in zip(("global_orient", "hand_pose", "betas", "transl", "fit_err_mean_mm", "fit_err_max_mm"),
                        (go, hp, be, tr, em, ex)):
            out[k][idx, s] = x
    ok = predicted
    meta = {"source": str(run / "inference_output.pt"), "egofound3r_provenance": d["provenance"],
            "h_fps": float(d["h_fps"]), "input_fps": float(d["r_fps"]),
            "fit": "per hand-frame Adam on mean sq. vertex error (600 it lr 3e-2, 400 it lr 5e-3), scale fixed 1",
            "fit_err_mean_mm_median": float(np.nanmedian(out["fit_err_mean_mm"][ok])),
            "fit_err_max_mm_median": float(np.nanmedian(out["fit_err_max_mm"][ok])),
            "fit_err_max_mm_p95": float(np.nanpercentile(out["fit_err_max_mm"][ok], 95)),
            "export_wall_time_s": time.time() - t0}
    np.savez_compressed(run / "export.npz", joints_native=J.astype(np.float32), verts778=V.astype(np.float32),
                        valid=valid, predicted=predicted, K_pred=K_pred, high_frame_indices=hi,
                        h_fps=float(d["h_fps"]), input_fps=float(d["r_fps"]), meta=json.dumps(meta, default=str),
                        **out)
    print(json.dumps({k: v for k, v in meta.items() if k != "egofound3r_provenance"}, default=str))


if __name__ == "__main__":
    main()
