"""Pick and refine the ego-rig extrinsics on the published videos.

Pools SIFT matches for the pairs (a,b), (a,c), (b,c) over several episodes/frames, scores each extrinsics hypothesis
by the robust (median) Sampson epipolar error in pixels, then refines R and the translation DIRECTION of cam_b and
cam_c relative to cam_a with a soft-L1 loss over all three pairs jointly. Baseline LENGTHS stay at the factory values
(epipolar geometry cannot see scale). Writes refs/calib/rig.json + a report.

Usage: python -m handbench.tools.calib_check [--episodes 0,12,23,40,44] [--frames 6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from handbench import paths
from handbench.calib import CAMS, Cam, load_factory, save_rig

PAIRS = [("ego_cam_a", "ego_cam_b"), ("ego_cam_a", "ego_cam_c"), ("ego_cam_b", "ego_cam_c")]


def read_frame(ep: int, cam: str, i: int):
    cap = cv2.VideoCapture(str(paths.video(ep, cam)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, im = cap.read()
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if ok else None


def sift_matches(A, B, sA=1.0, sB=1.0):
    sift = cv2.SIFT_create(6000)
    a = cv2.resize(A, None, fx=sA, fy=sA, interpolation=cv2.INTER_AREA) if sA != 1 else A
    b = cv2.resize(B, None, fx=sB, fy=sB, interpolation=cv2.INTER_AREA) if sB != 1 else B
    k1, d1 = sift.detectAndCompute(a, None)
    k2, d2 = sift.detectAndCompute(b, None)
    if d1 is None or d2 is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    m = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.7 * y.distance]
    p = np.float64([k1[g.queryIdx].pt for g in good]) / sA
    q = np.float64([k2[g.trainIdx].pt for g in good]) / sB
    return p, q


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def sampson_px(n1, n2, R, t, f):
    """Sampson distance (in px of focal f) for normalised points n1 (cam1) n2 (cam2), X2 = R X1 + t."""
    E = skew(t) @ R
    x1 = np.c_[n1, np.ones(len(n1))]
    x2 = np.c_[n2, np.ones(len(n2))]
    Ex1 = x1 @ E.T
    Etx2 = x2 @ E
    num = np.sum(x2 * Ex1, 1)
    den = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    return f * num / np.sqrt(den + 1e-18)


def rel(cams, i, j):
    T = cams[j].T_from_a @ np.linalg.inv(cams[i].T_from_a)
    return T[:3, :3], T[:3, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="0,11,12,23,31,40,44")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--out", default=str(paths.REFS / "calib" / "rig.json"))
    a = ap.parse_args()
    calib = paths.calib_json()
    eps = [int(e) for e in a.episodes.split(",")]
    base = load_factory(calib, "raw")
    scale = {"ego_cam_a": 0.5, "ego_cam_b": 1.0, "ego_cam_c": 1.0}
    data = {p: [[], []] for p in PAIRS}
    for ep in eps:
        n = paths.num_frames(ep)
        for fi in np.linspace(0, n - 1, a.frames).astype(int):
            ims = {c: read_frame(ep, c, int(fi)) for c in CAMS}
            for i, j in PAIRS:
                p, q = sift_matches(ims[i], ims[j], scale[i], scale[j])
                if len(p) < 8:
                    continue
                # keep geometric inliers of a loose fundamental-matrix fit (removes outlier matches, not hypotheses)
                Fm, inl = cv2.findFundamentalMat(p, q, cv2.FM_RANSAC, 3.0, 0.999)
                if inl is None:
                    continue
                inl = inl.ravel() > 0
                data[(i, j)][0].append(base[i].undistort_px(p[inl]))
                data[(i, j)][1].append(base[j].undistort_px(q[inl]))
        print(f"ep {ep}: " + ", ".join(f"{i[-1]}{j[-1]} {sum(len(x) for x in data[(i, j)][0])}" for i, j in PAIRS))
    data = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in data.items()}
    f_of = {c: base[c].K[0, 0] for c in CAMS}

    def score(cams):
        out = {}
        for i, j in PAIRS:
            R, t = rel(cams, i, j)
            e = sampson_px(*data[(i, j)], R, t / np.linalg.norm(t), min(f_of[i], f_of[j]))
            out[f"{i[-1]}{j[-1]}"] = {"median_px": float(np.median(np.abs(e))), "p90_px": float(np.percentile(np.abs(e), 90)),
                                      "n": int(len(e))}
        return out

    hyps = {"factory_raw": load_factory(calib, "raw"), "factory_conj": load_factory(calib, "conj")}
    report = {"episodes": eps, "hypotheses": {k: score(v) for k, v in hyps.items()}}
    for k, v in report["hypotheses"].items():
        print(f"{k:14s} " + "  ".join(f"{p}: med {s['median_px']:.2f}px p90 {s['p90_px']:.2f}" for p, s in v.items()))
    best = min(hyps, key=lambda k: sum(s["median_px"] for s in report["hypotheses"][k].values()))
    start = hyps[best]
    L = {c: np.linalg.norm(start[c].center_in_a) for c in ("ego_cam_b", "ego_cam_c")}

    def unpack(x):
        cams = {"ego_cam_a": Cam("ego_cam_a", base["ego_cam_a"].K, base["ego_cam_a"].dist, base["ego_cam_a"].size)}
        for k, c in enumerate(("ego_cam_b", "ego_cam_c")):
            R = Rot.from_rotvec(x[6 * k: 6 * k + 3]).as_matrix()
            d = x[6 * k + 3: 6 * k + 6]
            C = d / np.linalg.norm(d) * L[c]  # camera centre in a frame
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, -R @ C
            cams[c] = Cam(c, base[c].K, base[c].dist, base[c].size, T)
        return cams

    x0 = np.concatenate([np.r_[Rot.from_matrix(start[c].R).as_rotvec(), start[c].center_in_a] for c in ("ego_cam_b", "ego_cam_c")])

    def resid(x):
        cams = unpack(x)
        r = []
        for i, j in PAIRS:
            R, t = rel(cams, i, j)
            r.append(sampson_px(*data[(i, j)], R, t / np.linalg.norm(t), min(f_of[i], f_of[j])))
        return np.concatenate(r)

    sol = least_squares(resid, x0, loss="soft_l1", f_scale=1.0)
    refined = unpack(sol.x)
    report["start"] = best
    report["refined"] = score(refined)
    for c in ("ego_cam_b", "ego_cam_c"):
        dR = np.degrees(Rot.from_matrix(refined[c].R @ start[c].R.T).magnitude())
        dC = np.degrees(np.arccos(np.clip(refined[c].center_in_a @ start[c].center_in_a / L[c] ** 2, -1, 1)))
        report[f"{c}_change_vs_{best}"] = {"rot_deg": float(dR), "t_dir_deg": float(dC),
                                           "center_in_a_m": refined[c].center_in_a.tolist()}
        print(f"{c}: refined vs {best}: rot {dR:.3f} deg, t dir {dC:.2f} deg, centre in a {np.round(refined[c].center_in_a, 4)}")
    print("refined        " + "  ".join(f"{p}: med {s['median_px']:.2f}px p90 {s['p90_px']:.2f}" for p, s in report["refined"].items()))
    save_rig(refined, a.out, {"source": f"factory intrinsics; extrinsics refined from {best} on SIFT matches",
                              "report": report})
    json.dump(report, open(Path(a.out).with_name("calib_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
