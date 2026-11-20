"""Symmetry-aware object track and frame 0 (plans/RECONSTRUCTION_PLAN.md P3.1 + P3.4). Video only.

A tracker that sees a symmetric (or nearly symmetric) mesh can report any look-alike twin of the true pose, and can
jump between twins mid-clip. This replaces the tracker's per-frame choice with a global one:

  1. Per frame, the candidates are the tracked pose and its twins T @ S_c for each declared symmetry S, turned about
     the mesh's bounding-box centre (so twins share the same centre).
  2. Viterbi over the clip in the WORLD frame (video camera, e.g. tools/rgbd_vo.py), where a resting object does not
     move when the camera does: transition cost = rotation change (deg) + POS_WEIGHT x centre change (cm), plus a
     small cost for leaving the tracker's own choice, so the raw twin is kept wherever nothing argues against it.
  3. Frame 0 (P3.4): the first still stretch of the chosen track is replaced by its robust mean (outlier frames off
     the median by more than --tol-cm / --tol-deg are dropped), so the pose that anchors everything downstream comes
     from many frames, and its twin is the one consistent with the later motion (step 2 runs before step 3).
     --hold all holds every still stretch at its mean (as build_ego_bundle.py's still-snap does).

Not yet: top-k hypotheses from FoundationPose itself (needs a wrapper around the pinned module and a GPU run), and
the table-plane and hand-contact terms (need gravity from depth and hand tracks per frame). With only the tracker's
own pose per frame, twins of a perfectly symmetric mesh are indistinguishable; the gain is no mid-clip flips and a
steadier frame 0.

    python tools/sym_track.py --poses fp_box --camera camera_vo_video.npz --mesh BlueBox.obj \
        --symmetry '{"flips_180": ["x", "y", "z"]}' --fps 30 --out fp_sym [--object-masks M --hand-masks H]
Writes per-frame pose JSONs (camera <- mesh, same format as the input) and OUT/sym_track.json (stats)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e2s.geometry import load_pose_dir, mean_pose, pose_to_dict, rotation_angle_deg, runs, still_mask, symmetry_rotations  # noqa: E402

POS_WEIGHT = 1.0      # deg of rotation change worth 1 cm of centre change
STAY_COST = 0.5       # per frame, for using a twin other than the tracker's own pose


def mesh_centre(path):
    v = np.array([[float(x) for x in l.split()[1:4]] for l in open(path) if l.startswith("v ")])
    return (v.min(0) + v.max(0)) / 2


def twins(sym, centre):
    """Symmetry turns about the centre: T(c) S T(-c)."""
    C, Ci = np.eye(4), np.eye(4)
    C[:3, 3], Ci[:3, 3] = centre, -centre
    return [C @ S @ Ci for S in symmetry_rotations(sym)]


def viterbi(W, S, centre, pos_weight=POS_WEIGHT, stay=STAY_COST):
    """W (N,4,4) world <- mesh, NaN where missing. -> chosen twin index per frame (-1 where missing)."""
    N, K = len(W), len(S)
    ok = ~np.isnan(W[:, 0, 0])
    idx = np.nonzero(ok)[0]
    cand = np.stack([W[idx] @ s for s in S], 1)                                   # M,K,4,4
    ctr = np.einsum("mkij,j->mki", cand[..., :3, :3], centre) + cand[..., :3, 3]   # M,K,3
    unary = np.full(K, stay)
    unary[0] = 0.0
    cost, back = unary.copy(), np.zeros((len(idx), K), int)
    for m in range(1, len(idx)):
        tr = np.einsum("aij,bij->ab", cand[m, :, :3, :3], cand[m - 1, :, :3, :3])
        dr = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
        dp = np.linalg.norm(ctr[m][:, None] - ctr[m - 1][None], axis=2) * 100
        t = cost[None] + dr + pos_weight * dp
        back[m] = np.argmin(t, 1)
        cost = t.min(1) + unary
    s = np.zeros(len(idx), int)
    s[-1] = int(np.argmin(cost))
    for m in range(len(idx) - 1, 0, -1):
        s[m - 1] = back[m, s[m]]
    out = np.full(N, -1)
    out[idx] = s
    return out


def first_contact(obj_dir, hand_dir, n, px, min_overlap=50):
    """First frame where the dilated hand mask overlaps the object mask (n if never)."""
    import cv2
    hd = Path(hand_dir)
    subs = [d for d in sorted(hd.iterdir()) if d.is_dir()] or [hd]
    ker = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    for i in range(n):
        o = cv2.imread(str(Path(obj_dir) / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if o is None:
            continue
        h = np.zeros_like(o)
        for d in subs:
            x = cv2.imread(str(d / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if x is not None:
                h |= (cv2.resize(x, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        if ((cv2.dilate(h, ker) > 0) & (o > 0)).sum() > min_overlap:
            return i
    return n


def robust_mean(T, tol_cm, tol_deg):
    med = mean_pose(T)
    dp = np.linalg.norm(T[:, :3, 3] - med[:3, 3], axis=1) * 100
    dr = rotation_angle_deg(T[:, :3, :3], np.repeat(med[None, :3, :3], len(T), 0))
    keep = (dp < tol_cm) & (dr < tol_deg)
    return (mean_pose(T[keep]) if keep.sum() >= 3 else med), int(keep.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--poses", required=True, help="dir of per-frame camera <- mesh JSONs")
    ap.add_argument("--camera", required=True, help="npz with camera_to_world (video-only camera)")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--symmetry", required=True, help='JSON (or a file): {"flips_180": [...], "revolution_axis": ...}')
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", choices=("first", "all", "none"), default="first")
    ap.add_argument("--still-speed", type=float, default=0.02, help="m/s in the reconstructed world")
    ap.add_argument("--min-still-s", type=float, default=0.3)
    ap.add_argument("--tol-cm", type=float, default=1.5)
    ap.add_argument("--tol-deg", type=float, default=5.0)
    ap.add_argument("--object-masks", help="object mask dir (NNNNNN.png): with --hand-masks, the still start ends at the first hand contact")
    ap.add_argument("--hand-masks", help="hand mask dir (a dir of per-track subdirs is merged)")
    ap.add_argument("--contact-px", type=int, default=15, help="hand mask dilation (full-resolution px) that counts as contact")
    a = ap.parse_args(argv)
    sym = json.loads(Path(a.symmetry).read_text()) if Path(a.symmetry).is_file() else json.loads(a.symmetry)

    Tcm = load_pose_dir(a.poses)
    c2w = np.load(a.camera)["camera_to_world"].astype(float)
    N = min(len(Tcm), len(c2w))
    Tcm, c2w = Tcm[:N], c2w[:N]
    W = c2w @ np.nan_to_num(Tcm)
    W[np.isnan(Tcm[:, 0, 0])] = np.nan
    centre = mesh_centre(a.mesh)
    S = twins(sym, centre)
    choice = viterbi(W, S, centre)
    ok = choice >= 0
    Wc = W.copy()
    Wc[ok] = np.stack([W[i] @ S[choice[i]] for i in np.nonzero(ok)[0]])

    ctr = np.einsum("nij,j->ni", np.nan_to_num(Wc[:, :3, :3]), centre) + np.nan_to_num(Wc[:, :3, 3])
    # tracker noise alone exceeds the speed threshold frame to frame: detect stillness on a 0.5 s moving average
    k = max(1, int(0.5 * a.fps))
    ctr_s = np.stack([np.convolve(np.pad(ctr[:, j], (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
                      for j in range(3)], 1)
    still = still_mask(ctr_s, a.fps, speed=a.still_speed) & ok
    st = runs(still, int(a.min_still_s * a.fps))
    contact = first_contact(a.object_masks, a.hand_masks, N, a.contact_px) if a.object_masks and a.hand_masks else None
    if contact is not None and a.hold == "first":
        # video-only stillness cue from the recording protocol: the object rests until a hand reaches it. World speed
        # cannot show this while the camera path drifts (MoGe-depth odometry: ~5 cm/s apparent motion at rest).
        st = [(0, contact)] if contact >= int(a.min_still_s * a.fps) else []
    held = []
    for k, (s0, s1) in enumerate(st):
        if a.hold == "none" or (a.hold == "first" and k > 0):
            break
        if a.hold == "first" and s0 > int(0.5 * a.fps):
            break                                    # the clip does not start still: leave frame 0 as tracked
        M, n_used = robust_mean(Wc[s0:s1], a.tol_cm, a.tol_deg)
        Wc[s0:s1] = M
        held.append({"frames": [int(s0), int(s1)], "used": n_used})

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in np.nonzero(ok)[0]:
        json.dump(pose_to_dict(np.linalg.inv(c2w[i]) @ Wc[i]), open(out / f"{i:06d}.json", "w"), indent=1)

    def jumps(T):
        v = ~np.isnan(T[:, 0, 0])
        R = T[v][:, :3, :3]
        return int((rotation_angle_deg(R[1:], R[:-1]) > 90).sum())

    stats = {"frames": int(N), "tracked": int(ok.sum()), "twins": len(S),
             "twin_counts": np.bincount(choice[ok], minlength=len(S)).tolist(),
             "twin_switches": int((np.diff(choice[ok]) != 0).sum()),
             "raw_jumps_gt90deg": jumps(W), "out_jumps_gt90deg": jumps(Wc), "held_still": held, "first_hand_contact": contact,
             "symmetry": sym, "camera": str(a.camera), "source": str(a.poses), "trajectory_source": "video"}
    json.dump(stats, open(out / "sym_track.json", "w"), indent=1)
    print(f"{out}: {ok.sum()}/{N} frames, twins {stats['twin_counts']}, switches {stats['twin_switches']}, "
          f"jumps>90deg {stats['raw_jumps_gt90deg']} -> {stats['out_jumps_gt90deg']}, held {held}")


if __name__ == "__main__":
    main()
