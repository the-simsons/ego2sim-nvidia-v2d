"""Held-out / failure candidates for GMD given G-9's training data: per episode and reconstruction, S0 best (64-world ADD-AUC,
own reference), best-policy video, VOC replay and its hand / object tracking error. Writes outputs/gmd/full/debug/CANDIDATES.md."""
import glob
import json
import os
import re

import numpy as np

R = "/usr1/e2s"
O = f"{R}/v2d_challenge_data/outputs"; T = f"{O}/rl_train_req/training_s0"; D = f"{O}/gmd/full/debug"
G_OBJ = {"white_pot", "mini_dust_pan", "plastic_dish_rack", "mini_sweeper", "white_pot_lid", "wooden_spoon", "beige_cup"}
G_PAIRS = {frozenset(p) for p in [("plastic_dish_rack", "white_pot"), ("mini_sweeper", "mini_dust_pan"), ("white_pot", "white_pot_lid"),
                                  ("beige_cup", "white_pot"), ("white_pot", "wooden_spoon")]}
G_TRAIN = {41, 40, 26, 18, 2, 35, 8, 3, 36}
USED = {38: "held out (v1)", 6: "held out (v1)", 37: "held out (v1)", 28: "failed (v1)", 19: "failed (v1)", 5: "failed (v1)"}
EVAL = {1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38}
OBJ = {1: "white_pot white_pot_lid", 5: "white_pot wooden_spoon", 6: "white_pot wooden_spoon", 10: "blue_cup beige_cup",
       14: "blue_cup plastic_dish_rack", 15: "plastic_dish_rack wooden_spoon", 16: "plastic_dish_rack wooden_spoon",
       19: "mini_sweeper mini_dust_pan", 20: "blue_cup water_pitcher", 25: "wooden_piece_1 wooden_piece_2",
       28: "plastic_dish_rack white_pot", 30: "mini_sweeper plastic_dish_rack", 32: "blue_cup wooden_spoon",
       33: "blue_cup wooden_spoon", 37: "wooden_spoon", 38: "mini_dust_pan", 0: "white_pot white_pot_lid", 11: "blue_cup beige_cup",
       12: "blue_cup plastic_dish_rack", 13: "blue_cup plastic_dish_rack", 22: "blue_cup water_pitcher",
       23: "wooden_piece_1 wooden_piece_2", 24: "wooden_piece_1 wooden_piece_2", 27: "plastic_dish_rack white_pot",
       31: "mini_sweeper plastic_dish_rack", 42: "white_pot", 43: "white_pot", 44: "white_pot", 45: "white_pot"}
RUN = {"v1": "eval/ep{e}_v1_cleanedref_ray5decay", "v2recon": "eval/ep{e}_v2recon_ray5decay", "v2clean": "eval/ep{e}_v2clean_ray5decay",
       "pub": None, "pubtm": None}
REL = lambda p: os.path.relpath(p, D)  # noqa: E731


def s0_best(run_glob):
    best = None
    for d in glob.glob(f"{T}/{run_glob}/p1_*"):
        for f in glob.glob(f"{d}/eval/eval_*_w64.log"):
            a = re.findall(r'"evaluation/add_auc": ([0-9.]+)', open(f).read())
            if a:
                s = int(re.findall(r"eval_(\d+)_w64", f)[0])
                if best is None or float(a[-1]) > best[0]:
                    best = (float(a[-1]), s, f"{d}/eval/policy_{s}.mp4")
    return best


def public_run(e, tm):
    """Best S0 run of a public episode on our recon (tm=False) or the teammate recon (tm=True); the suffix comes from the queue job files."""
    suffix = {}
    for jf in glob.glob(f"{T}/_scratch/jobs*.txt"):
        for line in open(jf):
            p = line.split()
            if len(p) > 3:
                suffix[p[2]] = p[3]
    best = None
    for d in glob.glob(f"{T}/ep{e}_*"):
        run = os.path.relpath(d, T)
        if (suffix.get(run, "-") == "_teammate") != tm:
            continue
        b = s0_best(run)
        if b and (best is None or b[0] > best[0]):
            best = b
    return best


def voc(tag, e):
    for p in (f"{D}/voc_cand/{tag}_ep{e}", f"{D}/voc/ep{e}_voc" if tag == "v1" else None):
        if p and os.path.exists(p + ".json") and os.path.exists(p + ".mp4"):
            d = json.load(open(p + ".json"))
            w = np.array([r["wrist_pos_m"] for r in d]); o = np.array([r["object_pos_m"] for r in d])
            return 100 * w.mean(), 100 * o.mean(), p + ".mp4"
    return None


def ego(e):
    return f"{O}/rian_eval_compare/ep{e}/ego_video_cam_a.mp4" if e in EVAL else \
        f"{R}/v2d_challenge_data/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_{e:06d}.mp4"


def fit(e):
    objs = OBJ[e].split()
    unseen = [o for o in objs if o not in G_OBJ]
    if unseen:
        return "partly unseen objects (" + ", ".join(o.replace("_", " ") for o in unseen) + ")"
    if len(objs) == 2 and frozenset(objs) not in G_PAIRS:
        return "objects seen, pair new"
    return "objects and pair seen"


rows = []
for e in sorted(OBJ):
    tags = ["pub", "pubtm"] if e not in EVAL else ["v1", "v2recon", "v2clean"]
    for tag in tags:
        b = s0_best(RUN[tag].format(e=e)) if RUN[tag] else public_run(e, tag == "pubtm")
        v = voc(tag, e)
        if b is None and v is None:
            continue
        rows.append(dict(e=e, tag=tag, b=b, v=v))

f3 = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
L = ["# GMD held-out / failure candidates", "",
     "Every episode not in G-9's training set, on every reconstruction we have. S0 best = best 64-world ADD-AUC of any S0 run on that",
     "reconstruction (scored against that same reconstruction, so numbers across reconstructions are not comparable). VOC error = mean over",
     "frames of the worse hand's wrist distance / the object distance to the reconstruction in the VOC physical replay: a small hand error",
     "means the reconstructed hands can physically do the motion; a large one means they collide or cannot reach (bad tracking).", "",
     "Reconstructions: v1 = Rian's cleaned v1 (what the held-out/failed runs used); v2recon = Ego2Sim v2; v2clean = v2 with cleaned object",
     "poses; pub = our public reconstruction (track3_handfix); pubtm = the teammate's public reconstruction. v2 meshes differ from v1",
     "(G-9's shape codes came from v1 / public meshes).", "",
     "G-9 objects: white pot, mini dust pan, plastic dish rack, mini sweeper, white pot lid, wooden spoon, beige cup. G-9 pairs: rack+pot,",
     "sweeper+dust pan, pot+lid, cup+pot, pot+spoon.", "",
     "| ep | objects | fit to G-9 | used before | recon | S0 best (ckpt) | VOC hand / object error | ego | best policy | VOC |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    e = r["e"]; objs = OBJ[e].split()
    ot = " + ".join(o.replace("_", " ") for o in objs) + (" **(2 objects)**" if len(objs) == 2 else " (1 object)")
    b, v = r["b"], r["v"]
    sb = f"{b[0]:.3f} ({b[1] / 1e6:.1f}M)" if b else "no S0 run"
    pol = f"[policy]({REL(b[2])})" if b and os.path.exists(b[2]) else "–"
    vt = f"{v[0]:.0f} cm / {v[1]:.1f} cm" if v else "–"
    vl = f"[VOC]({REL(v[2])})" if v else "–"
    L.append(f"| {e} | {ot} | {fit(e)} | {USED.get(e, '')} | {r['tag']} | {sb} | {vt} | [ego]({REL(ego(e))}) | {pol} | {vl} |")
open(f"{D}/CANDIDATES.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
