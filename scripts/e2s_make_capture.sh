#!/usr/bin/env bash
# AgentSTAR Ego2Sim capture for one object of a prepared episode (stereo/, masks/, hand_masks/ exist):
#   inputs -> Pi3X capture (every 10th frame, depth) -> factory K -> depth-ordered occluders, stereo size target, stereo depth
#   -> ~/v2d/agenticSTAR/captures/<obj>_e2s + ego2sim/specs/ep<EP>_<obj>.json. Then the USER launches (sudo):
#   ~/v2d/scripts/launch_e2s_riddle.sh <obj>_e2s ep<EP>_<obj> [N] [GPU]
#   e2s_make_capture.sh EP MASK_ID OBJ [OTHER_MASK_ID|-] [GPU]
# Env DEPTH=stereo (SGBM, default) or stereo_fs (FoundationStereo, fs_validate_episode.sh): the folder under epEP/ whose
# stereo_depth/ gives the metric scale, occluder order, size target and capture depth. Non-default DEPTH writes
# captures/<obj>_e2s_<suffix> and spec ep<EP>_<obj>_<suffix> (SUFFIX, default "fs"), leaving the SGBM capture untouched.
set -euo pipefail
source ~/v2d/env.sh
EP=$1; ID=$2; n=$3; OTHER=${4:--}; GPU=${5:-0}
export CUDA_VISIBLE_DEVICES=$GPU
E2=~/v2d/ego2sim; E=$E2/ep$EP; S=$E/stereo; PY=~/v2d/envs/v2d_sam2/.venv/bin/python; T=~/v2d/tools/ego2sim_prep.py
DEPTH=${DEPTH:-stereo}; D=$E/$DEPTH; SFX=""; [ "$DEPTH" != stereo ] && SFX=_${SUFFIX:-fs}
C=$E/captures/$n$SFX; I=$E/inputs/$n; W=$E/e2s_$n$SFX
$PY -I ~/v2d/tools/agentstar_prep.py inputs $S/cam_a_undist.mp4 $E/masks $E/hand_masks $E/inputs $ID:$n
cd ~/v2d/agenticSTAR
rm -rf $C; SECONDS=0
micromamba run -n pi3x python tools/make_capture.py --src $I --out $C --every 10 --depth --fps 20.25 --name ep${EP}_$n 2>&1 \
  | grep -v -i warn | tail -1; echo "capture ${SECONDS}s"
$PY -I ~/v2d/tools/agentstar_prep.py metric $C $D/stereo_depth | head -2
$PY -I - $C/tracking/cameras.npz $S/intrinsics_undist.json <<'EOF'
import json, sys
import numpy as np
c = dict(np.load(sys.argv[1])); Wp, Hp = c["size"]
K = json.load(open(sys.argv[2])); sx, sy = Wp / K["width"], Hp / K["height"]
Kp = np.array([[K["fx"] * sx, 0, K["cx"] * sx], [0, K["fy"] * sy, K["cy"] * sy], [0, 0, 1]], np.float32)
c["intrinsics"] = np.repeat(Kp[None], len(c["intrinsics"]), 0); np.savez(sys.argv[1], **c); print("K set to factory", Kp[0, 0].round(1))
EOF
rm -rf $W; mkdir -p $W
others=""; [ "$OTHER" != "-" ] && others=$E/masks/$OTHER
hands=$(ls -d $E/hand_masks/* | tr "\n" "," | sed "s/,$//")
$PY $T occluders $I "$others" "$hands" $D $W/mask_hand --margin 0.02 | tail -1
$PY $T size $I $D $W/size_target.json | cut -c1-140
$PY $T stereo_depth $C $D $W/depth_stereo | tail -1
for f in $C/mask_hand/*.png; do cp $W/mask_hand/$(basename $f) $f; done
for f in $C/depth/*.npz; do cp $W/depth_stereo/$(basename $f) $f; done
rm -rf captures/${n}_e2s$SFX; cp -a $C captures/${n}_e2s$SFX; mkdir -p ego2sim/specs
$PY - $W/size_target.json ego2sim/specs/ep${EP}_$n$SFX.json $n $EP captures/${n}_e2s$SFX/capture.json $DEPTH <<'EOF'
import json, sys
size = json.load(open(sys.argv[1]))
spec = {"object": sys.argv[3], "episode": int(sys.argv[4]), "target_longest_m": round(size["target_longest_m"], 4),
        "size_tolerance": 0.08, "size_source": size["method"], "iou_median_min": 0.93, "iou_min": 0.85, "max_iterations": 12}
json.dump(spec, open(sys.argv[2], "w"), indent=2)
method = "FoundationStereo" if sys.argv[6] == "stereo_fs" else "SGBM"
m = json.load(open(sys.argv[5])); m.setdefault("source", {})["depth_values"] = f"{method} stereo, {sys.argv[6]}/ (ego2sim_prep.py stereo_depth); Pi3X cameras kept"
m["source"]["occluders"] = "depth-ordered (ego2sim_prep.py occluders, margin 0.02)"; json.dump(m, open(sys.argv[5], "w"), indent=2)
print("spec", sys.argv[2], "target", spec["target_longest_m"], "m")
EOF
echo "ready: ~/v2d/scripts/launch_e2s_riddle.sh ${n}_e2s$SFX ep${EP}_$n$SFX"