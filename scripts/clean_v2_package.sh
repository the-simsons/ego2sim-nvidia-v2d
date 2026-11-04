#!/usr/bin/env bash
# Cleaned copy of the v2 eval package with the teammate's method (NOTES Phase 8g): tools/clean_refs_v2.py stage1 (cup
# de-spin, flip unwrap, spikes, lift onto the table) -> tools/settle_spawn_v2.py per episode (frame-0 physics rest pose,
# native flash_chord venv) -> stage2 (spawn fix). Source package is only read.
#   clean_v2_package.sh [GPU] [EP ...]      (default: all 20 eval episodes)
# Output: /usr1/e2s/ego2sim_eval_recon_v2_clean/{package/, epEP_report.json, epEP_settle.json, epEP_cleaned.npz}
set -euo pipefail
GPU=${1:-0}; shift || true
EPS=${*:-"1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38"}
SRC=/usr1/e2s/ego2sim_eval_recon_v2; OUT=/usr1/e2s/ego2sim_eval_recon_v2_clean
PY=~/v2d/envs/kit/.venv/bin/python; FC=~/v2d/vtd_pinned/robotic_grounding/flash_chord
source ~/v2d/env.sh
mkdir -p $OUT; cd /tmp
$PY -I ~/v2d/tools/clean_refs_v2.py $SRC $OUT stage1 $EPS 2>&1 | grep -v -i warn
for EP in $EPS; do
  (cd $FC && CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python ~/v2d/tools/settle_spawn_v2.py \
    $OUT/package/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave $OUT/ep${EP}_settle.json replay.voc_scale=0.0 2>&1 | grep -E "settle:|Error|Traceback" | sed "s/^/ep$EP /")
done
$PY -I ~/v2d/tools/clean_refs_v2.py $SRC $OUT stage2 $EPS 2>&1 | grep -v -i warn
echo "-> $OUT/package"
