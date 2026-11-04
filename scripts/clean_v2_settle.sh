#!/usr/bin/env bash
# Second half of clean_v2_package.sh: waits until a GPU is completely free (< 100 MiB used, no compute apps, 3 checks a
# minute apart; shared machine), then frame-0 settle per episode (tools/settle_spawn_v2.py), stage2, and a hygiene summary.
#   clean_v2_settle.sh            (run in tmux; log ~/v2d/logs/clean_v2/settle.log)
set -uo pipefail
EPS="1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38"
SRC=/usr1/e2s/ego2sim_eval_recon_v2; OUT=/usr1/e2s/ego2sim_eval_recon_v2_clean
PY=~/v2d/envs/kit/.venv/bin/python; FC=~/v2d/vtd_pinned/robotic_grounding/flash_chord
source ~/v2d/env.sh
free_gpu() {  # prints the index of a GPU with nothing on it, else nothing
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | while IFS=', ' read -r i u m; do
    [ "$m" -lt 100 ] && ! grep -q "$u" <<< "$apps" && { echo "$i"; break; }; done; }
echo "$(date +%T) waiting for a free GPU"
n=0
while [ $n -lt 3 ]; do g=$(free_gpu); if [ -n "$g" ]; then n=$((n + 1)); else n=0; fi; sleep 60; done
g=$(free_gpu); [ -n "$g" ] || { echo "GPU taken again; rerun"; exit 1; }
echo "$(date +%T) GPU $g free: settling"
cd /tmp
for EP in $EPS; do
  (cd $FC && CUDA_VISIBLE_DEVICES=$g .venv/bin/python ~/v2d/tools/settle_spawn_v2.py \
    $OUT/package/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave $OUT/ep${EP}_settle.json replay.voc_scale=0.0 2>&1 | grep -E "settle:|Error|Traceback" | sed "s/^/ep$EP /")
done
$PY -I ~/v2d/tools/clean_refs_v2.py $SRC $OUT stage2 $EPS 2>&1 | grep -v -i warn
$PY -I ~/v2d/tools/ref_hygiene.py raw=$SRC clean=$OUT/package --eps ${EPS// /,} 2>&1 | grep -v -i warn
echo "$(date +%T) done -> $OUT/package"
