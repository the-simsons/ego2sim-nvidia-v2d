#!/usr/bin/env bash
# Rebuild the 20 evaluation-episode references with the v2 chain (NOTES Phase 8): FoundationStereo depth for
# FoundationPose, n_particles=16, frame-0 static start (driver default F0_K=5), meshes_final (fs38 pitcher). Mesh-free
# stages (SAM2, MoGe, VO, hands) are reused. No training. Two GPU queues balanced by measured cost
# (FS 1.3 s/frame, FP 0.22 s/frame/object), then one retry pass for episodes without a reference (asset install race).
#   eval_rebuild_v2.sh            (both queues; run in tmux)
# Output: ~/v2d/ego2sim/train/epEP_v2/{processed/sequence_id=epEP/robot_name=sharpa_wave, reconstructed_stage}
#         ~/v2d/logs/eval_v2/epEP.log, summary.tsv (frames, per-object FP silhouette IoU median, frame-0 action)
set -uo pipefail
TAG=_v2; M=~/v2d/ego2sim/meshes_final; L=~/v2d/logs/eval_v2; mkdir -p $L
ROSTER=~/v2d/submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json
Q0="35 33 3 5 6 19 28 20 10 37"
Q1="25 36 1 30 32 38 8 16 15 14"
objs() { python3 -c "import json;print(','.join(json.load(open('$ROSTER'))['episodes']['$1']))"; }
ref() { ls ~/v2d/ego2sim/train/ep$1$TAG/processed/sequence_id=ep$1/robot_name=sharpa_wave/*.parquet 2>/dev/null | head -1; }
run_queue() {  # GPU EP...
  local g=$1; shift
  for e in "$@"; do
    SPLIT=evaluation TAG=$TAG MESH_SRC=$M DEPTH=stereo_fs FP_KW=n_particles=16 DEVICE=cuda:0 \
      bash ~/v2d/scripts/ego2sim_public_episode.sh "$e" "$(objs $e)" "$g" > $L/ep$e.log 2>&1
    echo "$(date +%T) GPU$g ep$e rc=$? $(tail -1 $L/ep$e.log | cut -c1-80)"
  done
}
echo "$(date +%T) start"
run_queue 0 $Q0 & run_queue 1 $Q1 & wait
for e in $Q0 $Q1; do [ -n "$(ref $e)" ] || { echo "$(date +%T) retry ep$e"; run_queue 0 $e; }; done
# quality summary (no mocap on the eval split): FP silhouette IoU vs the SAM2 mask per object
printf "episode\tframes\tobjects\tiou_median\tframe0\treference\n" > $L/summary.tsv
cd /tmp
for e in $Q0 $Q1; do
  E=~/v2d/ego2sim/ep$e; IFS=, read -ra NS <<< "$(objs $e)"; ious=()
  for i in "${!NS[@]}"; do n=${NS[$i]}
    ious+=("$(~/v2d/envs/v2d_foundation_pose/.venv/bin/python -I ~/v2d/tools/fp_track_diag.py $E/stereo/intrinsics_undist.json \
      $E/masks/$i $n=$E/fp_$n$TAG=$M/$n.obj --every 100000 2>/dev/null | awk '/IoU median/ {print $4}' | tr -d ,)")
  done
  f0=$(grep -h "fp_.*$TAG:" $L/ep$e.log | sed -E 's/.*(frame 0 [^(]*\([^)]*\)|not still).*/\1/' | tr '\n' ';')
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" $e $(ls $E/hands/frames | wc -l) "$(objs $e)" "$(IFS=,; echo "${ious[*]}")" "$f0" \
    "$([ -n "$(ref $e)" ] && echo ok || echo MISSING)" >> $L/summary.tsv
done
column -t -s $'\t' $L/summary.tsv
echo "$(date +%T) done"
