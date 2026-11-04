#!/usr/bin/env bash
# compose_clean2_videos.sh <ep> <prev_run> <prev_label>: ego video | previous best S0 run | clean2 run (best checkpoints, real time)
# -> rl_train_req/policy_videos_v2clean2/ep<ep>_ego_prev_clean2.mp4
set -euo pipefail
EP=$1; PREV=$2; PL=$3; E=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0/eval; OUT=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/policy_videos_v2clean2; mkdir -p $OUT
PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
info() {  # <run dir> -> "<video> <auc>"
  local D=$1 B P; B=$(awk -F': ' '{print $2}' $D/best_checkpoint.txt); P=$(ls -d $D/p1_*M)
  echo "$P/eval/policy_$B.mp4 $($PY -c "import json,os;f='$P/eval_final/metrics_${B}_w64.json';f=f if os.path.exists(f) else '$P/eval/metrics_$B.json';print('%.3f'%json.load(open(f))['metrics']['add_auc'])") $((B/1000000))"
}
read -r V1 A1 S1 < <(info $E/$PREV); read -r V2 A2 S2 < <(info $E/ep${EP}_v2clean2_ray5decay)
T="fontfile=$F:x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6"
ffmpeg -loglevel error -y -i /usr1/e2s/v2d_challenge_data/outputs/rian_eval_compare/ep$EP/ego_video_cam_a.mp4 -i $V1 -i $V2 -filter_complex \
  "[0]fps=20,scale=480:360,drawtext=$T:text='ep$EP ego video'[a];[1]fps=20,scale=640:360,drawtext=$T:text='$PL  AUC $A1 (${S1}M)'[b];[2]fps=20,scale=640:360,drawtext=$T:text='clean2  AUC $A2 (${S2}M)'[c];[a][b][c]hstack=3:shortest=1" \
  -c:v libx264 -crf 20 -pix_fmt yuv420p $OUT/ep${EP}_ego_prev_clean2.mp4
echo "$OUT/ep${EP}_ego_prev_clean2.mp4"
