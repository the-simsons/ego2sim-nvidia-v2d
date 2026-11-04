#!/usr/bin/env bash
# After all M runs of the full experiment: real-time S0 vs M+G videos of each run's best checkpoint, curves, summary.
set -uo pipefail
cd /usr1/e2s/retarget_env/gmd
G=/usr1/e2s/v2d_challenge_data/outputs/gmd/full; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
RUNS="B_ep38 B_ep6 B_ep37 C_ep28 C_ep19 C_ep5 A_ep41 A_ep26"
all_done() { for r in $RUNS; do [[ -f $G/m/$r/DONE ]] || return 1; done; }
until all_done; do sleep 60; done
while docker ps --format '{{.Names}}' | grep -q "gmd_me_\|gmd_m_"; do sleep 30; done
echo "all runs done $(date -u)" >> $G/logs/finish.log
mkdir -p $G/videos
V="viewer.backend=mp4 viewer.video_fps=20 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra"
render() {  # <gpu> <run>
  local GPU=$1 R=$2 EP=${2##*_ep} BEST CP CT N
  BEST=$($PY -c "import json;r=[json.loads(l) for l in open('$G/m/$R/evals.jsonl')];print(max(r,key=lambda x:x['add_auc'])['step'])")
  echo "$BEST" > $G/m/$R/best_checkpoint.txt
  if grep -qw "$EP" <<< "1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38"; then
    read -r CP CT < <($PY /usr1/e2s/retarget_env/sub/cam_for_pkg.py $EP 0.5 0.25)
    N=$(awk -F, -v e=$EP '$1==e{print $3}' /home/rborah/v2d/share/ego2sim_eval_recon_v1/quality.csv)
  else
    read -r CP CT < <($PY /usr1/e2s/retarget_env/cam_for.py $EP 0.45 0.22)
    N=$($PY -c "import numpy as np;print(np.load('/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/episodes/episode_%06d/hand_trajectories.npz'%$EP)['time_s'].shape[0])")
  fi
  N=$((N - 1)); C="'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' viewer.num_frames=$N"
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP rs0_$EP -- "python /env/gmd/render_m.py evaluation.checkpoint=$(python3 episodes.py ckpt $EP) viewer.output_path=/out/full/videos/ep${EP}_S0.mp4 $C $V" > $G/logs/render_S0_$EP.log 2>&1
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP rmg_$EP -- "GMD_G=/out/full/g/g9.npz python /env/gmd/render_m.py evaluation.checkpoint=/out/full/m/$R/run/policy_$BEST.safetensors viewer.output_path=/out/full/videos/ep${EP}_MG.mp4 $C $V" > $G/logs/render_MG_$EP.log 2>&1
  local S0A MGA F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
  S0A=$($PY -c "import json;print('%.3f'%json.load(open('$G/summary_s0.json'))['$EP'])" 2>/dev/null || echo "?")
  MGA=$($PY -c "import json;r=[json.loads(l) for l in open('$G/m/$R/evals.jsonl')];b=max(r,key=lambda x:x['add_auc']);print('%.3f at %.1fM'%(b['add_auc'],b['step']/1e6))")
  ffmpeg -loglevel error -y -i $G/videos/ep${EP}_S0.mp4 -i $G/videos/ep${EP}_MG.mp4 -filter_complex \
    "[0]scale=640:360,drawtext=fontfile=$F:text='ep$EP S0 policy  AUC $S0A':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[a];[1]scale=640:360,drawtext=fontfile=$F:text='${R%%_*} M + frozen G-9  AUC $MGA':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[b];[a][b]hstack=2" \
    -c:v libx264 -crf 20 -pix_fmt yuv420p $G/videos/ep${EP}_M_vs_S0.mp4
}
$PY summarize_full.py s0only >> $G/logs/finish.log 2>&1
(for r in B_ep38 B_ep37 C_ep28 A_ep41; do render 0 $r; done) & (for r in B_ep6 C_ep19 C_ep5 A_ep26; do render 1 $r; done); wait
$PY summarize_full.py >> $G/logs/finish.log 2>&1
echo "finished $(date -u)" >> $G/logs/finish.log
