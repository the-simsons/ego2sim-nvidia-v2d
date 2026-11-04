#!/usr/bin/env bash
# G-alone videos (after go_all.sh): per episode, [reference replay | reference wrists + G-9 (human engage) | S0 policy], real time.
cd /usr1/e2s/retarget_env/gmd
G=/usr1/e2s/v2d_challenge_data/outputs/gmd/full; O=$G/gonly; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
mkdir -p $O/videos
V="viewer.backend=mp4 viewer.video_fps=20 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra"
auc() { $PY -c "import json;print('%.3f'%json.load(open('$1'))['add_auc'])" 2>/dev/null || echo "?"; }
render() {  # <gpu> <ep>
  local GPU=$1 EP=$2 CP CT N CK
  if grep -qw "$EP" <<< "1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38"; then
    read -r CP CT < <($PY /usr1/e2s/retarget_env/sub/cam_for_pkg.py $EP 0.5 0.25)
    N=$(awk -F, -v e=$EP '$1==e{print $3}' /home/rborah/v2d/share/ego2sim_eval_recon_v1/quality.csv)
  else
    read -r CP CT < <($PY /usr1/e2s/retarget_env/cam_for.py $EP 0.45 0.22)
    N=$($PY -c "import numpy as np;print(np.load('/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/episodes/episode_%06d/hand_trajectories.npz'%$EP)['time_s'].shape[0])")
  fi
  N=$((N - 1)); CK=$(python3 episodes.py ckpt $EP); C="'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' viewer.num_frames=$N"
  MEM=10g timeout 900 ./run_fc.sh $GPU $EP gr_ref_$EP -- "GMD_SCRIPT=ref python /env/gmd/render_m.py evaluation.checkpoint=$CK viewer.output_path=/out/full/gonly/videos/ep${EP}_ref.mp4 $C $V" > $O/logs/render_ref_$EP.log 2>&1
  MEM=10g timeout 900 ./run_fc.sh $GPU $EP gr_g_$EP -- "GMD_G=/out/full/g/g9.npz GMD_SCRIPT=/out/full/gonly/ep$EP/engage_gt.npy python /env/gmd/render_m.py evaluation.checkpoint=$CK viewer.output_path=/out/full/gonly/videos/ep${EP}_refG.mp4 $C $V" > $O/logs/render_refG_$EP.log 2>&1
  [[ -f $G/videos/ep${EP}_S0.mp4 ]] || MEM=10g timeout 900 ./run_fc.sh $GPU $EP gr_s0_$EP -- "python /env/gmd/render_m.py evaluation.checkpoint=$CK viewer.output_path=/out/full/videos/ep${EP}_S0.mp4 $C $V" > $O/logs/render_S0_$EP.log 2>&1
  local F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf S0A
  S0A=$($PY -c "import json;print('%.3f'%json.load(open('$G/summary_s0.json'))['$EP'])" 2>/dev/null || $PY -c "import json;print('%.3f'%json.load(open('$G/ep$EP/nominal.json'))['add_auc'])" 2>/dev/null || echo "?")
  lab() { echo "[$1]scale=640:360,drawtext=fontfile=$F:text='$2':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[$3]"; }
  ffmpeg -loglevel error -y -i $O/videos/ep${EP}_ref.mp4 -i $O/videos/ep${EP}_refG.mp4 -i $G/videos/ep${EP}_S0.mp4 -filter_complex \
    "$(lab 0 "ep$EP human replay (tracked)  AUC $(auc $O/ep$EP/track_ref.json)" a);$(lab 1 "human wrists + G-9  AUC $(auc $O/ep$EP/track_g_gt.json)" b);$(lab 2 "S0 policy  AUC $S0A" c);[a][b][c]hstack=3" \
    -c:v libx264 -crf 20 -pix_fmt yuv420p $O/videos/ep${EP}_ref_vs_G_vs_S0.mp4
}
(for e in 5 28 38 37 41; do render 0 $e; done) & (for e in 19 6 36 26 35; do render 1 $e; done); wait
echo "renders done $(date -u)" >> $O/logs/go_all.log
