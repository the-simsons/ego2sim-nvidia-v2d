#!/usr/bin/env bash
# VOC physical replay (replay_physical_video.py: objects held on the reconstruction by the VOC, hands PD-driven by the reference)
# of every GMD episode, real time. Out: outputs/gmd/full/debug/voc/ep<N>_voc.mp4 (+ .json per-frame errors).
cd /usr1/e2s/retarget_env/gmd
O=/usr1/e2s/v2d_challenge_data/outputs/gmd/full/debug/voc; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python; mkdir -p $O
voc() {  # <gpu> <ep>
  local GPU=$1 EP=$2 CP CT NF
  [[ -f $O/ep${EP}_voc.mp4 ]] && return
  if grep -qw "$EP" <<< "1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38"; then
    read -r CP CT < <($PY /usr1/e2s/retarget_env/sub/cam_for_pkg.py $EP 0.5 0.25)
    NF=$(awk -F, -v e=$EP '$1==e{print $3}' /home/rborah/v2d/share/ego2sim_eval_recon_v1/quality.csv)
  else
    read -r CP CT < <($PY /usr1/e2s/retarget_env/cam_for.py $EP 0.45 0.22)
    NF=$($PY -c "import numpy as np;print(np.load('/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/episodes/episode_%06d/hand_trajectories.npz'%$EP)['time_s'].shape[0])")
  fi
  MEM=10g timeout 900 ./run_fc.sh $GPU $EP voc_$EP -- "REPLAY_JSON=/out/full/debug/voc/ep${EP}_voc.json python /env/replay_physical_video.py \
    \"task.parquet='$(python3 episodes.py parquet $EP)'\" task.motion_speed=1.0 markers.contacts=false markers.keypoints=false viewer.backend=mp4 \
    viewer.output_path=/out/full/debug/voc/ep${EP}_voc.mp4 viewer.num_frames=$(( (NF - 1) * 5 / 2 )) viewer.video_fps=50 viewer.width=1280 viewer.height=720 \
    viewer.headless=true viewer.realtime=false 'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' hydra.run.dir=/tmp/hydra" > $O/ep${EP}_voc.log 2>&1
}
(for e in 41 26 2 8 36 6 28 5; do voc 0 $e; done) & (for e in 40 18 35 3 38 37 19; do voc 1 $e; done); wait
echo "voc done $(date -u)" >> $O/done.log
