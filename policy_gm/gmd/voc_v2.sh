#!/usr/bin/env bash
# Feedforward-VOC physical replays of the GMD run-2 episodes (each on its run-2 reconstruction), real time.
# Out: outputs/gmd/v2/voc/ep<N>_voc.mp4 (+ .json per-frame errors, .log)
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v2
O=/usr1/e2s/v2d_challenge_data/outputs; D=$O/gmd/v2/voc; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python; mkdir -p $D
eval "$(sed -n '/^hostpq()/,/^}/p' v2_pipeline.sh)"
voc() {  # <gpu> <ep>
  local GPU=$1 EP=$2 CP CT NF
  [[ -f $D/ep${EP}_voc.mp4 ]] && return
  read -r CP CT NF < <($PY cam_parquet.py $(hostpq $EP) 0.5 0.25)
  MEM=8g timeout 900 ./run_fc.sh $GPU $EP vocv2_$EP -- "VOC_MODE=ff REPLAY_JSON=/out/v2/voc/ep${EP}_voc.json python /env/voc/replay_voc.py \
    \"task.parquet='$(python3 episodes.py parquet $EP)'\" task.motion_speed=1.0 markers.contacts=false markers.keypoints=false viewer.backend=mp4 \
    viewer.output_path=/out/v2/voc/ep${EP}_voc.mp4 viewer.num_frames=$(( (NF - 1) * 5 / 2 )) viewer.video_fps=50 viewer.width=1280 viewer.height=720 \
    viewer.headless=true viewer.realtime=false 'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' hydra.run.dir=/tmp/hydra" > $D/ep${EP}_voc.log 2>&1
}
(for e in 41 3 6 30 28; do voc 0 $e; done) & (for e in 26 37 15 19; do voc 1 $e; done); wait
echo "done $(date -u)" >> $D/done.log
