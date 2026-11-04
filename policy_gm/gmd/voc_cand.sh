#!/usr/bin/env bash
# VOC physical replays of candidate held-out / failure episodes on each available reconstruction (real time).
#   jobs: "<tag> <ep>" with tag v1 (rian cleaned), v2recon, v2clean, pub (our public recon), pubtm (teammate public recon)
# Out: outputs/gmd/full/debug/voc_cand/<tag>_ep<N>.mp4 (+ .json per-frame errors, .log)
cd /usr1/e2s/retarget_env/gmd
O=/usr1/e2s/v2d_challenge_data/outputs; D=$O/gmd/full/debug/voc_cand; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python; mkdir -p $D
job() {  # <gpu> <tag> <ep>
  local GPU=$1 TAG=$2 EP=$3 PKG="" HOSTPQ PQ CP CT NF
  [[ -f $D/${TAG}_ep$EP.mp4 ]] && return
  case $TAG in
    v1) PKG=$O/rian_eval_compare/_cleaned/package ;;
    v2recon) PKG=$O/ego2sim_recon_v2_pkg ;;
    v2clean) PKG=$O/ego2sim_recon_v2clean_pkg ;;
  esac
  if [[ -n $PKG ]]; then
    PQ=/pkg/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave; HOSTPQ=$PKG/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave
  else
    local SUF=""; [[ $TAG == pubtm ]] && SUF=_teammate
    PQ=/v2d/assets/human_motion_data/ego_recon/processed/sequence_id=$(printf 'episode_%06d' $EP)$SUF/robot_name=sharpa_wave
    HOSTPQ=$O/rl_train_req/assets/human_motion_data/ego_recon/processed/sequence_id=$(printf 'episode_%06d' $EP)$SUF/robot_name=sharpa_wave
  fi
  read -r CP CT NF < <($PY cam_parquet.py $HOSTPQ 0.5 0.25)
  docker run --rm --name gmd_vc_${TAG}_$EP --memory=10g --memory-swap=10g --runtime=nvidia --gpus "device=$GPU" \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics --entrypoint bash --user "$(id -u):$(id -g)" --group-add 24518 -e HOME=/tmp -e PYGLET_HEADLESS=1 \
    -e PYTHONPATH=/fc/src -e FLASH_CHORD_CACHE_DIR=/cache -e WARP_CACHE_PATH=/cache/warp -e REPLAY_JSON=/out/${TAG}_ep$EP.json \
    -v /usr1/e2s/retarget_env/flashchord_cache:/cache -v /usr1/e2s/retarget_env:/env:ro -v /usr1/e2s/video_to_data/robotic_grounding/flash_chord:/fc:ro \
    -v $O/rl_train_req/assets:/v2d/assets:ro ${PKG:+-v $PKG:/pkg:ro} -v $D:/out \
    flashchord-cuda12-575-test:193382a1-20261018 -c "cd /tmp && python /env/replay_physical_video.py \"task.parquet='$PQ'\" task.motion_speed=1.0 \
      markers.contacts=false markers.keypoints=false viewer.backend=mp4 viewer.output_path=/out/${TAG}_ep$EP.mp4 viewer.num_frames=$(( (NF - 1) * 5 / 2 )) \
      viewer.video_fps=50 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false \
      'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' hydra.run.dir=/tmp/hydra" > $D/${TAG}_ep$EP.log 2>&1
}
JOBS=()
for e in 1 5 6 10 14 15 16 19 20 25 28 30 32 33 37 38; do JOBS+=("v2recon $e"); done
for e in 10 14 16 32 33; do JOBS+=("v2clean $e"); done
for e in 1 10 14 15 16 20 25 30 32 33; do JOBS+=("v1 $e"); done
for e in 0 11 12 13 22 23 24 27 31 42 43 44 45; do JOBS+=("pub $e"); done
for e in 11 13 23; do JOBS+=("pubtm $e"); done
i=0; for j in "${JOBS[@]}"; do echo "$((i % 2)) $j"; i=$((i + 1)); done > $D/jobs.txt
(grep "^0 " $D/jobs.txt | while read -r g t e; do job $g $t $e; done) & (grep "^1 " $D/jobs.txt | while read -r g t e; do job $g $t $e; done); wait
echo "done $(date -u)" >> $D/done.log
