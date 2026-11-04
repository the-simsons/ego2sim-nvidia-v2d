#!/usr/bin/env bash
# run_voc_replay.sh <episode> <seq_suffix> <out_stem> <gpu>   (physical replay with VOC, 1 world)
set -uo pipefail
EP=$1; SUF=$2; STEM=$3; GPU=$4; O=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/voc_replay; mkdir -p $O
read -r CP CT < <(/usr1/e2s/video_to_data/reconstruction/.venv/bin/python /usr1/e2s/retarget_env/cam_for.py $EP 0.6 0.3)
SEQ=$(printf 'episode_%06d' $EP)
docker run --rm --runtime=nvidia --gpus "device=$GPU" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics --entrypoint bash \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYGLET_HEADLESS=1 -e PYTHONPATH=/fc/src -e FLASH_CHORD_CACHE_DIR=/cache -e WARP_CACHE_PATH=/cache/warp \
  -e REPLAY_JSON=/out/$STEM.json -v /usr1/e2s/retarget_env/flashchord_cache:/cache -v /usr1/e2s/retarget_env:/env:ro \
  -v /usr1/e2s/video_to_data/robotic_grounding/flash_chord:/fc:ro -v /usr1/e2s/v2d_challenge_data/outputs/rl_train_req/assets:/v2d/assets:ro -v $O:/out \
  flashchord-cuda12-575-test:193382a1-20261018 -c "cd /tmp && python /env/replay_physical_video.py \
    \"task.parquet='/v2d/assets/human_motion_data/ego_recon/processed/sequence_id=${SEQ}${SUF}/robot_name=sharpa_wave'\" task.motion_speed=1.0 \
    markers.contacts=false markers.keypoints=false viewer.backend=mp4 viewer.output_path=/out/$STEM.mp4 viewer.num_frames=882 viewer.video_fps=50 \
    viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false 'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' \
    hydra.run.dir=/tmp/hydra 2>&1 | grep -v -i warn | tail -6" > $O/$STEM.log 2>&1
