#!/usr/bin/env bash
# run_voc_replay_pkg.sh <episode> <package_dir> <out_stem> <gpu> [hydra overrides...]   (run_voc_replay.sh for an eval package at /pkg)
# Physical replay with VOC, 1 world, whole episode at 50 fps -> voc_replay_eval/<out_stem>.{mp4,json,log}
set -uo pipefail
EP=$1; PKG=$2; STEM=$3; GPU=$4; EXTRA="${*:5}"; O=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/voc_replay_eval; mkdir -p $O
read -r CP CT NF < <(/usr1/e2s/video_to_data/reconstruction/.venv/bin/python /usr1/e2s/retarget_env/cam_for_pkg_traj.py $PKG $EP 0.6 0.3)
docker run --rm --runtime=nvidia --gpus "device=$GPU" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics --entrypoint bash \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYGLET_HEADLESS=1 -e PYTHONPATH=/fc/src -e FLASH_CHORD_CACHE_DIR=/cache -e WARP_CACHE_PATH=/cache/warp \
  -e REPLAY_JSON=/out/$STEM.json -v /usr1/e2s/retarget_env/flashchord_cache:/cache -v /usr1/e2s/retarget_env:/env:ro \
  -v /usr1/e2s/video_to_data/robotic_grounding/flash_chord:/fc:ro -v $PKG:/pkg:ro -v $O:/out \
  flashchord-cuda12-575-test:193382a1-20261018 -c "cd /tmp && python /env/replay_physical_video.py \
    \"task.parquet='/pkg/sim_reference/processed/sequence_id=ep${EP}/robot_name=sharpa_wave'\" task.motion_speed=1.0 sim.njmax=1024 sim.nconmax=512 \
    markers.contacts=false markers.keypoints=false viewer.backend=mp4 viewer.output_path=/out/$STEM.mp4 viewer.num_frames=$NF viewer.video_fps=50 \
    viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false 'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' \
    hydra.run.dir=/tmp/hydra $EXTRA 2>&1 | grep -v -i warn | tail -6" > $O/$STEM.log 2>&1
