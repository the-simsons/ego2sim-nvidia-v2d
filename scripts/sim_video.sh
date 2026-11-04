#!/usr/bin/env bash
# Render a FlashCHORD physical replay (zero-residual reference playback, contact on) of one reference to MP4, headless.
#   sim_video.sh REF_DIR OUT.mp4 [GPU] [extra hydra overrides...]
# REF_DIR = .../processed/sequence_id=epEP/robot_name=sharpa_wave. Camera/colours as make_demo.sh (EYE / TGT env).
set -euo pipefail
REF=${1:?reference dir}; OUT=${2:?out.mp4}; GPU=${3:-0}; shift 3 || shift $#
source ~/v2d/env.sh
export CUDA_VISIBLE_DEVICES=$GPU PYGLET_HEADLESS=true
EYE=${EYE:-"0,0.95,1.38"}; TGT=${TGT:-"0,-0.05,0.86"}  # camera on the person side (+y), like the ego view
FC=~/v2d/vtd_pinned/robotic_grounding/flash_chord
N=$(~/v2d/envs/kit/.venv/bin/python -c "import pyarrow.parquet as pq,glob;print(len(pq.read_table(glob.glob('$REF/*.parquet')[0],columns=['object_body_position']).to_pylist()[0]['object_body_position']))")
TMP=$(mktemp -d); mkdir -p "$(dirname "$OUT")"
cd $FC
.venv/bin/python scripts/debug/replay.py "task.parquet='$REF'" task.motion_speed=1.0 sim.njmax=1024 sim.nconmax=512 \
  viewer.backend=mp4 "viewer.output_path='$TMP/raw.mp4'" viewer.num_frames=$N viewer.realtime=false \
  viewer.width=1280 viewer.height=720 markers.enabled=false "viewer.camera_position=[${EYE}]" "viewer.camera_target=[${TGT}]" \
  "viewer.support_color=[0.72,0.50,0.32]" "viewer.ground_color=[0.55,0.57,0.62]" "hydra.run.dir=$TMP/hydra" "$@" \
  2>&1 | grep -E "^Video|rror" || true
ffmpeg -y -loglevel error -i "$TMP/raw.mp4" -c:v libx264 -pix_fmt yuv420p -crf 22 -movflags +faststart "$OUT"
rm -rf "$TMP"
echo "-> $OUT ($N frames)"
