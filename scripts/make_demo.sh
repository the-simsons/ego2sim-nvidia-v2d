#!/usr/bin/env bash
# Demo video for one episode: ego overlay | 3D reconstruction | robot policy rollout -> ~/v2d/videos/demo_epEP.mp4
#   make_demo.sh EP RUN_DIR CKPT_STEP "TITLE" "SUBTITLE" [GPU]
set -euo pipefail
EP=$1; RUN=$2; STEP=$3; TITLE=$4; SUB=$5; GPU=${6:-1}
source ~/v2d/env.sh
export CUDA_VISIBLE_DEVICES=$GPU PYGLET_HEADLESS=true
V=~/v2d/videos/ep$EP; mkdir -p $V; E=~/v2d/ego2sim/ep$EP; FC=~/v2d/vtd_pinned/robotic_grounding/flash_chord
P=$(ls ~/v2d/ego2sim/train/ep$EP/processed/sequence_id=ep$EP/robot_name=sharpa_wave/*.parquet)
N=$(~/v2d/envs/kit/.venv/bin/python -c "import pyarrow.parquet as pq;print(len(pq.read_table('$P',columns=['object_body_position']).to_pylist()[0]['object_body_position']))")
EYE=${EYE:-"0,0.95,1.38"}; TGT=${TGT:-"0,-0.05,0.86"}  # camera on the person side (+y), like the ego view
cd /tmp
~/v2d/envs/v2d_sam2/.venv/bin/python ~/v2d/tools/ego_overlay.py $E $V/ego.mp4
~/v2d/envs/v2d_hamer/.venv/bin/python ~/v2d/tools/mano_verts.py $P $V/mano.npz 2>&1 | grep -v -i warn
$FC/.venv/bin/python ~/v2d/tools/render_recon_gl.py $P $V/mano.npz $V/recon.mp4 --eye $EYE --target $TGT 2>&1 | grep -E "^Video|rror"
$FC/.venv/bin/python $FC/scripts/view_policy.py evaluation.checkpoint=$RUN/policy_$STEP.safetensors scene.world_count=1 \
  viewer.backend=mp4 viewer.output_path=$V/sim.mp4 viewer.num_frames=$N viewer.realtime=false viewer.width=1280 viewer.height=720 \
  markers.enabled=false "viewer.camera_position=[${EYE}]" "viewer.camera_target=[${TGT}]" \
  "viewer.support_color=[0.72,0.50,0.32]" "viewer.ground_color=[0.55,0.57,0.62]" "hydra.run.dir=$V/hydra" 2>&1 | grep -E "^Video|rror"
~/v2d/envs/v2d_sam2/.venv/bin/python ~/v2d/tools/compose_demo.py $V/ego.mp4 $V/recon.mp4 $V/sim.mp4 $V/demo.mp4 --title "$TITLE" --subtitle "$SUB"
ffmpeg -y -loglevel error -i $V/demo.mp4.tmp.mp4 -c:v libx264 -pix_fmt yuv420p -crf 22 -movflags +faststart ~/v2d/videos/demo_ep$EP.mp4 \
  && rm $V/demo.mp4.tmp.mp4 && echo "-> ~/v2d/videos/demo_ep$EP.mp4 ($N frames)"