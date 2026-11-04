#!/usr/bin/env bash
# ep21 end-to-end with candidate pitcher meshes (fs at 34 cm, fs38 at 38 cm; blue_cup from meshes_final), each built as
# a reference and scored alone vs mocap, plus silhouette IoU of each pitcher track.
set -uo pipefail
source ~/v2d/env.sh
W=~/v2d/ego2sim; E=$W/ep21; K=~/v2d/envs/kit/.venv/bin/python; KIT=~/v2d/submission_kit/v2d_submission_kit
A=~/v2d/vtd_pinned/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed
L=~/v2d/logs/pitcher_swap; mkdir -p $L
for v in fs fs38; do
  M=$W/pitcher_swap_$v; mkdir -p $M
  cp $W/meshes_final/blue_cup.obj $W/meshes_final/blue_cup.glb $M/
  if [ $v = fs ]; then cp $W/meshes_fs/water_pitcher.obj $M/; else cp $W/pitcher_test/fs38.obj $M/water_pitcher.obj; fi
  ~/v2d/envs/meshclean/.venv/bin/python -c "import trimesh; trimesh.load('$M/water_pitcher.obj', force='mesh').export('$M/water_pitcher.glb')"
  # the loader refuses a different mesh under an existing asset name: park the current pitcher asset
  mkdir -p $W/assets_pitcher_parked/before_$v; mv $A/water_pitcher* $W/assets_pitcher_parked/before_$v/ 2>/dev/null
  TAG=_p$v MESH_SRC=$M DEPTH=stereo_fs FP_KW=n_particles=16 DEVICE=cuda:0 \
    bash ~/v2d/scripts/ego2sim_public_episode.sh 21 blue_cup,water_pitcher 0 > $L/ep21_$v.log 2>&1
  echo "$(date +%T) $v rc=$? $(tail -1 $L/ep21_$v.log | cut -c1-90)"
  ref=$(ls $W/train/ep21_p$v/processed/sequence_id=ep21/robot_name=sharpa_wave/*.parquet)
  $K -I ~/v2d/tools/ref_parquet_as_rollout.py $ref ~/v2d/share/ego2sim_public_eval_vs_mocap/ep13/reference_as_rollout.parquet $L/ep21_$v.parquet > /dev/null
  (cd /tmp && echo "  $v pair: $($K ~/v2d/tools/score_public_episode.py $KIT 21 $L/ep21_$v.parquet --objects blue_cup,water_pitcher \
     --mesh blue_cup=$M/blue_cup.glb --mesh water_pitcher=$M/water_pitcher.glb --axial blue_cup 2>&1 | awk '/^  [A-Z]/ {printf "%s %s  ", $1, $2}')"
   echo "  $v pitcher alone: $($K ~/v2d/tools/score_public_episode.py $KIT 21 $L/ep21_$v.parquet --objects water_pitcher \
     --mesh water_pitcher=$M/water_pitcher.glb 2>&1 | awk '/^  [A-Z]/ {printf "%s %s  ", $1, $2}')")
done
cd /tmp && ~/v2d/envs/v2d_foundation_pose/.venv/bin/python -I ~/v2d/tools/fp_track_diag.py $E/stereo/intrinsics_undist.json $E/masks/1 \
  scan=$E/fp_water_pitcher_v2s=$W/assets_public_scan/water_pitcher.obj \
  fs=$E/fp_water_pitcher_pfs=$W/pitcher_swap_fs/water_pitcher.obj fs38=$E/fp_water_pitcher_pfs38=$W/pitcher_swap_fs38/water_pitcher.obj \
  --every 30 2>&1 | grep -v -i warn
