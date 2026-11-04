#!/usr/bin/env bash
# Refinement Pass 0: FoundationPose again with its own tracking options (more refiner iterations / particle filter),
# FoundationStereo depth, scan meshes, same masks/camera/settings as fs_validate_episode.sh, into epEP/fp_<obj>_fs_<VAR>/;
# then every variant vs mocap with validate_world_tracks.py (same camera VO).
#   fp_pass0.sh EP OBJ1[,OBJ2...] GPU VAR=KWARGS [VAR=KWARGS ...]
#   e.g. fp_pass0.sh 21 blue_cup,water_pitcher 1 "it8=track_iteration=8" "pf16=n_particles=16"
# Object order must match the mask ids (masks/0, masks/1, ...).
set -euo pipefail
EP=${1:?episode}; OBJS=${2:?objects}; GPU=${3:?gpu}; shift 3
source ~/v2d/env.sh
export CUDA_VISIBLE_DEVICES=$GPU
E=~/v2d/ego2sim/ep$EP; S=$E/stereo; F=$E/stereo_fs; DATA=~/v2d/v2d_data/track_3/public
RECON=~/v2d/vtd_pinned/reconstruction; HOST=~/v2d/envs/recon_host/.venv/bin/python
FPY=~/v2d/envs/v2d_foundation_pose/.venv/bin/python
IFS=',' read -ra NAMES <<< "$OBJS"
scan() { for c in "$DATA/mesh/$1/$1.glb" "$DATA/mesh/$1/$1_visual.glb"; do [ -f "$c" ] && { echo "$c"; return 0; }; done; }
cd "$RECON"
VARS=()
for spec in "$@"; do
  var=${spec%%=*}; kw=${spec#*=}; VARS+=("$var")
  for i in "${!NAMES[@]}"; do
    n=${NAMES[$i]}; out=$E/fp_${n}_fs_$var; [ -f "$out/done" ] && continue
    rm -rf "$out"; t0=$SECONDS
    $HOST -c "
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
run_video_to_poses('$S/cam_a_undist.mp4', '$F/stereo_depth', '$E/masks/$i', '$S/intrinsics_undist.json', '$(scan "$n")',
                   '$out', 'data/weights/foundation_pose', backend='nvlabs_pytorch', target_width=1014, target_height=760,
                   mask_depth=True, reregister_iou_thresh=0.3, $kw)" > "$out.log" 2>&1
    touch "$out/done"; echo "ep$EP $n $var ($kw): $((SECONDS - t0)) s"
  done
done
PQ=$(ls $DATA/data/*/episode_$(printf %06d "$EP").parquet)
for v in "" "${VARS[@]/#/_}"; do
  echo "=== ep$EP tracks _fs$v"
  (cd /tmp && $FPY ~/v2d/tools/validate_world_tracks.py "$PQ" "$E/camera_vo.npz" \
    $(for n in "${NAMES[@]}"; do echo -n "$n=$E/fp_${n}_fs$v "; done) 2>&1 | grep -v -i warn | grep "initial-scene\|travel")
done
