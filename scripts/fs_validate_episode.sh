#!/usr/bin/env bash
# Does FoundationStereo depth track better than SGBM? For one PUBLIC episode: FoundationStereo depth into
# epEP/stereo_fs/, FoundationPose again with that depth (same masks, scan meshes, camera, settings) into
# epEP/fp_<obj>_fs/, then both track sets against mocap with validate_world_tracks.py (same camera VO for both).
#   fs_validate_episode.sh EP OBJ1[,OBJ2...] [GPU]
set -euo pipefail
EP=${1:?episode}; OBJS=${2:?objects}; GPU=${3:-0}
source ~/v2d/env.sh
export CUDA_VISIBLE_DEVICES=$GPU
E=~/v2d/ego2sim/ep$EP; S=$E/stereo; F=$E/stereo_fs; DATA=~/v2d/v2d_data/track_3/public
RECON=~/v2d/vtd_pinned/reconstruction; HOST=~/v2d/envs/recon_host/.venv/bin/python
FPY=~/v2d/envs/v2d_foundation_pose/.venv/bin/python; FSPY=~/v2d/envs/foundation_stereo/.venv/bin/python
IFS=',' read -ra NAMES <<< "$OBJS"
scan() { for c in "$DATA/mesh/$1/$1.glb" "$DATA/mesh/$1/$1_visual.glb"; do [ -f "$c" ] && { echo "$c"; return 0; }; done; }
t0=$SECONDS; lap() { echo "ep$EP [$((SECONDS - t0))s] $*"; }

[ -f "$F/stereo_stats.json" ] || $FSPY ~/v2d/tools/ego_stereo_depth.py "$DATA/meta/camera_calibration.json" \
  ~/v2d/tests_recon/stereo/extr_ep12.json "$DATA/videos/chunk-000" "$EP" "$F" --method fs 2>&1 | grep -v -i -E "warn|autocast" | tail -1
rm -f "$F/cam_a_undist.mp4"  # identical to stereo/cam_a_undist.mp4
lap "FoundationStereo depth"
cd "$RECON"
for i in "${!NAMES[@]}"; do
  n=${NAMES[$i]}; [ -d "$E/fp_${n}_fs" ] && continue
  $HOST -c "
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
run_video_to_poses('$S/cam_a_undist.mp4', '$F/stereo_depth', '$E/masks/$i', '$S/intrinsics_undist.json', '$(scan "$n")',
                   '$E/fp_${n}_fs', 'data/weights/foundation_pose', backend='nvlabs_pytorch', target_width=1014, target_height=760,
                   mask_depth=True, reregister_iou_thresh=0.3)" > "$E/fp_${n}_fs.log" 2>&1
done
lap "FoundationPose with FoundationStereo depth"
PQ=$(ls $DATA/data/*/episode_$(printf %06d "$EP").parquet)
for v in "" _fs; do
  echo "=== ep$EP tracks${v:- (SGBM)}"
  (cd /tmp && $FPY ~/v2d/tools/validate_world_tracks.py "$PQ" "$E/camera_vo.npz" \
    $(for n in "${NAMES[@]}"; do echo -n "$n=$E/fp_$n$v "; done) 2>&1 | grep -v -i warn)
done
