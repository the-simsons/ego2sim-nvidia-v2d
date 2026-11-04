#!/usr/bin/env bash
# Refinement Passes 1-2 (tools/track_refine.py) on one public episode's tracks, then input / Pass 1 / Passes 1+2 vs
# mocap (validate_world_tracks.py, same camera VO). Scan meshes, as the input FoundationPose run used.
#   refine_validate.sh EP OBJ1[,OBJ2...] IN_SUFFIX [EXTRA_ARGS...]   e.g. refine_validate.sh 21 blue_cup,water_pitcher _fs_pf16
# Object order must match the mask ids. Outputs epEP/fp_<obj><IN_SUFFIX>_r1/, _r2/, _refine.json.
set -euo pipefail
EP=${1:?episode}; OBJS=${2:?objects}; IN=${3:?input suffix}; shift 3
E=~/v2d/ego2sim/ep$EP; DATA=~/v2d/v2d_data/track_3/public; FPY=~/v2d/envs/v2d_foundation_pose/.venv/bin/python
IFS=',' read -ra NAMES <<< "$OBJS"
scan() { for c in "$DATA/mesh/$1/$1.glb" "$DATA/mesh/$1/$1_visual.glb"; do [ -f "$c" ] && { echo "$c"; return 0; }; done; }
cd /tmp
for i in "${!NAMES[@]}"; do
  n=${NAMES[$i]}
  OMP_NUM_THREADS=2 $FPY -I ~/v2d/tools/track_refine.py "$E" "$n" "$i" "$(scan "$n")" "$E/fp_$n$IN" "$E/fp_$n$IN" "$@" \
    2>&1 | grep -v -i warn &
done
wait
PQ=$(ls $DATA/data/*/episode_$(printf %06d "$EP").parquet)
for v in "" _r1 _r2; do
  echo "=== ep$EP tracks $IN$v"
  $FPY ~/v2d/tools/validate_world_tracks.py "$PQ" "$E/camera_vo.npz" \
    $(for n in "${NAMES[@]}"; do echo -n "$n=$E/fp_$n$IN$v "; done) 2>&1 | grep -v -i warn | grep "initial-scene" | head -${#NAMES[@]}
done
