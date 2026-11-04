#!/usr/bin/env bash
# Track the ep21 pitcher with candidate meshes (FS depth, pf16), then per-frame silhouette IoU vs the SAM2 mask.
set -uo pipefail
source ~/v2d/env.sh
E=~/v2d/ego2sim/ep21; S=$E/stereo; W=~/v2d/ego2sim; O=$W/pitcher_test; mkdir -p $O
HOST=~/v2d/envs/recon_host/.venv/bin/python; P=~/v2d/envs/v2d_foundation_pose/.venv/bin/python
cd /tmp
$P -c "
import trimesh
m = trimesh.load('$W/meshes_final/water_pitcher.obj', force='mesh'); m.apply_scale([-1, 1, 1]); m.invert()
m.export('$O/mirror_x.obj')
" 2>&1 | grep -v -i warn
run() {  # label mesh gpu
  [ -d $O/fp_$1 ] && return
  (cd ~/v2d/vtd_pinned/reconstruction && CUDA_VISIBLE_DEVICES=$3 $HOST -c "
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
run_video_to_poses('$S/cam_a_undist.mp4', '$E/stereo_fs/stereo_depth', '$E/masks/1', '$S/intrinsics_undist.json', '$2',
                   '$O/fp_$1', 'data/weights/foundation_pose', backend='nvlabs_pytorch', target_width=1014, target_height=760,
                   mask_depth=True, reregister_iou_thresh=0.3, n_particles=16)" > $O/fp_$1.log 2>&1)
  echo "$(date +%T) $1 done ($(ls $O/fp_$1 | wc -l) poses)"
}
( run v1 $W/meshes/water_pitcher.obj 0; run mirror $O/mirror_x.obj 0 ) &
( run fs $W/meshes_fs/water_pitcher.obj 1; run v2 $W/meshes_v2/water_pitcher.obj 1 ) &
wait
$P -I ~/v2d/tools/fp_track_diag.py $S/intrinsics_undist.json $E/masks/1 \
  scan=$E/fp_water_pitcher_v2s=$W/assets_public_scan/water_pitcher.obj \
  final=$E/fp_water_pitcher_v2f=$W/meshes_final/water_pitcher.obj \
  v2=$O/fp_v2=$W/meshes_v2/water_pitcher.obj v1=$O/fp_v1=$W/meshes/water_pitcher.obj \
  fs=$O/fp_fs=$W/meshes_fs/water_pitcher.obj mirror=$O/fp_mirror=$O/mirror_x.obj --every 30 2>&1 | grep -v -i warn
