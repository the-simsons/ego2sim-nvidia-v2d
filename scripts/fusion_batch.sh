#!/usr/bin/env bash
# Depth-fusion meshes (tools/depth_fusion.py --vo --still) for public-episode objects, each scored with local CD-O
# against its public scan. Jobs: EP:MASK_ID:NAME. Outputs ~/v2d/ego2sim/fusion/<name>_ep<EP>.obj (+ .log).
#   fusion_batch.sh STILL_M JOB [JOB ...]     e.g. fusion_batch.sh 0.015 41:0:white_pot 40:0:mini_dust_pan
set -uo pipefail
STILL=${1:?still tolerance m}; shift
P=~/v2d/envs/v2d_foundation_pose/.venv/bin/python; K=~/v2d/envs/kit/.venv/bin/python
KIT=~/v2d/submission_kit/v2d_submission_kit; M=~/v2d/v2d_data/track_3/public/mesh; F=~/v2d/ego2sim/fusion
mkdir -p $F; cd /tmp
for job in "$@"; do
  IFS=: read -r ep id n <<< "$job"
  ref=$(ls $M/$n/$n.glb $M/$n/${n}_visual.glb 2>/dev/null | head -1)
  out=$F/${n}_ep$ep.obj
  ( OMP_NUM_THREADS=4 $P -I ~/v2d/tools/depth_fusion.py ~/v2d/ego2sim/ep$ep $id $out --vo --still $STILL 2>&1 | grep -v -i warn > $out.log
    if [ -f $out ] && [ $out -nt $out.log -o $out -ot $out.log ] && grep -q "verts" $out.log; then
      s=$($K -I ~/v2d/tools/local_cd_o.py $KIT $n $out $ref 2>&1 | grep "packed to 4096  " | sed -E "s/.*mean ([0-9.]+) cm.*/\1/")
    else s=fail; fi
    echo "ep$ep $n: CD-O $s | $(grep -E "still run|^/" $out.log | sed -E "s#.*/fusion/##" | cut -c1-150 | tr "\n" " ")" ) &
done
wait
