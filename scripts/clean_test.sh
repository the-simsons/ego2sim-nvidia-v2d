#!/usr/bin/env bash
# Clean every final mesh (union / visible), then score each variant on the local CD-O board.
P=~/v2d/envs/meshclean/.venv/bin/python; K=~/v2d/envs/kit/.venv/bin/python; KIT=~/v2d/submission_kit/v2d_submission_kit
F=~/v2d/ego2sim/meshes_final; cd /tmp
for mode in union visible; do
  O=~/v2d/ego2sim/meshes_clean_$mode; mkdir -p $O
  for f in $F/*.obj; do
    n=$(basename $f .obj)
    $P ~/v2d/tools/mesh_clean.py $f $O/$n.obj --mode $mode 2>&1 | grep -v -i warn
    $P -c "import trimesh; trimesh.load('$O/$n.obj', force='mesh').export('$O/$n.glb')"
  done
  echo "== board $mode"
  OMP_NUM_THREADS=8 $K -I ~/v2d/tools/cdo_board.py $KIT $O --json ~/v2d/logs/cdo_board/clean_$mode.json 2>&1 | grep -v -i warn
done
