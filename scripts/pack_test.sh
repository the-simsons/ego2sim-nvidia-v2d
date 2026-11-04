#!/usr/bin/env bash
# Packing loss: score our own simplifications of the over-budget meshes (and their union-cleaned versions) vs the packer.
P=~/v2d/envs/meshclean/.venv/bin/python; K=~/v2d/envs/kit/.venv/bin/python; KIT=~/v2d/submission_kit/v2d_submission_kit
F=~/v2d/ego2sim/meshes_final; U=~/v2d/ego2sim/meshes_clean_union; O=~/v2d/ego2sim/presimp; mkdir -p $O; cd /tmp
for src in $F/plastic_dish_rack.obj $F/white_pot_lid.obj $U/white_pot_lid.obj $F/wooden_piece_1.obj $U/plastic_dish_rack.obj; do
  tag=$(basename $(dirname $src)); n=$(basename $src .obj); d=$O/$tag; echo "== $tag/$n"
  $P ~/v2d/tools/presimplify.py $src $d 2>&1 | grep -v -i warn
  for v in pm_qec pm_qec_q pm_remesh fs_agg3; do
    echo "  $v: $(OMP_NUM_THREADS=8 $K -I ~/v2d/tools/cdo_board.py $KIT $F --only $n --set $n=$d/${n}_$v.obj 2>&1 | grep "^$n" | awk '{print $3}')"
  done
done
