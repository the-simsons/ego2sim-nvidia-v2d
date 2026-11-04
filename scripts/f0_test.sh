#!/usr/bin/env bash
# Frame-0 sensitivity on the public episodes: A = v2s with v2f-oracle frame 0; S5/S10 = frame 0 := mean of first K steps.
K=~/v2d/envs/kit/.venv/bin/python; KIT=~/v2d/submission_kit/v2d_submission_kit; SC=~/v2d/v2d_data/track_3/public/mesh
F=~/v2d/ego2sim/meshes_final; L=~/v2d/logs/f0; mkdir -p $L; cd /tmp
sc() {  # ep rollout meshsrc names...
  local e=$1 r=$2 src=$3; shift 3; local args=() n m
  for n in "$@"; do
    if [ $src = scan ]; then m=$(ls $SC/$n/$n.glb $SC/$n/${n}_visual.glb 2>/dev/null | head -1); else m=$F/$n.glb; fi
    args+=(--mesh "$n=$m"); case $n in *cup) args+=(--axial $n);; white_pot_lid) args+=(--order2 $n);; esac
  done
  $K ~/v2d/tools/score_public_episode.py $KIT $e $r --objects $(printf "%s\n" "$@" | sort | paste -sd,) "${args[@]}" 2>&1 \
    | awk '/^  (AUC|MPPE)/ {printf "%s %s ", $1, $2}'
}
for x in 2:white_pot_lid,white_pot 11:blue_cup,beige_cup 13:blue_cup,plastic_dish_rack 18:mini_sweeper,mini_dust_pan \
         21:blue_cup,water_pitcher 26:plastic_dish_rack,white_pot 40:mini_dust_pan 41:white_pot; do
  e=${x%%:*}; IFS=, read -ra NS <<< "${x#*:}"
  S=~/v2d/logs/chain_v2s/ep${e}_ref.parquet; V=~/v2d/logs/chain_v2f/ep${e}_ref.parquet
  [ $e = 21 ] && V=~/v2d/logs/pitcher_swap/ep21_fs38.parquet  # meshes_final pitcher is now fs38 (Phase 8c)
  line="ep$e"
  if [ -f ~/v2d/logs/oracle_reg/ep$e.parquet ]; then
    $K -I ~/v2d/tools/frame0_variant.py $S $L/ep${e}_A.parquet --from ~/v2d/logs/oracle_reg/ep$e.parquet
    line+=" | scan+oracleF0: $(sc $e $L/ep${e}_A.parquet scan "${NS[@]}")"
  fi
  for k in 5 10; do
    $K -I ~/v2d/tools/frame0_variant.py $S $L/ep${e}_s$k.parquet --static $k
    $K -I ~/v2d/tools/frame0_variant.py $V $L/ep${e}_f$k.parquet --static $k
    line+=" | scan S$k: $(sc $e $L/ep${e}_s$k.parquet scan "${NS[@]}") | final S$k: $(sc $e $L/ep${e}_f$k.parquet final "${NS[@]}")"
  done
  echo "$line"
done
