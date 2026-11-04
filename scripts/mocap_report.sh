#!/usr/bin/env bash
# Where does the error come from? For each public episode, score against motion capture (kit metric code,
# tools/score_public_episode.py) both the RECONSTRUCTED REFERENCE alone and every training run's selected checkpoint
# (lowest MPPE vs our reference, from eval_table.txt).
#   mocap_report.sh OUT_DIR
# OUT_DIR/summary.tsv              one row per (episode, source): AUC SP-SR MP-SR RPE MPPE vs mocap
# OUT_DIR/epEP/<source>.txt        full scorer output per row
# OUT_DIR/epEP/ref_vs_mocap.txt    per-object motion of reference vs mocap (tools/compare_ref_mocap.py)
set -uo pipefail
OUT=${1:?out dir}; mkdir -p "$OUT"
PY=~/v2d/envs/kit/.venv/bin/python; KIT=~/v2d/submission_kit/v2d_submission_kit; SC=~/v2d/v2d_data/track_3/public/mesh
printf "episode\tsource\tcheckpoint\tAUC\tSP-SR\tMP-SR\tRPE_cm\tMPPE_cm\n" > "$OUT/summary.tsv"
score() {  # EP OBJS PARQUET -> fills $S (tab-separated five metrics), full output to $LOG
  local meshes=() sym=() n roster
  IFS=, read -ra NS <<< "$2"
  for n in "${NS[@]}"; do local m=$SC/$n/$n.glb; [ -f "$m" ] || m=$SC/$n/${n}_visual.glb; meshes+=(--mesh "$n=$m")
    case $n in *cup) sym+=(--axial "$n");; white_pot_lid) sym+=(--order2 "$n");; esac; done
  roster=$(printf "%s\n" "${NS[@]}" | sort | paste -sd,)
  (cd /tmp && $PY ~/v2d/tools/score_public_episode.py "$KIT" "$1" "$3" --objects "$roster" "${meshes[@]}" "${sym[@]}") > "$LOG" 2>&1
  S=$(awk '/^  [A-Z]/ {printf "%s\t", $2}' "$LOG" | sed 's/\t$//')
}
for x in 2:white_pot_lid,white_pot 11:blue_cup,beige_cup 13:blue_cup,plastic_dish_rack 18:mini_sweeper,mini_dust_pan \
         21:blue_cup,water_pitcher 23:wooden_piece_1,wooden_piece_2 26:plastic_dish_rack,white_pot 40:mini_dust_pan 41:white_pot; do
  EP=${x%%:*}; OBJS=${x#*:}; D="$OUT/ep$EP"; mkdir -p "$D"; first=""
  for R in ~/v2d/runs/e2s_ep${EP}_40M ~/v2d/runs/e2s_ep${EP}_40M_hf ~/v2d/runs/e2s_ep${EP}_40M_x2; do
    [ -f "$R/eval_table.txt" ] || continue
    step=$(awk 'NR>1 && $2+0>0 {print $2, $1}' "$R/eval_table.txt" | sort -n | head -1 | awk '{print $2}')
    P="$R/eval/official_$step.parquet"; [ -f "$P" ] || continue
    [ -z "$first" ] && first=$P
    LOG="$D/rollout_$(basename "$R").txt"; score "$EP" "$OBJS" "$P"
    printf "%s\trollout %s\t%s\t%s\n" "$EP" "$(basename "$R")" "$step" "$S" >> "$OUT/summary.tsv"
  done
  # reference alone: the rollout parquet with achieved := reference (tools/ref_as_rollout.py)
  $PY ~/v2d/tools/ref_as_rollout.py "$first" "$D/reference_as_rollout.parquet"
  LOG="$D/reference.txt"; score "$EP" "$OBJS" "$D/reference_as_rollout.parquet"
  printf "%s\treference only\t-\t%s\n" "$EP" "$S" >> "$OUT/summary.tsv"
  (cd /tmp && $PY ~/v2d/tools/compare_ref_mocap.py "$EP" "$D/reference_as_rollout.parquet") > "$D/ref_vs_mocap.txt" 2>&1
  echo "ep$EP done"
done
