#!/usr/bin/env bash
# Rebuild all 9 public-episode references with one pipeline variant (ego2sim_public_episode.sh, two GPU queues), then
# score each REFERENCE ALONE against mocap with the kit's metric code (ref_parquet_as_rollout.py +
# score_public_episode.py, the mesh the tracks were built with, registered to the scan like Kaggle).
#   public_chain_validate.sh TAG MESH_SRC DEPTH "FP_KW"
#   e.g. public_chain_validate.sh _v2s scan stereo_fs n_particles=16
#        public_chain_validate.sh _v2f ~/v2d/ego2sim/meshes_final stereo_fs n_particles=16
# Run ONE variant at a time (variants share per-object intermediates and assets). The loader installs each clip's mesh
# as the object asset (robotic_grounding .../human_motion_data/ego_recon/processed/<obj>.obj) and refuses a different
# mesh under the same name, so empty that folder's object assets (move them aside) before a variant with other meshes.
# Output ~/v2d/logs/chain$TAG/:
# summary.tsv (episode, AUC, SP-SR, MP-SR, RPE_cm, MPPE_cm), epEP.log (driver), epEP_score.txt.
set -uo pipefail
TAG=${1:?tag}; MESH_SRC=${2:?mesh src}; DEPTH=${3:?depth}; FP_KW=${4:-}
L=~/v2d/logs/chain$TAG; mkdir -p "$L"
K=~/v2d/envs/kit/.venv/bin/python; KIT=~/v2d/submission_kit/v2d_submission_kit; SC=~/v2d/v2d_data/track_3/public/mesh
TEMPLATE=~/v2d/share/ego2sim_public_eval_vs_mocap/ep13/reference_as_rollout.parquet
Q0="2:white_pot_lid,white_pot 13:blue_cup,plastic_dish_rack 21:blue_cup,water_pitcher 40:mini_dust_pan 26:plastic_dish_rack,white_pot"
Q1="11:blue_cup,beige_cup 18:mini_sweeper,mini_dust_pan 23:wooden_piece_1,wooden_piece_2 41:white_pot"
run_queue() {  # GPU ITEMS...
  local g=$1; shift
  for x in "$@"; do
    EP=${x%%:*}
    TAG=$TAG MESH_SRC=$MESH_SRC DEPTH=$DEPTH FP_KW=$FP_KW DEVICE=cuda:0 \
      bash ~/v2d/scripts/ego2sim_public_episode.sh "$EP" "${x#*:}" "$g" > "$L/ep$EP.log" 2>&1
    echo "$(date +%T) ep$EP rc=$? $(tail -1 "$L/ep$EP.log" | cut -c1-100)"
  done
}
run_queue 0 $Q0 & run_queue 1 $Q1 & wait
# retry (one queue) episodes without a reference: the two queues can race installing the same object asset
for x in $Q0 $Q1; do
  ls ~/v2d/ego2sim/train/ep${x%%:*}$TAG/processed/sequence_id=ep${x%%:*}/robot_name=sharpa_wave/*.parquet > /dev/null 2>&1 \
    || run_queue 0 "$x"
done
mesh() {
  if [ "$MESH_SRC" = scan ]; then for c in "$SC/$1/$1.glb" "$SC/$1/$1_visual.glb"; do [ -f "$c" ] && { echo "$c"; return; }; done
  else for c in "$MESH_SRC/$1.glb" "$MESH_SRC/$1.obj"; do [ -f "$c" ] && { echo "$c"; return; }; done; fi; }
printf "episode\tAUC\tSP-SR\tMP-SR\tRPE_cm\tMPPE_cm\n" > "$L/summary.tsv"
cd /tmp
for x in $Q0 $Q1; do
  EP=${x%%:*}; IFS=, read -ra NS <<< "${x#*:}"
  ref=$(ls ~/v2d/ego2sim/train/ep$EP$TAG/processed/sequence_id=ep$EP/robot_name=sharpa_wave/*.parquet 2>/dev/null | head -1)
  [ -n "$ref" ] || { printf "%s\tFAILED\n" "$EP" >> "$L/summary.tsv"; continue; }
  $K -I ~/v2d/tools/ref_parquet_as_rollout.py "$ref" "$TEMPLATE" "$L/ep${EP}_ref.parquet" > /dev/null
  args=(); for n in "${NS[@]}"; do args+=(--mesh "$n=$(mesh "$n")")
    case $n in *cup) args+=(--axial "$n");; white_pot_lid) args+=(--order2 "$n");; esac; done
  roster=$(printf "%s\n" "${NS[@]}" | sort | paste -sd,)
  $K ~/v2d/tools/score_public_episode.py "$KIT" "$EP" "$L/ep${EP}_ref.parquet" --objects "$roster" "${args[@]}" \
    > "$L/ep${EP}_score.txt" 2>&1
  printf "%s\t%s\n" "$EP" "$(awk '/^  [A-Z]/ {printf "%s\t", $2}' "$L/ep${EP}_score.txt" | sed 's/\t$//')" >> "$L/summary.tsv"
done
column -t "$L/summary.tsv"
