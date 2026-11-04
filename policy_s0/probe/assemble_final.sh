#!/usr/bin/env bash
# The 4 Nov final Track 3 package: the 3 Nov best-per-episode package with three episodes replaced by the versions that
# scored higher in single-episode swap tests (probe/swap_eval.sh), then packed with the kit's packer.
#   ep8   S0 on the v1 reference (ep8_v1_cleanedref_ray5decay, 92.3M)                   swap test +0.097 episode AUC
#   ep19  S0 with the contact-reward curriculum + mid-task resets (ep19_v2recon_contact_ray5decay, 67.1M)   +0.194
#   ep36  S0 on the v2clean reference with the spoon turned face-up (ep36_v2cleanflip_ray5decay, 67.1M)     +0.550
#   assemble_final.sh <kit python> <code commit URL> <out dir>
set -euo pipefail
PY=$1; COMMIT=$2; F=$3
O=/usr1/e2s/v2d_challenge_data/outputs; B=$O/submission_track3_best_1103; W=$O/submission_track3_swaps_1104
mkdir -p $F/results $F/meshes
cp $B/results/*.parquet $B/results/*.json $F/results/
cp -r $B/meshes/. $F/meshes/
for s in ep8_v1 ep36_flip67 ep19_contact; do
  E=$(echo $s | sed 's/ep\([0-9]*\)_.*/\1/'); P=$(printf %06d $E)
  cp $W/$s/results/episode_$P.parquet $W/$s/results/episode_$P.json $F/results/
  cp $W/$s/meshes/episode_$P/* $F/meshes/episode_$P/
done
cd /usr1/e2s/v2d_submission_kit
$PY tools/pack_submission.py --track track_3 --results $F/results --meshes $F/meshes --code-commit "$COMMIT" -o $F/track3_submission_final.parquet
