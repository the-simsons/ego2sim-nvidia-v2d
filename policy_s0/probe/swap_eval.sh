#!/usr/bin/env bash
# Official kit evaluation of one episode's S0 checkpoint on a chosen reference version (for single-episode swaps of the
# 3 Nov package): results/episode_XXXXXX.{parquet,json} + meshes/episode_XXXXXX/<object>.glb from that version's package.
#   swap_eval.sh <episode> <v1|v2raw|v2clean|v2clean2> <run dir under training_s0/eval> <phase> <step> <gpu> <out dir>
#   env PKG_OVERRIDE / FEAT_OVERRIDE: use another reference package / extras folder than the version's own
set -uo pipefail
EP=$1; VER=$2; RUN=$3; PH=$4; STEP=$5; GPU=$6; OUT=$7
O=/usr1/e2s/v2d_challenge_data/outputs; R=$O/rl_train_req
case $VER in
  v1) PKG=$O/rian_eval_compare/_cleaned/package; FEAT=$R/s0_features_eval ;;
  v2raw) PKG=$O/ego2sim_recon_v2_pkg; FEAT=$R/s0_features_v2 ;;
  v2clean) PKG=$O/ego2sim_recon_v2clean_pkg; FEAT=$R/s0_features_v2clean ;;
  v2clean2) PKG=$O/ego2sim_recon_v2clean2_pkg; FEAT=$R/s0_features_v2clean2 ;;
  *) echo "unknown version $VER"; exit 1 ;;
esac
PKG=${PKG_OVERRIDE:-$PKG}; FEAT=${FEAT_OVERRIDE:-$FEAT}     # corrected references (e.g. probe/flip_reference.py) and their extras
mkdir -p $OUT/results $OUT/meshes
echo "{\"$EP\": \"/runs/$RUN/$PH/policy_$STEP.safetensors\"}" > $OUT/checkpoints.json
docker run --rm --name "swap_ep${EP}_$VER" --runtime=nvidia --gpus device=$GPU --entrypoint bash --user $(id -u):$(id -g) \
  --group-add "$(getent group e2sshare | cut -d: -f3)" -e HOME=/tmp -e PYTHONPATH=/fc/src:/env/s0 -e FLASH_CHORD_CACHE_DIR=/cache \
  -e WARP_CACHE_PATH=/cache/warp -v /usr1/e2s/retarget_env/flashchord_cache:/cache -v /usr1/e2s/retarget_env:/env:ro \
  -v /usr1/e2s/video_to_data/robotic_grounding/flash_chord:/fc:ro -v /usr1/e2s/v2d_submission_kit:/kit:ro -v $PKG:/pkg:ro -v $FEAT:/feat:ro \
  -v $R/training_s0/eval:/runs:ro -v $OUT:/sub flashchord-cuda12-575-test:193382a1-20261018 \
  -c "cd /tmp && python /kit/examples/run_policy_evaluation.py --checkpoints /sub/checkpoints.json --results /sub/results --runtime-root /fc" > $OUT/kit_evaluation.log 2>&1
python3 - <<PY
import json, shutil
from pathlib import Path
objs = json.load(open("/usr1/e2s/v2d_submission_kit/data/track_3_evaluation_objects.json"))["episodes"]["$EP"]
d = Path("$OUT/meshes") / f"episode_{int($EP):06d}"; d.mkdir(parents=True, exist_ok=True)
for o in objs:
    shutil.copy2(f"$PKG/meshes/{o}.glb", d / f"{o}.glb")
PY
ls $OUT/results/episode_$(printf %06d $EP).parquet > /dev/null 2>&1 && echo "ep$EP $VER: done" || echo "ep$EP $VER: FAILED (see $OUT/kit_evaluation.log)"
