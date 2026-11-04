#!/usr/bin/env bash
# Run a command in the FlashCHORD container for one teacher episode (mini decoupling experiment).
#   run_fc.sh <gpu> <episode> <name> -- <command...>
# Mounts: /fc, /env read-only; /ckpt (training_s0) read-only; /out = outputs/gmd (read-write). Memory capped.
# RECON (evaluation episodes): v1 (default, Rian's cleaned v1), v2recon (Ego2Sim v2), v2clean or v2clean2 (the 2026-11-02
# rebuild): reference package + S0 features.
set -euo pipefail
GPU=$1; EP=$2; NAME=$3; shift 4
O=/usr1/e2s/v2d_challenge_data/outputs; R=$O/rl_train_req
case " 1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38 " in
  *" $EP "*) case ${RECON:-$(cd /usr1/e2s/retarget_env/gmd && python3 episodes.py recon $EP 2>/dev/null || echo v1)} in                                                     # evaluation episode
               v2recon) FEAT=$R/s0_features_v2; PKG=$O/ego2sim_recon_v2_pkg ;;
               v2clean) FEAT=$R/s0_features_v2clean; PKG=$O/ego2sim_recon_v2clean_pkg ;;
               v2clean2) FEAT=$R/s0_features_v2clean2; PKG=$O/ego2sim_recon_v2clean2_pkg ;;
               *) FEAT=$R/s0_features_eval; PKG=$O/rian_eval_compare/_cleaned/package ;;
             esac ;;
  *) FEAT=$R/s0_features; PKG="" ;;                                                    # public episode
esac
exec docker run --rm --name "gmd_${NAME}" --memory=${MEM:-20g} --memory-swap=${MEM:-20g} --runtime=nvidia --gpus "device=$GPU" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics --entrypoint bash --user "$(id -u):$(id -g)" --group-add "$(getent group e2sshare | cut -d: -f3)" -e HOME=/tmp \
  -e PYGLET_HEADLESS=1 -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e PYTHONPATH=/fc/src:/env/s0:/env/gmd \
  -e FLASH_CHORD_CACHE_DIR=/cache -e WARP_CACHE_PATH=/cache/warp \
  -v /usr1/e2s/retarget_env/flashchord_cache:/cache -v /usr1/e2s/retarget_env:/env:ro \
  -v /usr1/e2s/video_to_data/robotic_grounding/flash_chord:/fc:ro \
  -v $R/assets:/v2d/assets:ro -v $R/assets:/data/outputs/rl_train_req/assets:ro -v "$FEAT":/feat:ro ${PKG:+-v $PKG:/pkg:ro} \
  -v $R/training_s0:/ckpt:ro -v $O/gmd:/out \
  flashchord-cuda12-575-test:193382a1-20261018 -c "cd /tmp && $*"
