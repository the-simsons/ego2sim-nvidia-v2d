#!/usr/bin/env bash
# Run a command in the FlashCHORD container for one episode (gmd_ft). Copy of retarget_env/gmd/run_fc.sh with these mounts:
#   /ft = gmd_ft (read-only), /gmd = outputs/gmd (read-only: v4 checkpoints, G), /out = outputs/gmd_ft (read-write),
#   /env, /fc, /ckpt, assets, features and the reconstruction package as in run_fc.sh. Containers are named ft_<name>.
#   ft_fc.sh <gpu> <episode> <name> -- <command...>      env: MEM (default 20g), GMD_SET (episode table, see gmd/episodes.py)
set -euo pipefail
GPU=$1; EP=$2; NAME=$3; shift 4
O=/usr1/e2s/v2d_challenge_data/outputs; R=$O/rl_train_req
case " 1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38 " in
  *" $EP "*) case ${RECON:-$(cd /usr1/e2s/retarget_env/gmd && python3 episodes.py recon $EP 2>/dev/null || echo v1)} in
               v2recon) FEAT=$R/s0_features_v2; PKG=$O/ego2sim_recon_v2_pkg ;;
               v2clean) FEAT=$R/s0_features_v2clean; PKG=$O/ego2sim_recon_v2clean_pkg ;;
               *) FEAT=$R/s0_features_eval; PKG=$O/rian_eval_compare/_cleaned/package ;;
             esac ;;
  *) FEAT=$R/s0_features; PKG="" ;;
esac
exec docker run --rm --name "ft_${NAME}" --memory=${MEM:-20g} --memory-swap=${MEM:-20g} --runtime=nvidia --gpus "device=$GPU" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics --entrypoint bash --user "$(id -u):$(id -g)" --group-add "$(getent group e2sshare | cut -d: -f3)" -e HOME=/tmp \
  -e PYGLET_HEADLESS=1 -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e PYTHONPATH=/fc/src:/env/s0:/env/gmd:/ft \
  -e FLASH_CHORD_CACHE_DIR=/cache -e WARP_CACHE_PATH=/cache/warp \
  -v /usr1/e2s/retarget_env/flashchord_cache:/cache -v /usr1/e2s/retarget_env:/env:ro -v /usr1/e2s/gmd_ft:/ft:ro \
  -v /usr1/e2s/video_to_data/robotic_grounding/flash_chord:/fc:ro \
  -v $R/assets:/v2d/assets:ro -v $R/assets:/data/outputs/rl_train_req/assets:ro -v "$FEAT":/feat:ro ${PKG:+-v $PKG:/pkg:ro} \
  -v $R/training_s0:/ckpt:ro -v $O/gmd:/gmd:ro -v $O/gmd_ft:/out \
  flashchord-cuda12-575-test:193382a1-20261018 -c "cd /tmp && $*"
