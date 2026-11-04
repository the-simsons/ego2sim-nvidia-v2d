#!/usr/bin/env bash
# Rerun the GPU1 S0 jobs of start_v2clean2_queue.sh at the recipe's 96.8M steps (the first attempt inherited gmd's TOTAL=81.8M).
#   rerun_v2clean2_gpu1.sh <jobfile>     Run as: setsid nohup sg e2sshare -c "./rerun_v2clean2_gpu1.sh <jobfile>"
unset TOTAL SUB GMD_SET GMD_VOC BAND_AFTER NOBEST_N
export PKG_DIR=/usr1/e2s/v2d_challenge_data/outputs/ego2sim_recon_v2clean2_pkg FEAT_DIR=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/s0_features_v2clean2 NO_COMPARE=1
cd /usr1/e2s/retarget_env/s0 && exec ./s0_queue.sh 1 "$1"
