#!/usr/bin/env bash
# Rerun on rborah's 2026-11-02 cleaned-v2 rebuild (copy: outputs/ego2sim_recon_v2clean2_pkg, features: s0_features_v2clean2):
#   training_s0 (latest recipe, tracking_96m_pickup_ray5decay) on eval episodes 1, 10, 14, 20, 25, 28; ep25 with wrist termination
#   loosened to 2.0 m (its reference right wrist still jumps up to 24 cm/frame);
#   gmd v4 (g9gated, v4_pipeline.sh settings) on ep28 -> outputs/gmd/v4_clean2/m/X_ep28.
# GPU0: S0 10, 25, 20.  GPU1: gmd ep28, then S0 14, 1, 28.
# Run as: setsid nohup sg e2sshare -c ./start_v2clean2_queue.sh   (the containers need e2sshare as the primary group)
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; S=$T/_scratch/v2clean2; cd /usr1/e2s/retarget_env/s0
export PKG_DIR=/usr1/e2s/v2d_challenge_data/outputs/ego2sim_recon_v2clean2_pkg FEAT_DIR=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/s0_features_v2clean2 NO_COMPARE=1
GL=/usr1/e2s/v2d_challenge_data/outputs/gmd/v4_clean2/logs; mkdir -p $GL
setsid nohup ./s0_queue.sh 0 $S/jobs_gpu0.txt > $T/eval/queue_v2clean2_gpu0.out 2>&1 < /dev/null &
( cd /usr1/e2s/retarget_env/gmd_v4c2
  export GMD_SET=v2 GMD_VOC=ff SUB=v4_clean2 BAND_AFTER=52428800 NOBEST_N=6 TOTAL=81788928
  echo "$(date -u '+%F %T') M X_ep28 starting on GPU 1" >> $GL/pipeline.log
  ./run_m.sh 28 1 X_ep28 g9gated.npz > $GL/run_m_X_ep28.log 2>&1
  echo "$(date -u '+%F %T') M X_ep28 finished: $(cat /usr1/e2s/v2d_challenge_data/outputs/gmd/v4_clean2/m/X_ep28/STOP 2>/dev/null || echo cap)" >> $GL/pipeline.log
  cd /usr1/e2s/retarget_env/s0 && exec ./s0_queue.sh 1 $S/jobs_gpu1.txt ) > $T/eval/queue_v2clean2_gpu1.out 2>&1 < /dev/null &
