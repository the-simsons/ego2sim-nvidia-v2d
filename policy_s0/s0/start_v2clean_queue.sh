#!/usr/bin/env bash
# Train the latest recipe (as start_v2_all_queue.sh) on eval episodes 8, 10, 14, 16, 32, 33, 36 with the cleaned Ego2Sim v2
# references (rborah, 2026-10-30: v2 meshes, object poses cleaned with clean_refs / settle_spawn). Features first.
# Run as: setsid nohup sg e2sshare -c ./start_v2clean_queue.sh   (the containers need e2sshare as the primary group)
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; S=$T/_scratch/v2clean; cd /usr1/e2s/retarget_env/s0
export PKG_DIR=/usr1/e2s/v2d_challenge_data/outputs/ego2sim_recon_v2clean_pkg FEAT_DIR=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/s0_features_v2clean NO_COMPARE=1
EPS="8 10 14 16 32 33 36"
../run_in_env.sh --gpu 0 /env/s0/s0_features_v2clean.py $EPS > $S/features.log 2>&1 || { echo "features failed" >> $S/features.log; exit 1; }
for ep in $EPS; do [[ -f $FEAT_DIR/$(printf 'episode_%06d' $ep).npz ]] || { echo "missing features for ep$ep" >> $S/features.log; exit 1; }; done
setsid nohup ./s0_queue.sh 0 $S/jobs_gpu0.txt > $T/eval/queue_v2clean_gpu0.out 2>&1 < /dev/null &
setsid nohup ./s0_queue.sh 1 $S/jobs_gpu1.txt > $T/eval/queue_v2clean_gpu1.out 2>&1 < /dev/null &
