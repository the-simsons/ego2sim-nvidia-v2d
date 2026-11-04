#!/usr/bin/env bash
# run_after.sh <gpu> <PID of the queue to wait for> then: <episode> <exp> <worlds> <overrides> <pkg> <feat> ... (6 fields per job)
G=$1; WAIT=$2; shift 2; T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; cd /usr1/e2s/retarget_env/s0
while kill -0 $WAIT 2>/dev/null; do sleep 30; done                    # WAIT = PID of the queue to wait for
while [[ $# -ge 6 ]]; do
  EP=$1; EXP=$2; W=$3; OV=$4; PKG=$5; FEAT=$6; shift 6
  J=$T/_scratch/contact/job_$EP.txt; echo "single $EP $EXP - $W $OV tracking_96m_pickup_ray5decay_contact wrench 0.001" > $J
  PKG_DIR=$PKG FEAT_DIR=$FEAT NO_COMPARE=1 sg e2sshare -c "./s0_queue.sh $G $J" > $T/eval/queue_contact_ep$EP.out 2>&1
done
