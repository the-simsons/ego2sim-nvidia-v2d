#!/usr/bin/env bash
# Record teacher rollouts for a list of episodes on one GPU.   rec_all.sh <gpu> <episodes...>
GPU=$1; shift; cd /usr1/e2s/retarget_env/gmd; L=/usr1/e2s/v2d_challenge_data/outputs/gmd/full/logs
S="nominal perturb dart0.1 dart0.2 perturb_dart0.1 perturb_dart0.2 fingers_open"
for EP in "$@"; do
  MEM=10g timeout 2400 ./run_fc.sh $GPU $EP rec$EP -- python /env/gmd/rollout.py $(python3 episodes.py ckpt $EP) /out/full/ep$EP $S --worlds 64 > $L/rec$EP.log 2>&1
  echo "ep$EP $(grep -c '^\[' $L/rec$EP.log) lines, exit $?" >> $L/rec_status.txt
done
