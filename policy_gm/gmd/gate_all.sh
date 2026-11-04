#!/usr/bin/env bash
# Gate 1 for one G on a list of episodes, one GPU, 3 at a time.   gate_all.sh <gpu> <g name> <episodes...>
GPU=$1; GN=$2; shift 2; cd /usr1/e2s/retarget_env/gmd; L=/usr1/e2s/v2d_challenge_data/outputs/gmd/full/logs
for EP in "$@"; do
  while [ $(docker ps --format '{{.Names}}' | grep -c "gmd_gate_g${GPU}_") -ge 3 ]; do sleep 5; done
  MEM=8g timeout 1800 ./run_fc.sh $GPU $EP gate_g${GPU}_$EP -- python /env/gmd/rollout.py $(python3 episodes.py ckpt $EP) /out/full/ep$EP fingers_g \
    --g /out/full/g/${GN}_for_ep$EP.npz --tag _$GN > $L/gate_${GN}_$EP.log 2>&1 &
  sleep 2
done
wait
