#!/usr/bin/env bash
# gmd_ft queue 3: ablation of plan (a) with unfreezing switched off. One worker per GPU, each waits until its GPU is free
# (< 1500 MiB, < 10 % utilisation, 3 checks a minute apart) before taking the next job:
#   jo:<run>  joint setup (jenv.py: G inputs + G output in the observation, 58-d action) with FJ_UNFREEZE never reached,
#             so the fingers are frozen G for the whole run (ft_job.sh joint, name jo_<run>), ~60 min each
# Log: outputs/gmd_ft/logs/queue3.log
set -uo pipefail
cd /usr1/e2s/gmd_ft
FO=/usr1/e2s/v2d_challenge_data/outputs/gmd_ft; L=$FO/logs; mkdir -p $L
JOBS="jo:B_ep6 jo:B_ep15 jo:A_ep3"
log() { echo "$(date -u '+%F %T') $*" | tee -a $L/queue3.log; }
gpu_free() {
  local used util
  read -r used util < <(nvidia-smi -i $1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
  [[ $used -lt 1500 && $util -lt 10 ]]
}
wait_gpu() { local n=0; while (( n < 3 )); do if gpu_free $1; then n=$((n + 1)); else n=0; fi; sleep 60; done; }
best_auc() { python3 -c "import json,sys;r=[json.loads(l) for l in open(sys.argv[1])];b=max(r,key=lambda x:x['add_auc']);print('%.3f at %.1fM' % (b['add_auc'], b['step']/1e6))" "$1" 2>/dev/null || echo "-"; }

worker() {
  local GPU=$1 J KIND RUN
  for J in $JOBS; do
    [[ -d $L/claim3_${J//:/_} ]] && continue
    wait_gpu $GPU
    mkdir $L/claim3_${J//:/_} 2> /dev/null || continue
    IFS=: read -r KIND RUN <<< "$J"
    log "$J starting on GPU $GPU"
    if FJ_UNFREEZE=1000000000000000 ./ft_job.sh joint $GPU $RUN jo_$RUN; then log "$J done: best $(best_auc $FO/jo_$RUN/evals.jsonl)"; else log "$J FAILED (see $FO/jo_$RUN/train.log)"; fi
  done
  log "worker GPU $GPU done"
}

log "queue 3 started: $JOBS"
worker 0 & sleep 5; worker 1 & wait
log "queue 3 finished"
