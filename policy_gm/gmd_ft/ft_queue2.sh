#!/usr/bin/env bash
# gmd_ft queue 2, started after queue 1 (ft_queue.sh) exits. One worker per GPU, each waits until its GPU is free
# (< 1500 MiB, < 10 % utilisation, 3 checks a minute apart) before taking the next job:
#   harsh:<run>  grip test at plan-(b) strength + noise floor (ft_job.sh harsh), ~10 min each
#   joint:<run>  plan (a), joint training of M and finger deviations from scratch (ft_job.sh joint), ~75 min each
# Log: outputs/gmd_ft/logs/queue2.log
set -uo pipefail
cd /usr1/e2s/gmd_ft
FO=/usr1/e2s/v2d_challenge_data/outputs/gmd_ft; L=$FO/logs; mkdir -p $L
JOBS="harsh:B_ep6 harsh:B_ep15 harsh:A_ep3 joint:B_ep6 joint:B_ep15 joint:A_ep3"
log() { echo "$(date -u '+%F %T') $*" | tee -a $L/queue2.log; }
gpu_free() {
  local used util
  read -r used util < <(nvidia-smi -i $1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
  [[ $used -lt 1500 && $util -lt 10 ]]
}
wait_gpu() { local n=0; while (( n < 3 )); do if gpu_free $1; then n=$((n + 1)); else n=0; fi; sleep 60; done; }
auc() { python3 -c "import json,sys;print('%.3f' % json.load(open(sys.argv[1]))['add_auc'])" "$1" 2>/dev/null || echo "-"; }
best_auc() { python3 -c "import json,sys;r=[json.loads(l) for l in open(sys.argv[1])];b=max(r,key=lambda x:x['add_auc']);print('%.3f at %.1fM' % (b['add_auc'], b['step']/1e6))" "$1" 2>/dev/null || echo "-"; }

worker() {
  local GPU=$1 J KIND RUN
  for J in $JOBS; do
    [[ -d $L/claim2_${J//:/_} ]] && continue
    wait_gpu $GPU
    mkdir $L/claim2_${J//:/_} 2> /dev/null || continue
    IFS=: read -r KIND RUN <<< "$J"
    log "$J starting on GPU $GPU"
    if [[ $KIND == harsh ]]; then
      ./ft_job.sh harsh $GPU $RUN
      H=$FO/harsh/$RUN
      log "$J done: unshaken R off $(auc ${H}_normal_r_off_rep1.json) / $(auc ${H}_normal_r_off_rep2.json) / $(auc ${H}_normal_r_off_rep3.json); harsh R off $(auc ${H}_harsh_zero_r1.json), harsh R on $(auc ${H}_harsh_zero_r0.json)"
    else
      if ./ft_job.sh joint $GPU $RUN j_$RUN; then log "$J done: best $(best_auc $FO/j_$RUN/evals.jsonl)"; else log "$J FAILED (see $FO/j_$RUN/train.log)"; fi
    fi
  done
  log "worker GPU $GPU done"
}

log "queue 2 waiting for queue 1 to finish (pid ${Q1:-none})"
while [[ -n ${Q1:-} ]] && kill -0 $Q1 2> /dev/null; do sleep 60; done
log "queue 2 started: $JOBS"
worker 0 & sleep 5; worker 1 & wait
log "queue 2 finished"
