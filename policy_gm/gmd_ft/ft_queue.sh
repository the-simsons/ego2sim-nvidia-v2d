#!/usr/bin/env bash
# gmd_ft pilot queue. One worker per GPU; each worker waits until its GPU is free (< 1500 MiB used and < 10 % utilisation for
# 3 checks a minute apart) before every training job, then takes the next unclaimed chain. Chains (in order):
#   ep6, ep15, ep3 (K = 3: squeeze + pc1 + pc2):  parity -> R -> [gate] -> M resume
#   ep6 ablation (K = 1: squeeze only):           parity -> R
# Gates: parity |AUC(FTEnv, R = 0) - AUC(v4 best)| <= 0.03, else the chain stops (mechanism bug);
#        M is resumed only if the best R beats parity by >= 0.01 (otherwise R did not help on this episode).
# Log: outputs/gmd_ft/logs/queue.log; summary: outputs/gmd_ft/SUMMARY.md (summarize.py) when both workers finish.
set -uo pipefail
cd /usr1/e2s/gmd_ft
FO=/usr1/e2s/v2d_challenge_data/outputs/gmd_ft; L=$FO/logs; mkdir -p $L
CHAINS="B_ep6:3:full B_ep15:3:full A_ep3:3:full B_ep6:1:ronly"
log() { echo "$(date -u '+%F %T') $*" | tee -a $L/queue.log; }
gpu_free() {
  local used util
  read -r used util < <(nvidia-smi -i $1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
  [[ $used -lt 1500 && $util -lt 10 ]]
}
wait_gpu() { local n=0; while (( n < 3 )); do if gpu_free $1; then n=$((n + 1)); else n=0; fi; sleep 60; done; }
auc() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['add_auc'])" "$1" 2>/dev/null; }
best_auc() { python3 -c "import json,sys;print(max(json.loads(l)['add_auc'] for l in open(sys.argv[1])))" "$1" 2>/dev/null; }

chain() {  # <gpu> <run> <K> <full|ronly>
  local GPU=$1 RUN=$2 K=$3 KIND=$4 RN=r_${2}_k$3 MN=m_${2}_k$3
  local V4B P RB
  V4B=$(best_auc /usr1/e2s/v2d_challenge_data/outputs/gmd/v4/m/$RUN/evals.jsonl)
  if [[ ! -f $FO/parity/$RUN.json ]]; then
    wait_gpu $GPU; log "$RUN parity on GPU $GPU"
    ./ft_job.sh parity $GPU $RUN
  fi
  P=$(auc $FO/parity/$RUN.json)
  if [[ -z $P ]] || ! python3 -c "import sys;sys.exit(0 if abs($P - $V4B) <= 0.03 else 1)"; then
    log "$RUN PARITY FAILED: FTEnv with R = 0 gives ${P:-no result}, v4 best $V4B -> chain stopped"; return
  fi
  log "$RUN parity ok: $P (v4 best $V4B)"
  wait_gpu $GPU; log "$RN (R, K = $K) starting on GPU $GPU"
  if ! ./ft_job.sh train_r $GPU $RUN $RN $K; then log "$RN FAILED (see $FO/$RN/train.log)"; return; fi
  RB=$(best_auc $FO/$RN/evals.jsonl)
  log "$RN finished: best R $RB vs parity $P; perturbed with R $(auc $FO/$RN/perturbed_zero_r0.json), without R $(auc $FO/$RN/perturbed_zero_r1.json)"
  [[ $KIND == ronly ]] && return
  if ! python3 -c "import sys;sys.exit(0 if $RB >= $P + 0.01 else 1)"; then log "$RUN: R gave no gain (< +0.01) -> M not resumed"; return; fi
  wait_gpu $GPU; log "$MN (M resume with R) starting on GPU $GPU"
  if ! ./ft_job.sh resume_m $GPU $RUN $MN $RN; then log "$MN FAILED (see $FO/$MN/train.log)"; return; fi
  log "$MN finished: best $(best_auc $FO/$MN/evals.jsonl) (v4 best $V4B); best M with R off $(auc $FO/$MN/best_r_off.json)"
}

worker() {
  local GPU=$1 C
  for C in $CHAINS; do
    [[ -d $L/claim_${C//:/_} ]] && continue
    wait_gpu $GPU                                   # claim a chain only once this GPU is free
    mkdir $L/claim_${C//:/_} 2> /dev/null || continue
    IFS=: read -r RUN K KIND <<< "$C"
    chain $GPU $RUN $K $KIND
  done
  log "worker GPU $GPU done"
}

log "queue started (pid $$): chains $CHAINS"
worker 0 & sleep 5; worker 1 & wait
/usr1/e2s/video_to_data/reconstruction/.venv/bin/python summarize.py >> $L/queue.log 2>&1
log "queue finished; summary in $FO/SUMMARY.md"
