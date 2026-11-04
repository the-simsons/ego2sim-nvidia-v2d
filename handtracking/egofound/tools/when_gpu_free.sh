#!/usr/bin/env bash
# Run a command once a GPU stays free; stop it and wait again if another user's process lands on that GPU.
#
#   tools/when_gpu_free.sh LOG -- CMD [ARGS ...]        (from handtracking/egofound/; run detached)
# The literal word GPU in CMD/ARGS is replaced by the chosen GPU index (CUDA_VISIBLE_DEVICES is also set).
# The command must be resumable: after a back-off it is started again from the beginning.
# Free = no compute processes, < FREE_MIB used and < FREE_UTIL % for FREE_CHECKS checks POLL_S apart.
# Our own GPU watchers never share a GPU: each holds logs/gpu<N>.lock (flock) while its job runs.
# Exit code of CMD 75 also means "backed off" (the command noticed a foreign process itself).
set -uo pipefail
cd "$(dirname "$0")/.."
FREE_MIB=${FREE_MIB:-2000}; FREE_UTIL=${FREE_UTIL:-10}; FREE_CHECKS=${FREE_CHECKS:-10}; POLL_S=${POLL_S:-30}
LOG=$1; shift; [ "$1" = "--" ] && shift
log() { echo "[$(date -u +%FT%TZ)] $*"; }

free_gpu() {
  local busy
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits |
    while IFS=', ' read -r i u m t; do
      if ! grep -q "$u" <<<"$busy" && [ "$m" -lt "$FREE_MIB" ] && [ "$t" -lt "$FREE_UTIL" ]; then echo "$i"; break; fi
    done
}

claim_gpu() {  # take logs/gpu<N>.lock (shared by our GPU watchers) on fd 9; 1 if another of our jobs holds it
  exec 9>"logs/gpu$1.lock"
  flock -n 9 || { exec 9>&-; return 1; }
}

wait_for_gpu() {
  local streak=0 g last=""
  while true; do
    g=$(free_gpu)
    if [ -n "$g" ] && [ "$g" = "$last" ]; then streak=$((streak + 1)); elif [ -n "$g" ]; then streak=1; else streak=0; fi
    last=$g
    [ "$streak" -ge "$FREE_CHECKS" ] && { echo "$g"; return; }
    sleep "$POLL_S"
  done
}

foreign_on() {
  local uuid pids p
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$1" '$1==g{print $2}')
  pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u{print $2}')
  for p in $pids; do [ "$(ps -o user= -p "$p" | tr -d ' ')" != "$USER" ] && return 0; done
  return 1
}

while true; do
  log "waiting for a GPU free for $FREE_CHECKS x ${POLL_S}s: $*"
  g=$(wait_for_gpu)
  claim_gpu "$g" || { log "GPU $g held by another of our jobs"; sleep "$POLL_S"; continue; }
  args=(); for x in "$@"; do args+=("${x//GPU/$g}"); done
  log "GPU $g free, starting"
  setsid env CUDA_VISIBLE_DEVICES="$g" "${args[@]}" >> "$LOG" 2>&1 &
  job=$!
  backed=0
  while kill -0 "$job" 2>/dev/null; do
    if foreign_on "$g"; then
      log "another user's process appeared on GPU $g, backing off"; kill -- -"$job" 2>/dev/null; backed=1; break
    fi
    sleep 5
  done
  wait "$job"; rc=$?
  exec 9>&-
  { [ $backed -eq 1 ] || [ $rc -eq 75 ]; } && { sleep 5; continue; }
  [ $rc -ne 0 ] && { log "FAILED (exit $rc), see $LOG"; exit $rc; }
  log "DONE"; exit 0
done
