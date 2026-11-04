#!/usr/bin/env bash
# Wait until a GPU stays free, then run EgoFound3R (F0) on prepared input videos; back off if it gets busy.
#
#   [INPUT_DIR=data/track3_input OUT_DIR=data/egofound3r/track3] tools/run_egofound3r_when_gpu_free.sh EP [EP ...]
#   (from handtracking/egofound/; run detached). Input: <INPUT_DIR>/<EP>.mp4 and <EP>.json (its "fps" is passed on).
#
# KNOWN_K=1: EgoFound3R uses the true camera K (<INPUT_DIR>/<EP>_rootK.json, tools/write_root_k_sidecars.py) instead of
# its own predicted focal (--root-k-source external).
#
# Once a GPU is acquired, the following episodes run back to back on it while no other user's process is on it; the
# free-for-FREE_CHECKS wait applies only when a GPU has to be acquired.
#
# Free = no compute processes, < FREE_MIB used and < FREE_UTIL % for FREE_CHECKS checks POLL_S apart.
# While running: nvidia-smi samples memory every 200 ms (logs/vram_<ep>.csv). If another user's process appears on our
# GPU, our run is stopped and we go back to waiting (INSTRUCTIONS.md: back off). Per episode writes
# <OUT_DIR>/<ep>/ (EgoFound3R outputs + overlay) and run_meta.json (commit, checkpoints, args, GPU, time).
# Our own GPU watchers never share a GPU: each holds logs/gpu<N>.lock (flock) while its job runs.
set -uo pipefail
cd "$(dirname "$0")/.."
FREE_MIB=${FREE_MIB:-2000}; FREE_UTIL=${FREE_UTIL:-10}; FREE_CHECKS=${FREE_CHECKS:-10}; POLL_S=${POLL_S:-30}
EF=../methods/EgoFound3R
PY=$(cd ../methods/.venv-egofound3r/bin && pwd)/python  # not realpath: that resolves the venv symlink to the system python
CFG=$(realpath configs/egofound3r.toml)
CKPT=$(realpath weights-vggt/EgoFound3R.pt)
INPUT_DIR=${INPUT_DIR:-data/track3_input}; OUT_DIR=${OUT_DIR:-data/egofound3r/track3}; KNOWN_K=${KNOWN_K:-0}
log() { echo "[$(date -u +%FT%TZ)] $*"; }

free_gpu() {  # prints a GPU index that is free right now, or nothing
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

foreign_on() {  # 0 if a process not started by us runs on GPU $1
  local uuid pids p
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$1" '$1==g{print $2}')
  pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u{print $2}')
  for p in $pids; do [ "$(ps -o user= -p "$p" | tr -d ' ')" != "$USER" ] && return 0; done
  return 1
}

run_episode() {  # ep gpu -> 0 done, 1 failed, 2 backed off
  local ep=$1 g=$2 out=$PWD/$OUT_DIR/$1 vid fps t0 rc kargs=()
  [ "$KNOWN_K" = 1 ] && kargs=(--root-k-source external --root-k-file "$(realpath "$INPUT_DIR/${ep}_rootK.json")")
  vid=$(realpath "$INPUT_DIR/$ep.mp4")
  fps=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['fps'])" "$INPUT_DIR/$ep.json")
  mkdir -p "$out" logs
  nvidia-smi -i "$g" --query-gpu=timestamp,memory.used --format=csv -lms 200 > "logs/vram_$ep.csv" &
  local smi=$!
  t0=$(date +%s.%N)
  (cd "$EF" && exec env CUDA_VISIBLE_DEVICES="$g" "$PY" infer_marker_video.py --config "$CFG" --checkpoint "$CKPT" \
     --video-path "$vid" --output-dir "$out" --input-fps "$fps" --device cuda:0 --save-overlay \
     --no-overlay-frame-images --upsample-mano --allow-output-overwrite "${kargs[@]}") > "logs/egofound3r_$ep.log" 2>&1 &
  local job=$!
  while kill -0 "$job" 2>/dev/null; do
    if foreign_on "$g"; then
      log "$ep: another user's process appeared on GPU $g, backing off"; pkill -P "$job"; kill "$job" 2>/dev/null
      wait "$job" 2>/dev/null; kill "$smi"; return 2
    fi
    sleep 5
  done
  wait "$job"; rc=$?
  kill "$smi"
  [ $rc -ne 0 ] && { log "$ep: FAILED (exit $rc), see logs/egofound3r_$ep.log"; return 1; }
  "$PY" - "$out" "$ep" "$g" "$t0" "$vid" "$CKPT" "$CFG" "$fps" "${kargs[*]}" <<'EOF'
import hashlib, json, subprocess, sys, time
out, ep, g, t0, vid, ckpt, cfg, fps, kargs = sys.argv[1:]
peak = 0
for line in open(f"logs/vram_{ep}.csv").read().splitlines()[1:]:
    try: peak = max(peak, int(line.split(",")[1].split()[0]))
    except (IndexError, ValueError): pass
summ = json.load(open(f"{out}/summary.json"))
gpu = subprocess.run(["nvidia-smi", "-i", g, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
meta = {"method": "F0", "episode": ep, "input_video": vid, "gpu_index": int(g), "gpu": gpu,
        "wall_time_s": time.time() - float(t0), "nvidia_smi_peak_mib": peak,
        "repo_commit": git("rev-parse", "HEAD"),
        "egofound3r_commit": git("-C", "../methods/EgoFound3R", "rev-parse", "HEAD"),
        "checkpoint": ckpt, "config": cfg, "seed": 0,
        "args": f"--input-fps {fps} --save-overlay --no-overlay-frame-images --upsample-mano (input resize: keep_aspect default) " + kargs,
        "summary_performance": {k: v for k, v in summ.get("performance", summ).items() if "peak" in k or "fps" in k}}
json.dump(meta, open(f"{out}/run_meta.json", "w"), indent=1)
print(json.dumps({k: meta[k] for k in ("wall_time_s", "nvidia_smi_peak_mib", "gpu")}))
EOF
  log "$ep: DONE"
  return 0
}

g=""  # GPU we hold (lock on fd 9), kept across episodes while no other user's process appears on it
for ep in "$@"; do
  while true; do
    if [ -z "$g" ]; then
      log "$ep: waiting for a GPU free for $FREE_CHECKS x ${POLL_S}s"
      g=$(wait_for_gpu)
      claim_gpu "$g" || { log "$ep: GPU $g held by another of our jobs"; g=""; sleep "$POLL_S"; continue; }
      log "$ep: GPU $g free, starting EgoFound3R"
    elif foreign_on "$g"; then
      log "$ep: another user's process is on GPU $g, releasing it"; exec 9>&-; g=""; continue
    else
      log "$ep: continuing on GPU $g"
    fi
    run_episode "$ep" "$g"; rc=$?
    [ $rc -eq 2 ] && { exec 9>&-; g=""; continue; }
    break
  done
done
exec 9>&-
log "ALL_DONE"
