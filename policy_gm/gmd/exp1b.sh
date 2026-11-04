#!/usr/bin/env bash
# Experiment 1b: broad grip reward (12 cm / 45 deg dense + 3 cm / 20 deg bonus), windows over approach + hold, exploration 0.1.
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v2 GMD_VOC=ff SUB=v2 BAND_AFTER=52428800 NOBEST_N=6 GMD_GRIP_W=10 GMD_GRIP_KERNEL=broad
V=/usr1/e2s/v2d_challenge_data/outputs/gmd/v2; E=$V/exp; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
log() { echo "$(date -u '+%F %T') $*" | tee -a $E/pipeline.log; }
run() {  # <ep> <gpu>
  log "X1b_ep$1 (broad grip reward, exploration 0.1) starting on GPU $2"
  GMD_GRIP_LIB=/out/v2/exp/grip_lib_ep$1_b.npz ./run_m.sh $1 $2 X1b_ep$1 g9v2.npz > $E/run_X1b_ep$1.log 2>&1
  log "X1b_ep$1 finished: $(cat $V/m/X1b_ep$1/STOP 2>/dev/null || echo cap)"
  B=$($PY -c "import json;r=[json.loads(l) for l in open('$V/m/X1b_ep$1/evals.jsonl')];print(max(r,key=lambda x:x['add_auc'])['step'])")
  MEM=12g timeout 1200 ./run_fc.sh $2 $1 gdiag_X1b_$1 -- "GMD_G=/out/v2/g/g9v2.npz python /env/gmd/gdiag.py /out/v2/m/X1b_ep$1/run/policy_$B.safetensors /out/v2/exp/gdiag_X1b_ep$1.npz --worlds 16" > $E/gdiag_X1b_ep$1.log 2>&1
}
run 19 1 & sleep 60; run 6 0 & wait
log "exp1b finished"
