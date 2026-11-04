#!/usr/bin/env bash
# Resume after the spoon fix: X2B_ep26 (M with g9v2negB) on GPU 1, X1_ep6 (exploration 0.2 + grip reward, g9v2) on GPU 0,
# then gdiag diagnostics of both best checkpoints.
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v2 GMD_VOC=ff SUB=v2 BAND_AFTER=52428800 NOBEST_N=6
V=/usr1/e2s/v2d_challenge_data/outputs/gmd/v2; E=$V/exp; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
log() { echo "$(date -u '+%F %T') $*" | tee -a $E/pipeline.log; }
( log "X2B_ep26 (M with g9v2negB) starting on GPU 1"; ./run_m.sh 26 1 X2B_ep26 g9v2negB.npz > $E/run_X2B_ep26.log 2>&1; log "X2B_ep26 finished: $(cat $V/m/X2B_ep26/STOP 2>/dev/null || echo cap)" ) &
sleep 60
( log "X1_ep6 (exploration 0.2 + grip reward) starting on GPU 0"; GMD_GRIP_LIB=/out/v2/exp/grip_lib_ep6.npz GMD_GRIP_W=10 EXTRA_OV="training.exploration.target_sigma=0.2" ./run_m.sh 6 0 X1_ep6 g9v2.npz > $E/run_X1_ep6.log 2>&1; log "X1_ep6 finished: $(cat $V/m/X1_ep6/STOP 2>/dev/null || echo cap)" ) &
wait
for r in X2B_ep26 X1_ep6; do
  EP=${r##*_ep}; GF=g9v2.npz; [[ $r == X2B_* ]] && GF=g9v2negB.npz
  B=$($PY -c "import json;r=[json.loads(l) for l in open('$V/m/$r/evals.jsonl')];print(max(r,key=lambda x:x['add_auc'])['step'])")
  MEM=12g timeout 1200 ./run_fc.sh 0 $EP gdiag_$r -- "GMD_G=/out/v2/g/$GF python /env/gmd/gdiag.py /out/v2/m/$r/run/policy_$B.safetensors /out/v2/exp/gdiag_$r.npz --worlds 16" > $E/gdiag_$r.log 2>&1
done
log "resume finished"
