#!/usr/bin/env bash
# Two experiments in parallel (approved 2026-11-01), outputs in outputs/gmd/v2/exp and v2/m/X*:
#  X1 (G unchanged): M with exploration target_sigma 0.2 + grip reward (grip_lib, weight 10, hand_ray schedule) on ep19, ep6
#  X2: G retrained with engage counterfactuals + contact dropout (g9v2neg) -> validation (a) skill kept (Gate 1 >= 90% on
#      >= 7/9), (b) engage causal (engage forced off -> within 0.1 of the fingers-open floor on >= 7/9), (c) flip test ->
#      M on ep26 with g9v2neg if (a) and (b) pass
# GPU 0: X1_ep19 -> X1_ep6 (if unclaimed); GPU 1: X2 validation -> X2_ep26 -> X1_ep6 (if unclaimed). Then diagnostics.
set -uo pipefail
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v2 GMD_VOC=ff SUB=v2 BAND_AFTER=52428800 NOBEST_N=6
O=/usr1/e2s/v2d_challenge_data/outputs; V=$O/gmd/v2; E=$V/exp; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
TEACH="35 36 41 40 18 2 3 26 8"
log() { echo "$(date -u '+%F %T') $*" | tee -a $E/pipeline.log; }
x1() {  # <ep> <gpu>
  mkdir $E/claim_X1_ep$1 2> /dev/null || return 0
  log "X1_ep$1 (exploration 0.2 + grip reward) starting on GPU $2"
  GMD_GRIP_LIB=/out/v2/exp/grip_lib_ep$1.npz GMD_GRIP_W=10 EXTRA_OV="training.exploration.target_sigma=0.2" ./run_m.sh $1 $2 X1_ep$1 g9v2.npz > $E/run_X1_ep$1.log 2>&1
  log "X1_ep$1 finished: $(cat $V/m/X1_ep$1/STOP 2>/dev/null || echo cap)"
}
gpu0() { x1 19 0; x1 6 0; }
gpu1() {
  until [[ -f $V/g/g9v2neg.npz ]] && ! docker ps --format '{{.Names}}' | grep -q gmd_train_g9v2neg; do sleep 20; done
  log "X2: g9v2neg trained ($(grep -c 'step ' $E/train_g9v2neg.log) progress lines); validation"
  for e in $TEACH; do
    $PY -c "import numpy as np;p=dict(np.load('$V/g/g9v2neg.npz'));np.savez('$V/g/g9v2neg_for_ep$e.npz',engage_timeline=np.load('$V/ep$e/engage_timeline.npy'),**p)"
  done
  jobs=(); for e in $TEACH; do jobs+=("$e fingers_g g9v2neg" "$e fingers_g_off g9v2neg" "$e fingers_g_off g9v2"); done
  for j in "${jobs[@]}"; do
    set -- $j
    [[ -f $V/ep$1/${2}_$3.json ]] && continue
    while [ $(docker ps --format '{{.Names}}' | grep -c "gmd_val_") -ge 4 ]; do sleep 5; done
    MEM=8g timeout 1800 ./run_fc.sh 1 $1 val_$1_$2_$3 -- python /env/gmd/rollout.py $(python3 episodes.py ckpt $1) /out/v2/ep$1 $2 \
      --g /out/v2/g/${3}_for_ep$1.npz --tag _$3 > $E/val_$1_$2_$3.log 2>&1 &
    sleep 3
  done
  wait
  $PY flip_test.py $V/g/g9v2.npz $V/g/g9v2neg.npz > $E/flip_test.txt 2>&1
  PASS=$($PY - <<PYEOF
import json
def a(e, n):
    try: return json.load(open(f"$V/ep{e}/{n}.json"))["add_auc"]
    except FileNotFoundError: return None
out = open("$E/validation.txt", "w"); na = nb = 0
print("ep | teacher | fingers-open floor | G-neg normal (ratio) | G-neg engage off | G-v2 engage off", file=out)
for e in "$TEACH".split():
    t, fo, gn, gno, g2o = a(e, "nominal"), a(e, "fingers_open"), a(e, "fingers_g_g9v2neg"), a(e, "fingers_g_off_g9v2neg"), a(e, "fingers_g_off_g9v2")
    r = gn / t if gn and t else 0
    na += r >= 0.9; nb += gno is not None and fo is not None and gno - fo <= 0.1
    f = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"{e} | {f(t)} | {f(fo)} | {f(gn)} ({100 * r:.0f}%) | {f(gno)} | {f(g2o)}", file=out)
print(f"(a) skill kept: {na}/9 >= 90%   (b) engage causal: {nb}/9 within 0.1 of the fingers-open floor", file=out)
print(int(na >= 7 and nb >= 7))
PYEOF
)
  log "X2 validation: $(tail -n 1 $E/validation.txt); flip test: $(tail -n 1 $E/flip_test.txt | cut -c1-160)"
  if [[ $PASS == 1 ]]; then
    log "X2_ep26 (M with g9v2neg) starting on GPU 1"
    ./run_m.sh 26 1 X2_ep26 g9v2neg.npz > $E/run_X2_ep26.log 2>&1
    log "X2_ep26 finished: $(cat $V/m/X2_ep26/STOP 2>/dev/null || echo cap)"
  else
    log "X2 validation failed: M on ep26 skipped (decision needed)"
  fi
  x1 6 1
}
gpu0 & sleep 5; gpu1 & wait
while docker ps --format '{{.Names}}' | grep -q "gmd_me_\|gmd_m_"; do sleep 30; done
log "diagnostics"
for r in X1_ep19 X1_ep6 X2_ep26; do
  [[ -f $V/m/$r/evals.jsonl ]] || continue
  EP=${r##*_ep}; GF=g9v2.npz; [[ $r == X2_* ]] && GF=g9v2neg.npz
  B=$($PY -c "import json;r=[json.loads(l) for l in open('$V/m/$r/evals.jsonl')];print(max(r,key=lambda x:x['add_auc'])['step'])")
  MEM=12g timeout 1200 ./run_fc.sh 0 $EP gdiag_$r -- "GMD_G=/out/v2/g/$GF python /env/gmd/gdiag.py /out/v2/m/$r/run/policy_$B.safetensors /out/v2/exp/gdiag_$r.npz --worlds 16" > $E/gdiag_$r.log 2>&1
done
log "finished"
