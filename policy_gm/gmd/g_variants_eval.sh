#!/usr/bin/env bash
# Evaluate G variants (teacher wrists, 64 worlds): fingers_g on all 9 teachers, fingers_g_off on the spoon episodes.
#   g_variants_eval.sh <gpu> <g name> ...      results: outputs/gmd/v2/ep<N>/fingers_g[_off]_<g>.json
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v2
V=/usr1/e2s/v2d_challenge_data/outputs/gmd/v2; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
GPU=$1; shift
TEACH="35 36 41 40 18 2 3 26 8"
for G in "$@"; do
  until [[ -f $V/g/$G.npz ]]; do sleep 15; done
  for e in $TEACH; do
    [[ -f $V/g/${G}_for_ep$e.npz ]] || $PY -c "import numpy as np;p=dict(np.load('$V/g/$G.npz'));np.savez('$V/g/${G}_for_ep$e.npz',engage_timeline=np.load('$V/ep$e/engage_timeline.npy'),**p)"
  done
done
jobs=()
for G in "$@"; do
  for e in $TEACH; do jobs+=("$e fingers_g $G"); done
  for e in 35 36 3; do jobs+=("$e fingers_g_off $G"); done
done
for j in "${jobs[@]}"; do
  set -- $j
  [[ -f $V/ep$1/${2}_$3.json ]] && continue
  while [ $(docker ps --format '{{.Names}}' | grep -c "gmd_gv_") -ge 4 ]; do sleep 5; done
  MEM=8g timeout 1800 ./run_fc.sh $GPU $1 gv_$1_$2_$3 -- python /env/gmd/rollout.py $(python3 episodes.py ckpt $1) /out/v2/ep$1 $2 \
    --g /out/v2/g/${3}_for_ep$1.npz --tag _$3 > $V/exp/gv_$1_$2_$3.log 2>&1 &
  sleep 5
done
wait
