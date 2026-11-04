#!/usr/bin/env bash
# Rescore run 3's best M checkpoints at inference time:  rescore.sh <gpu> <g name (in v3/g)> <debounce K> ...
# Out: outputs/gmd/v3/rescore/<run>_<g>_k<K>.json. 4 containers at a time on the GPU.
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v2
G=/usr1/e2s/v2d_challenge_data/outputs/gmd/v3; D=$G/rescore; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
GPU=$1; GN=$2; shift 2
for K in "$@"; do
  for R in A_ep41 A_ep26 A_ep3 B_ep37 B_ep6 B_ep15 B_ep30 C_ep19 C_ep28; do
    [[ -f $D/${R}_${GN}_k$K.json ]] && continue
    EP=${R##*_ep}
    B=$($PY -c "import json;r=[json.loads(l) for l in open('$G/m/$R/evals.jsonl')];print(max(r,key=lambda x:x['add_auc'])['step'])")
    CK=/out/v3/m/$R/run/policy_$B.safetensors; [[ $R == A_ep26 ]] && CK=/out/v2/m/X2B_ep26/run/policy_$B.safetensors
    while [ $(docker ps --format '{{.Names}}' | grep -c "gmd_rs${GPU}_") -ge 4 ]; do sleep 5; done
    MEM=8g timeout 1800 ./run_fc.sh $GPU $EP rs${GPU}_${R}_${GN}_k$K -- "GMD_G=/out/v3/g/$GN.npz GMD_DEBOUNCE=$K python /env/gmd/score_ck.py $CK /out/v3/rescore/${R}_${GN}_k$K.json --worlds 64" > $D/${R}_${GN}_k$K.log 2>&1 &
    sleep 5
  done
done
wait
