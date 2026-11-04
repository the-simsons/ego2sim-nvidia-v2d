#!/usr/bin/env bash
# After g9v2off trains: gated G = g9v2 (engage on) + g9v2off (engage off) -> v3/g/g9gated.npz; validation with teacher
# wrists (fingers_g, fingers_g_off on 9 teachers); then rescore run 3's checkpoints with the gated G at k = 0 and 3 (GPU 0).
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v2
V2=/usr1/e2s/v2d_challenge_data/outputs/gmd/v2; G=/usr1/e2s/v2d_challenge_data/outputs/gmd/v3; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
until [[ -f $V2/g/g9v2off.npz ]] && ! docker ps --format '{{.Names}}' | grep -q gmd_train_goff; do sleep 15; done
$PY -c "
import numpy as np
on=dict(np.load('$V2/g/g9v2.npz')); off=dict(np.load('$V2/g/g9v2off.npz'))
out=dict(on); out.update({'off_'+k:v for k,v in off.items()})
np.savez('$G/g/g9gated.npz',**out); np.savez('$V2/g/g9gated.npz',**out); print('gated keys',len(out))"
echo "gated G built $(date -u)" >> $G/rescore/gated.log
for e in 35 36 41 40 18 2 3 26 8; do $PY -c "import numpy as np;p=dict(np.load('$V2/g/g9gated.npz'));np.savez('$V2/g/g9gated_for_ep$e.npz',engage_timeline=np.load('$V2/ep$e/engage_timeline.npy'),**p)"; done
for e in 35 36 41 40 18 2 3 26 8; do for S in fingers_g fingers_g_off; do
  while [ $(docker ps --format '{{.Names}}' | grep -c "gmd_gval_") -ge 4 ]; do sleep 5; done
  MEM=8g timeout 1800 ./run_fc.sh 0 $e gval_${e}_$S -- python /env/gmd/rollout.py $(python3 episodes.py ckpt $e) /out/v2/ep$e $S --g /out/v2/g/g9gated_for_ep$e.npz --tag _g9gated > $G/rescore/gval_${e}_$S.log 2>&1 &
  sleep 5
done; done
wait
echo "gated validation done $(date -u)" >> $G/rescore/gated.log
./rescore.sh 0 g9gated 0 3
echo "gated rescore done $(date -u)" >> $G/rescore/gated.log
