#!/usr/bin/env bash
# E0 (standalone, cached-run HaMeR settings, --reference-frame auto) on every AssemblyHands and HOT3D clip; resumable.
# Run through tools/when_gpu_free.sh; CUDA_VISIBLE_DEVICES is set by it, the runner's --gpu is that index.
cd "$(dirname "$0")/.."
G=$1
for ds in assemblyhands hot3d; do
  for v in data/${ds}_input/*.mp4; do
    c=$(basename "$v" .mp4)
    [ -f data/e0_standalone/$ds/$c/.done/Interpolate_HaMeR_aligned_frames ] && continue
    methods/.venv-e0/bin/python tools/run_e0_standalone.py --video "$v" --intrinsics data/e0_inputs/$ds/$c.json \
      --out data/e0_standalone/$ds/$c --gpu "$G" --reference-frame auto --cached-track3-settings \
      > logs/e0_${ds}_$c.log 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "$ds/$c exit $rc"; exit $rc; }
    echo "$ds/$c done"
  done
done
