#!/usr/bin/env bash
# Run M trainings back to back on one GPU.   m_queue.sh <gpu> <episode:name> ...
GPU=$1; shift; cd /usr1/e2s/retarget_env/gmd
for J in "$@"; do ./run_m.sh ${J%%:*} $GPU ${J#*:} g9.npz; done
