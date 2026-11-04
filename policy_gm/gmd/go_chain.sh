#!/usr/bin/env bash
cd /usr1/e2s/retarget_env/gmd
until grep -q "all done" /usr1/e2s/v2d_challenge_data/outputs/gmd/full/gonly/logs/go_all.log 2>/dev/null; do sleep 30; done
./go_render.sh
