#!/usr/bin/env bash
# Label every episode once its recording is done (4 at a time).
G=/usr1/e2s/v2d_challenge_data/outputs/gmd/full; cd /usr1/e2s/retarget_env/gmd
PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
for EP in 41 40 26 18 2 35 8 3 36 38 6 37; do
  until grep -q "^ep$EP " $G/logs/rec_status.txt 2>/dev/null; do sleep 15; done
  [[ -f $G/ep$EP/labels_summary.txt ]] && continue
  while [ $(pgrep -fc "label.py $G") -ge 4 ]; do sleep 10; done
  nice $PY label.py $G/ep$EP > $G/logs/label$EP.log 2>&1 &
done
wait; echo done > $G/logs/label_done
