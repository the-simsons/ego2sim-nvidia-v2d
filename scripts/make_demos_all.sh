#!/usr/bin/env bash
# Build the three showcase demo videos (best checkpoint per episode) and a contact sheet of one frame from each.
S="egocentric video -> AgentSTAR mesh + FoundationPose track + WiLoR/MANO hands -> retarget -> FlashSAC policy (Sharpa hands, FlashCHORD)"
R=~/v2d/runs
bash ~/v2d/scripts/make_demo.sh 38 $R/e2s_ep38_40M_eval 37748736 "Ego2Sim  |  eval episode 38: dust pan" "$S" 1 2>&1 | grep -- "->"
bash ~/v2d/scripts/make_demo.sh 6 $R/e2s_ep6_40M_eval 37748736 "Ego2Sim  |  eval episode 6: saucepan + wooden spoon" "$S" 1 2>&1 | grep -- "->"
bash ~/v2d/scripts/make_demo.sh 15 $R/e2s_ep15_40M_eval2 58720256 "Ego2Sim  |  eval episode 15: wooden spoon into dish rack" "$S" 1 2>&1 | grep -- "->"
~/v2d/envs/v2d_sam2/.venv/bin/python - <<'PY'
import cv2, numpy as np
out = []
for e, i in ((38, 120), (6, 110), (15, 120)):
    c = cv2.VideoCapture(f"/home/rborah/v2d/videos/demo_ep{e}.mp4"); c.set(1, i); ok, f = c.read()
    out.append(cv2.resize(f, (1328, 361)))
cv2.imwrite("/tmp/demos.jpg", np.vstack(out))
PY
ls -la ~/v2d/videos/*.mp4
