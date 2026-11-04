#!/usr/bin/env bash
# gmd_ft diagnostics on one GPU (outputs under outputs/gmd_ft/):
#   gate1/ep<N>/   G-vs-M test: the episode's S0 teacher drives the wrists, v4's G (g9gated) drives the fingers (64 worlds).
#                  nominal (teacher as is) and fingers_open (fingers frozen: floor) are recorded first, label.py gives the
#                  engage timeline, then fingers_g. Same rollout.py as tmurugan's Gate 1 (gmd/rollout.py, read-only).
#   videos/        ep<N>_S0_v4_joint_jointfrozen.mp4: 2 x 2 grid, S0 | v4 M+G / Exp 3 joint | Exp 4 joint with G frozen
#   diag_jobs.sh <gpu> <gate1|videos> <episode...>
set -uo pipefail
GPU=$1; JOB=$2; shift 2
export GMD_SET=v2
cd /usr1/e2s/gmd_ft
O=/usr1/e2s/v2d_challenge_data/outputs; FO=$O/gmd_ft; GM=/usr1/e2s/retarget_env/gmd; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
run() { case $1 in 6) echo B_ep6 ;; 15) echo B_ep15 ;; 3) echo A_ep3 ;; esac; }
best() { python3 -c "import json,sys;r=[json.loads(l) for l in open(sys.argv[1])];x=max(r,key=lambda d:d['add_auc']);print(x['step'], '%.3f' % x['add_auc'])" "$1"; }
eval "$(sed -n '/^hostpq()/,/^}/p' $GM/v2_pipeline.sh)"

for EP in "$@"; do
  R=$(run $EP)
  if [[ $JOB == gate1 ]]; then
    D=$FO/gate1/ep$EP; mkdir -p $D
    CK=$(cd $GM && python3 episodes.py ckpt $EP)
    MEM=10g timeout 2400 ./ft_fc.sh $GPU $EP g1rec_$EP -- "python /env/gmd/rollout.py $CK /out/gate1/ep$EP nominal fingers_open --worlds 64" > $D/rec.log 2>&1
    (cd $GM && $PY label.py $D nominal) > $D/label.log 2>&1
    $PY -c "import numpy as np;p=dict(np.load('$O/gmd/v4/g/g9gated.npz'));np.savez('$D/g9gated_for_ep$EP.npz',engage_timeline=np.load('$D/engage_timeline.npy'),**p)"
    MEM=10g timeout 2400 ./ft_fc.sh $GPU $EP g1g_$EP -- "python /env/gmd/rollout.py $CK /out/gate1/ep$EP fingers_g --g /out/gate1/ep$EP/g9gated_for_ep$EP.npz --worlds 64" > $D/fingers_g.log 2>&1
    echo "ep$EP $(for s in nominal fingers_open fingers_g; do python3 -c "import json;print('$s %.3f' % json.load(open('$D/$s.json'))['add_auc'])" 2>/dev/null || echo "$s -"; done | tr '\n' ' ')" | tee -a $FO/gate1/results.txt
  else
    V="viewer.backend=mp4 viewer.video_fps=20 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra"
    mkdir -p $FO/videos
    read -r CP CT NF < <(cd $GM && O=$O $PY cam_parquet.py $(cd $GM && hostpq $EP) 0.5 0.25)
    for X in j jo; do
      read -r B A < <(best $FO/${X}_$R/evals.jsonl); eval "A_$X=$A"
      U=50000000; [[ $X == jo ]] && U=1000000000000000
      MEM=12g timeout 900 ./ft_fc.sh $GPU $EP r${X}_$EP -- "GMD_G=/gmd/v4/g/g9gated.npz FT_ENV=joint FJ_UNFREEZE=$U python /ft/render_ft.py evaluation.checkpoint=/out/${X}_$R/run/policy_$B.safetensors viewer.output_path=/out/videos/ep${EP}_${X}.mp4 'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' viewer.num_frames=$((NF - 1)) $V" > $FO/videos/render_${X}_$EP.log 2>&1
    done
    S0V=$O/gmd/v2/videos/ep${EP}_S0.mp4; [[ $EP == 15 ]] && S0V=$O/gmd/v2/debug/ep15_S0_rep2.mp4
    S0A=$(python3 -c "import json;print('%.3f'%json.load(open('$O/gmd/v4/summary_s0.json'))['$EP'])")
    read -r _ V4A < <(best $O/gmd/v4/m/$R/evals.jsonl)
    F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf; T="x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6"
    ffmpeg -loglevel error -y -i $S0V -i $O/gmd/v4/videos/ep${EP}_MG4.mp4 -i $FO/videos/ep${EP}_j.mp4 -i $FO/videos/ep${EP}_jo.mp4 -filter_complex \
      "[0]fps=20,scale=640:360,drawtext=fontfile=$F:text='ep$EP S0  AUC $S0A':$T[a];[1]fps=20,scale=640:360,drawtext=fontfile=$F:text='v4 M + frozen G  $V4A':$T[b];[2]fps=20,scale=640:360,drawtext=fontfile=$F:text='Exp 3 joint training  $A_j':$T[c];[3]fps=20,scale=640:360,drawtext=fontfile=$F:text='Exp 4 joint setup, G frozen  $A_jo':$T[d];[a][b]hstack=2:shortest=1[top];[c][d]hstack=2:shortest=1[bot];[top][bot]vstack=2:shortest=1" \
      -c:v libx264 -crf 20 -pix_fmt yuv420p $FO/videos/ep${EP}_S0_v4_joint_jointfrozen.mp4 && echo "ep$EP video done" >> $FO/videos/status.txt
  fi
done
