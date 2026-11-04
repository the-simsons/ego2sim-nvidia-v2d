#!/usr/bin/env bash
# GMD run 4: M retrained with the gated G (g9gated: g9v2 while engage is on, g9v2off while it is off) on run 3's 9 episodes
# (ep26 trained too this time). Otherwise as run 3: v2/best reconstructions, own meshes, feedforward VOC, stop rule >= 0.9 x3,
# or converged + no new best in 6 after 52.4M; cap 81.8M (39 x 2097152, the first checkpoint at or past 80M).
# 9 runs on a shared queue over both GPUs, then 3-panel videos (S0 | run 3 M+G | run 4 M+G), summary + curves, gdiag.
# Outputs: outputs/gmd/v4/.
set -uo pipefail
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v2 GMD_VOC=ff SUB=v4 BAND_AFTER=52428800 NOBEST_N=6 TOTAL=81788928
O=/usr1/e2s/v2d_challenge_data/outputs; G=$O/gmd/v4; V2=$O/gmd/v2; V3=$O/gmd/v3; L=$G/logs; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
GN=g9gated; mkdir -p $L $G/g $G/m $G/videos
[[ -f $G/g/$GN.npz ]] || cp $V3/g/$GN.npz $G/g/$GN.npz
M_JOBS="B_ep37 A_ep41 A_ep26 B_ep6 A_ep3 B_ep15 B_ep30 C_ep19 C_ep28"
log() { echo "$(date -u '+%F %T') $*" | tee -a $L/pipeline.log; }
worker() {
  local GPU=$1 R EP
  for R in $M_JOBS; do
    mkdir $L/claim_$R 2> /dev/null || continue
    [[ -f $G/m/$R/DONE ]] && continue
    EP=${R##*_ep}
    log "M $R starting on GPU $GPU"
    ./run_m.sh $EP $GPU $R $GN.npz > $L/run_m_$R.log 2>&1
    log "M $R finished: $(cat $G/m/$R/STOP 2>/dev/null || echo cap)"
  done
}
export -f worker log; export M_JOBS
log "run 4: M with $GN on $M_JOBS, cap $TOTAL"
worker 0 & sleep 90; worker 1 & wait
while docker ps --format '{{.Names}}' | grep -q "gmd_me_\|gmd_m_"; do sleep 30; done
log "videos, summary, diagnostics"
export GMD_ROOT=$G GMD_GNAME=$GN GMD_STOP="within 0.05 for 5 checkpoints and no new best in the last 6, after 52.4M; cap 81.8M"
export GMD_RUNS="A:seen:41 A:seen:26 A:seen:3 B:unseen:37 B:unseen:6 B:unseen:15 B:unseen:30 C:S0_weak:19 C:S0_weak:28"
export GMD_NOTE="Run 4: M retrained with the gated G (g9gated: g9v2 when engage on, g9v2off when off); otherwise as run 3, cap 81.8M."
$PY summarize_full.py s0only >> $L/pipeline.log 2>&1
eval "$(sed -n '/^hostpq()/,/^}/p' v2_pipeline.sh)"
V="viewer.backend=mp4 viewer.video_fps=20 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra"
render() {  # <gpu> <run>: run 4 M+G render with run 2's camera, composed with S0 and run 3's M+G videos
  local GPU=$1 R=$2 EP=${2##*_ep} BEST CP CT NF F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf A3 A4 S0A S0V R3V R3E
  [[ -f $G/m/$R/evals.jsonl ]] || return
  BEST=$($PY -c "import json;r=[json.loads(l) for l in open('$G/m/$R/evals.jsonl')];print(max(r,key=lambda x:x['add_auc'])['step'])")
  echo "$BEST" > $G/m/$R/best_checkpoint.txt
  read -r CP CT NF < <($PY cam_parquet.py $(hostpq $EP) 0.5 0.25)
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP r4_$EP -- "GMD_G=/out/v4/g/$GN.npz python /env/gmd/render_m.py evaluation.checkpoint=/out/v4/m/$R/run/policy_$BEST.safetensors viewer.output_path=/out/v4/videos/ep${EP}_MG4.mp4 'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' viewer.num_frames=$((NF - 1)) $V" > $L/render_$EP.log 2>&1
  S0V=$V2/videos/ep${EP}_S0.mp4; [[ $EP == 15 ]] && S0V=$V2/debug/ep15_S0_rep2.mp4
  R3V=$V3/videos/ep${EP}_MG3.mp4; R3E=$V3/m/$R/evals.jsonl
  [[ $EP == 26 ]] && { R3V=$V2/exp/videos/ep26_X2B.mp4; R3E=$V2/m/X2B_ep26/evals.jsonl; }
  S0A=$($PY -c "import json;print('%.3f'%json.load(open('$G/summary_s0.json'))['$EP'])")
  A3=$($PY -c "import json;print('%.3f'%max(json.loads(l)['add_auc'] for l in open('$R3E')))")
  A4=$($PY -c "import json;print('%.3f'%max(json.loads(l)['add_auc'] for l in open('$G/m/$R/evals.jsonl')))")
  ffmpeg -loglevel error -y -i $S0V -i $R3V -i $G/videos/ep${EP}_MG4.mp4 -filter_complex \
    "[0]fps=20,scale=640:360,drawtext=fontfile=$F:text='ep$EP S0  AUC $S0A':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[a];[1]fps=20,scale=640:360,drawtext=fontfile=$F:text='run 3 M + g9v2negB  $A3':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[b];[2]fps=20,scale=640:360,drawtext=fontfile=$F:text='run 4 M + gated G  $A4':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[c];[a][b][c]hstack=3:shortest=1" \
    -c:v libx264 -crf 20 -pix_fmt yuv420p $G/videos/ep${EP}_S0_vs_run3_vs_run4.mp4
  MEM=12g timeout 1200 ./run_fc.sh $GPU $EP gd4_$EP -- "GMD_G=/out/v4/g/$GN.npz python /env/gmd/gdiag.py /out/v4/m/$R/run/policy_$BEST.safetensors /out/v4/logs/gdiag_$R.npz --worlds 16" > $L/gdiag_$R.log 2>&1
}
(for r in B_ep37 B_ep6 B_ep30 C_ep28 A_ep26; do render 0 $r; done) & (for r in A_ep41 A_ep3 B_ep15 C_ep19; do render 1 $r; done); wait
$PY summarize_full.py >> $L/pipeline.log 2>&1
log "finished"
