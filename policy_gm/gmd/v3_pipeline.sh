#!/usr/bin/env bash
# GMD run 3: M retrained with the causal-grasp-bit G (g9v2negB) on run 2's episodes; everything else as run 2 (v2/best
# reconstructions, own meshes, feedforward VOC, stop rule: >= 0.9 x3, or converged + no new best in 6 after 52.4M, cap 96.5M).
# ep26 = v2/m/X2B_ep26 (linked). 8 runs on a shared queue over both GPUs, then 3-panel videos (S0 | run 2 M+G | run 3 M+G),
# summary + curves, and a gdiag diagnostic of each best checkpoint (engage behaviour). Outputs: outputs/gmd/v3/.
set -uo pipefail
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v2 GMD_VOC=ff SUB=v3 BAND_AFTER=52428800 NOBEST_N=6
O=/usr1/e2s/v2d_challenge_data/outputs; G=$O/gmd/v3; V2=$O/gmd/v2; L=$G/logs; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
GN=g9v2negB
M_JOBS="B_ep37 A_ep41 B_ep6 A_ep3 B_ep15 B_ep30 C_ep19 C_ep28"
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
log "run 3: M with $GN on $M_JOBS (+ A_ep26 from v2/X2B_ep26)"
worker 0 & sleep 90; worker 1 & wait
while docker ps --format '{{.Names}}' | grep -q "gmd_me_\|gmd_m_"; do sleep 30; done
log "videos, summary, diagnostics"
export GMD_ROOT=$G GMD_GNAME=$GN GMD_STOP="within 0.05 for 5 checkpoints and no new best in the last 6, after 52.4M"
export GMD_RUNS="A:seen:41 A:seen:26 A:seen:3 B:unseen:37 B:unseen:6 B:unseen:15 B:unseen:30 C:S0_weak:19 C:S0_weak:28"
export GMD_NOTE="Run 3: M retrained with g9v2negB (grasp bit causal: engage counterfactuals + contact dropout + spoon oversampling); otherwise as run 2."
$PY summarize_full.py s0only >> $L/pipeline.log 2>&1
eval "$(sed -n '/^hostpq()/,/^}/p' v2_pipeline.sh)"
V="viewer.backend=mp4 viewer.video_fps=20 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra"
render() {  # <gpu> <run>: run 3 M+G render with run 2's camera, composed with run 2's S0 and M+G videos
  local GPU=$1 R=$2 EP=${2##*_ep} BEST CP CT NF F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf A2 A3 S0A S0V
  [[ -f $G/m/$R/evals.jsonl ]] || return
  BEST=$($PY -c "import json;r=[json.loads(l) for l in open('$G/m/$R/evals.jsonl')];print(max(r,key=lambda x:x['add_auc'])['step'])")
  echo "$BEST" > $G/m/$R/best_checkpoint.txt 2>/dev/null
  read -r CP CT NF < <($PY cam_parquet.py $(hostpq $EP) 0.5 0.25)
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP r3_$EP -- "GMD_G=/out/v3/g/$GN.npz python /env/gmd/render_m.py evaluation.checkpoint=/out/v3/m/$R/run/policy_$BEST.safetensors viewer.output_path=/out/v3/videos/ep${EP}_MG3.mp4 'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' viewer.num_frames=$((NF - 1)) $V" > $L/render_$EP.log 2>&1
  S0V=$V2/videos/ep${EP}_S0.mp4; [[ $EP == 15 ]] && S0V=$V2/debug/ep15_S0_rep2.mp4      # run 2's ep15 S0 take was a rare failure
  S0A=$($PY -c "import json;print('%.3f'%json.load(open('$G/summary_s0.json'))['$EP'])")
  A2=$($PY -c "import json,glob;f=glob.glob('$V2/m/*_ep$EP/evals.jsonl');f=[x for x in f if '/X' not in x][0];print('%.3f'%max(json.loads(l)['add_auc'] for l in open(f)))")
  A3=$($PY -c "import json;print('%.3f'%max(json.loads(l)['add_auc'] for l in open('$G/m/$R/evals.jsonl')))")
  ffmpeg -loglevel error -y -i $S0V -i $V2/videos/ep${EP}_MG.mp4 -i $G/videos/ep${EP}_MG3.mp4 -filter_complex \
    "[0]fps=20,scale=640:360,drawtext=fontfile=$F:text='ep$EP S0  AUC $S0A':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[a];[1]fps=20,scale=640:360,drawtext=fontfile=$F:text='run 2 M+G (bit ignored)  $A2':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[b];[2]fps=20,scale=640:360,drawtext=fontfile=$F:text='run 3 M+G (bit causal)  $A3':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[c];[a][b][c]hstack=3:shortest=1" \
    -c:v libx264 -crf 20 -pix_fmt yuv420p $G/videos/ep${EP}_S0_vs_run2_vs_run3.mp4
  local GF=/out/v3/g/$GN.npz
  MEM=12g timeout 1200 ./run_fc.sh $GPU $EP gd3_$EP -- "GMD_G=$GF python /env/gmd/gdiag.py /out/v3/m/$R/run/policy_$BEST.safetensors /out/v3/logs/gdiag_$R.npz --worlds 16" > $L/gdiag_$R.log 2>&1
}
(for r in B_ep37 B_ep6 B_ep30 C_ep28; do render 0 $r; done) & (for r in A_ep41 A_ep3 B_ep15 C_ep19; do render 1 $r; done); wait
cp $O/gmd/v2/exp/videos/ep26_S0_vs_v2_vs_X2B.mp4 $G/videos/ep26_S0_vs_run2_vs_run3.mp4
$PY summarize_full.py >> $L/pipeline.log 2>&1
log "finished"
