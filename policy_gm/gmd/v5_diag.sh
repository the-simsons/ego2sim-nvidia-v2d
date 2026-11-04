#!/usr/bin/env bash
# Run 5 diagnostics for selected episodes on the first GPU the M queue frees (hold files in v5/logs, see run_m.sh):
# S0 vs v5 M+G videos (ep37 also with run 2's M+G), and gdiag rollouts (16 worlds) of the v5 runs and of run 2's ep37.
#   v5_diag.sh <run> ...     e.g. B_ep37 B_ep38 A_ep2 A_ep1      outputs: v5/videos/diag/, v5/logs/gdiag_<run>.npz
set -uo pipefail
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v5
O=/usr1/e2s/v2d_challenge_data/outputs; G=$O/gmd/v5; V2=$O/gmd/v2; L=$G/logs; D=$G/videos/diag; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
mkdir -p $D
log() { echo "$(date -u '+%F %T') [diag] $*" | tee -a $L/pipeline.log; }
touch $L/hold_gpu0 $L/hold_gpu1
log "holding both GPUs; diagnostics run on the first one the M queue frees"
GPU=""
until [[ -n $GPU ]]; do
  for g in 0 1; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ $used -lt 1500 ]] && { GPU=$g; break; }
  done
  [[ -n $GPU ]] || sleep 30
done
rm -f $L/hold_gpu$(( 1 - GPU ))
log "GPU $GPU free: rendering $*"
eval "$(sed -n '/^hostpq()/,/^}/p' v5_pipeline.sh)"
V="viewer.backend=mp4 viewer.video_fps=20 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra"
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
T="x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6"
best() { $PY -c "import json;r=[json.loads(l) for l in open('$1')];b=max(r,key=lambda x:x['add_auc']);print(b['step'],'%.3f'%b['add_auc'])"; }
for R in "$@"; do
  EP=${R##*_ep}
  read -r B5 A5 < <(best $G/m/$R/evals.jsonl)
  read -r CP CT NF < <($PY cam_parquet.py $(hostpq $EP) 0.5 0.25)
  C="'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' viewer.num_frames=$((NF - 1))"
  S0V=$D/ep${EP}_S0.mp4
  [[ $EP == 37 ]] && S0V=$V2/videos/ep37_S0.mp4                       # same S0 checkpoint and reconstruction as run 2
  [[ -f $S0V ]] || MEM=12g timeout 900 ./run_fc.sh $GPU $EP drs0_$EP -- "python /env/gmd/render_m.py evaluation.checkpoint=$(python3 episodes.py ckpt $EP) viewer.output_path=/out/v5/videos/diag/ep${EP}_S0.mp4 $C $V" > $L/diag_render_S0_$EP.log 2>&1
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP drmg_$EP -- "GMD_G=/out/v5/g/g12.npz python /env/gmd/render_m.py evaluation.checkpoint=/out/v5/m/$R/run/policy_$B5.safetensors viewer.output_path=/out/v5/videos/diag/ep${EP}_MG5.mp4 $C $V" > $L/diag_render_MG_$EP.log 2>&1
  MEM=12g timeout 1200 ./run_fc.sh $GPU $EP dgd_$EP -- "GMD_G=/out/v5/g/g12.npz python /env/gmd/gdiag.py /out/v5/m/$R/run/policy_$B5.safetensors /out/v5/logs/gdiag_$R.npz --worlds 16" > $L/gdiag_$R.log 2>&1
  S0A=$(python3 -c "
import glob,json,sys; sys.path.insert(0,'.'); from episodes import S0
d=S0[$EP]; ph=d[4] if len(d)>4 else 'p1_96M'
print('%.3f'%max(json.load(open(f))['metrics']['add_auc'] for f in glob.glob('$O/rl_train_req/training_s0/'+d[0]+'/'+ph+'/eval/metrics_*.json')))")
  if [[ $EP == 37 ]]; then
    read -r B2 A2 < <(best $V2/m/B_ep37/evals.jsonl)
    MEM=12g timeout 1200 ./run_fc.sh $GPU 37 dgd2_37 -- "GMD_G=/out/v2/g/g9v2.npz python /env/gmd/gdiag.py /out/v2/m/B_ep37/run/policy_$B2.safetensors /out/v5/logs/gdiag_v2_B_ep37.npz --worlds 16" > $L/gdiag_v2_B_ep37.log 2>&1
    ffmpeg -loglevel error -y -i $S0V -i $V2/videos/ep37_MG.mp4 -i $D/ep37_MG5.mp4 -filter_complex \
      "[0]fps=20,scale=640:360,drawtext=fontfile=$F:text='ep37 S0  AUC $S0A':$T[a];[1]fps=20,scale=640:360,drawtext=fontfile=$F:text='v2 M + g9v2  AUC $A2':$T[b];[2]fps=20,scale=640:360,drawtext=fontfile=$F:text='v5 M + g12  AUC $A5':$T[c];[a][b][c]hstack=3:shortest=1" \
      -c:v libx264 -crf 20 -pix_fmt yuv420p $D/ep37_S0_v2_v5.mp4
  else
    ffmpeg -loglevel error -y -i $S0V -i $D/ep${EP}_MG5.mp4 -filter_complex \
      "[0]fps=20,scale=640:360,drawtext=fontfile=$F:text='ep$EP S0  AUC $S0A':$T[a];[1]fps=20,scale=640:360,drawtext=fontfile=$F:text='v5 M + g12  AUC $A5':$T[b];[a][b]hstack=2:shortest=1" \
      -c:v libx264 -crf 20 -pix_fmt yuv420p $D/ep${EP}_S0_v5.mp4
  fi
  log "ep$EP done: $(ls $D | grep -E "ep${EP}_S0_v" | tr '\n' ' ')"
done
rm -f $L/hold_gpu$GPU
log "released GPU $GPU"
