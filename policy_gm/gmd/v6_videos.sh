#!/usr/bin/env bash
# Comparison videos for run 6 as each M run finishes (instead of only at the end): both GPUs are held (run_m.sh hold files),
# so when a run ends its GPU stays free; every finished run without a video is rendered there (S0 | [v2 M+G] | [v5 M+G] |
# v6 M+G, whichever exist) plus a gdiag rollout, then the hold is lifted until the next run has started on that GPU.
#   v6_videos.sh        outputs: v6/videos/diag/ep<N>_compare.mp4, v6/logs/gdiag_<run>.npz; stops when the M queue is done
set -uo pipefail
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v5
O=/usr1/e2s/v2d_challenge_data/outputs; G=$O/gmd/v6; V5=$O/gmd/v5; V2=$O/gmd/v2; L=$G/logs; D=$G/videos/diag
PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
M_JOBS="B_ep37 C_ep19 C_ep16 C_ep5 C_ep20 C_ep13"
mkdir -p $D
log() { echo "$(date -u '+%F %T') [videos] $*" | tee -a $L/pipeline.log; }
eval "$(sed -n '/^hostpq()/,/^}/p' v6_pipeline.sh)"
V="viewer.backend=mp4 viewer.video_fps=20 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra"
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
T="x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6"
best() { $PY -c "import json;r=[json.loads(l) for l in open('$1')];b=max(r,key=lambda x:x['add_auc']);print(b['step'],'%.3f'%b['add_auc'])"; }
s0auc() { python3 -c "
import glob,json,sys; sys.path.insert(0,'.'); from episodes import S0
d=S0[$1]; ph=d[4] if len(d)>4 else 'p1_96M'
print('%.3f'%max(json.load(open(f))['metrics']['add_auc'] for f in glob.glob('$O/rl_train_req/training_s0/'+d[0]+'/'+ph+'/eval/metrics_*.json')))"; }
free_gpu() { local g; for g in 0 1; do [[ $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1500 ]] && { echo $g; return; }; done; }

render() {  # <gpu> <run>
  local GPU=$1 R=$2 EP=${2##*_ep} B6 A6 B5 A5 B2 A2 CP CT NF C S0V S0A IN=() LB=() i n f
  read -r B6 A6 < <(best $G/m/$R/evals.jsonl)
  read -r CP CT NF < <($PY cam_parquet.py $(hostpq $EP) 0.5 0.25)
  C="'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' viewer.num_frames=$((NF - 1))"
  S0V=$D/ep${EP}_S0.mp4
  [[ -f $V5/videos/diag/ep${EP}_S0.mp4 ]] && S0V=$V5/videos/diag/ep${EP}_S0.mp4
  [[ $EP == 37 ]] && S0V=$V2/videos/ep37_S0.mp4
  [[ -f $S0V ]] || MEM=12g timeout 900 ./run_fc.sh $GPU $EP vrs0_$EP -- "python /env/gmd/render_m.py evaluation.checkpoint=$(python3 episodes.py ckpt $EP) viewer.output_path=/out/v6/videos/diag/ep${EP}_S0.mp4 $C $V" > $L/vid_S0_$EP.log 2>&1
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP vrmg_$EP -- "GMD_G=/out/v6/g/g17.npz python /env/gmd/render_m.py evaluation.checkpoint=/out/v6/m/$R/run/policy_$B6.safetensors viewer.output_path=/out/v6/videos/diag/ep${EP}_MG6.mp4 $C $V" > $L/vid_MG_$EP.log 2>&1
  MEM=12g timeout 1200 ./run_fc.sh $GPU $EP vgd_$EP -- "GMD_G=/out/v6/g/g17.npz python /env/gmd/gdiag.py /out/v6/m/$R/run/policy_$B6.safetensors /out/v6/logs/gdiag_$R.npz --worlds 16" > $L/gdiag_$R.log 2>&1
  IN=($S0V); LB=("ep$EP S0  AUC $(s0auc $EP)")
  if [[ $EP == 37 ]]; then read -r B2 A2 < <(best $V2/m/B_ep37/evals.jsonl); IN+=($V2/videos/ep37_MG.mp4); LB+=("v2 M + g9v2  AUC $A2"); fi
  f=$(ls $V5/m/*_ep$EP/evals.jsonl 2>/dev/null | head -1)
  if [[ -n $f && -f $(dirname $f)/DONE && -f $V5/videos/diag/ep${EP}_MG5.mp4 ]]; then
    read -r B5 A5 < <(best $f); IN+=($V5/videos/diag/ep${EP}_MG5.mp4); LB+=("v5 M + g12  AUC $A5"); fi
  IN+=($D/ep${EP}_MG6.mp4); LB+=("v6 M + g17  AUC $A6")
  n=${#IN[@]}; local args=() fc=""
  for i in $(seq 0 $((n - 1))); do args+=(-i "${IN[$i]}"); fc+="[$i]fps=20,scale=640:360,drawtext=fontfile=$F:text='${LB[$i]}':$T[v$i];"; done
  if [[ $n -eq 4 ]]; then fc+="[v0][v1]hstack=2:shortest=1[t];[v2][v3]hstack=2:shortest=1[b];[t][b]vstack=2:shortest=1"
  else fc+="$(for i in $(seq 0 $((n - 1))); do printf '[v%s]' $i; done)hstack=$n:shortest=1"; fi
  ffmpeg -loglevel error -y "${args[@]}" -filter_complex "$fc" -c:v libx264 -crf 20 -pix_fmt yuv420p $D/ep${EP}_compare.mp4
  log "ep$EP: $D/ep${EP}_compare.mp4 (${LB[*]})"
}

waiting_on() { pgrep -f "run_m.sh [0-9]+ $1 " > /dev/null; }             # a run on GPU <g> sits at the hold
pending() { local R; for R in $M_JOBS; do [[ -f $G/m/$R/DONE && ! -f $D/ep${R##*_ep}_compare.mp4 ]] && echo $R; done; }
release() {  # <gpu>: let the run waiting on it start, then hold again so its end is caught too
  local g=$1 i
  waiting_on $g || return
  rm -f $L/hold_gpu$g
  for i in $(seq 40); do [[ $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) -gt 3000 ]] && break; sleep 15; done
  touch $L/hold_gpu$g
}

touch $L/hold_gpu0 $L/hold_gpu1
log "watching: a video per finished M run (GPUs held between runs)"
while :; do
  P=($(pending))
  if [[ ${#P[@]} -gt 0 ]]; then
    g=$(free_gpu)
    [[ -n $g ]] && { for R in "${P[@]}"; do render $g $R; done; release $g; }
  fi
  for g in 0 1; do                                  # idle GPU, nothing left to render: start its waiting run
    [[ -z $(pending) && $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1500 ]] && release $g
  done
  done_all=1; for R in $M_JOBS; do [[ -f $G/m/$R/DONE ]] || done_all=0; done
  [[ $done_all -eq 1 && -z $(pending) && $(docker ps --format '{{.Names}}' | grep -c "^gmd_m_") -eq 0 ]] && break
  sleep 30
done
rm -f $L/hold_gpu0 $L/hold_gpu1
log "all M runs rendered"
