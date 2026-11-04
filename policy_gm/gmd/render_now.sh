#!/usr/bin/env bash
# Render finished v2 M runs while other runs train: one render at a time, JAX memory capped, on the GPU with most free memory.
#   render_now.sh <run> ...      (same render() as v2_pipeline.sh; the pipeline re-renders everything at the end)
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v2 GMD_VOC=ff
O=/usr1/e2s/v2d_challenge_data/outputs; G=$O/gmd/v2; L=$G/logs; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python; GN=g9v2
hostpq() {  # host parquet dir of an episode
  local EP=$1 RC; RC=$(python3 episodes.py recon $EP)
  case $RC in
    v2recon) echo $O/ego2sim_recon_v2_pkg/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave ;;
    v2clean) echo $O/ego2sim_recon_v2clean_pkg/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave ;;
    v1) echo $O/rian_eval_compare/_cleaned/package/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave ;;
    *) echo $O/rl_train_req/assets$(python3 episodes.py parquet $EP | sed 's|^/v2d/assets||') ;;
  esac
}
V="viewer.backend=mp4 viewer.video_fps=20 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra"
render() {  # <gpu> <run>
  local GPU=$1 R=$2 EP=${2##*_ep} BEST CP CT NF F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf S0A MGA
  [[ -f $G/m/$R/evals.jsonl ]] || return
  BEST=$($PY -c "import json;r=[json.loads(l) for l in open('$G/m/$R/evals.jsonl')];print(max(r,key=lambda x:x['add_auc'])['step'])")
  echo "$BEST" > $G/m/$R/best_checkpoint.txt
  read -r CP CT NF < <($PY cam_parquet.py $(hostpq $EP) 0.5 0.25)
  local C="'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' viewer.num_frames=$((NF - 1))"
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP rs0_$EP -- "XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python /env/gmd/render_m.py evaluation.checkpoint=$(python3 episodes.py ckpt $EP) viewer.output_path=/out/v2/videos/ep${EP}_S0.mp4 $C $V" > $L/render_S0_$EP.log 2>&1
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP rmg_$EP -- "XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 GMD_G=/out/v2/g/$GN.npz python /env/gmd/render_m.py evaluation.checkpoint=/out/v2/m/$R/run/policy_$BEST.safetensors viewer.output_path=/out/v2/videos/ep${EP}_MG.mp4 $C $V" > $L/render_MG_$EP.log 2>&1
  S0A=$($PY -c "import json;print('%.3f'%json.load(open('$G/summary_s0.json'))['$EP'])" 2>/dev/null || echo "?")
  MGA=$($PY -c "import json;r=[json.loads(l) for l in open('$G/m/$R/evals.jsonl')];b=max(r,key=lambda x:x['add_auc']);print('%.3f at %.1fM'%(b['add_auc'],b['step']/1e6))")
  ffmpeg -loglevel error -y -i $G/videos/ep${EP}_S0.mp4 -i $G/videos/ep${EP}_MG.mp4 -filter_complex \
    "[0]scale=640:360,drawtext=fontfile=$F:text='ep$EP S0 policy  AUC $S0A':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[a];[1]scale=640:360,drawtext=fontfile=$F:text='${R%%_*} M + frozen $GN  AUC $MGA':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[b];[a][b]hstack=2" \
    -c:v libx264 -crf 20 -pix_fmt yuv420p $G/videos/ep${EP}_M_vs_S0.mp4
}
for R in "$@"; do
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
  [[ $FREE -ge 2500 ]] || { echo "$R: only ${FREE} MiB free, skipped"; continue; }
  echo "$R on GPU $GPU (${FREE} MiB free)"
  render $GPU $R
  ls -la $G/videos/ep${R##*_ep}_M_vs_S0.mp4 2>/dev/null || echo "$R: render failed (see $L/render_*_${R##*_ep}.log)"
done
