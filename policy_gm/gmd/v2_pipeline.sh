#!/usr/bin/env bash
# GMD run 2, end to end: v2 / best reconstructions, each episode's own mesh, best compatible S0 checkpoints, feedforward VOC.
#   1. record the 9 teachers (7 settings x 64 worlds), 4 containers at a time (2 per GPU); label each as soon as it is recorded
#   2. train G (g9v2) on all 9                          3. Gate 1 (teacher wrists + G), 4 at a time
#   4. stop here unless Gate 1 >= 90% of the teacher on >= 7 of 9
#   5. M on 9 episodes, one run per GPU at a time, both GPUs pulling from one shared queue (longest first)
#   6. videos (S0 vs M+G, real time) and SUMMARY.md / curves_all.png
# Outputs: outputs/gmd/v2/ (ep<N>/, g/, m/<run>/, videos/, logs/pipeline.log). Resumable: finished pieces are skipped.
set -uo pipefail
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v2 GMD_VOC=ff
O=/usr1/e2s/v2d_challenge_data/outputs; G=$O/gmd/v2; L=$G/logs; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
GN=g9v2; mkdir -p $L $G/g $G/videos
log() { echo "$(date -u '+%F %T') $*" | tee -a $L/pipeline.log; }
TEACH="35 36 41 40 18 2 3 26 8"                           # 2048-world / long episodes first
M_JOBS="A_ep41 B_ep37 A_ep26 A_ep3 B_ep6 B_ep15 B_ep30 C_ep19 C_ep28"
S="nominal perturb dart0.1 dart0.2 perturb_dart0.1 perturb_dart0.2 fingers_open"
export O G L PY GN S

# ---- 1. record + label --------------------------------------------------------------------------------------------------
rec() {  # <slot> <ep>
  local GPU=$(( $1 % 2 )) EP=$2
  [[ -f $G/ep$EP/fingers_open.json ]] || MEM=10g timeout 3600 ./run_fc.sh $GPU $EP rec$EP -- \
    python /env/gmd/rollout.py $(python3 episodes.py ckpt $EP) /out/v2/ep$EP $S --worlds 64 > $L/rec$EP.log 2>&1
  [[ -f $G/ep$EP/fingers_open.json ]] || { echo "ep$EP recording FAILED" >> $L/pipeline.log; return; }
  [[ -f $G/ep$EP/labels_summary.txt ]] || (nice $PY label.py $G/ep$EP > $L/label$EP.log 2>&1 &)
}
export -f rec
log "1. recording + labelling: $TEACH"
i=0; for e in $TEACH; do echo "$i $e"; i=$((i + 1)); done | xargs -P 4 -L 1 bash -c 'rec $(( $0 % 4 )) $1'
for e in $TEACH; do
  until [[ -f $G/ep$e/labels_summary.txt ]] || ! pgrep -f "label.py $G/ep$e" > /dev/null; do sleep 10; done
  [[ -f $G/ep$e/labels_summary.txt ]] || { log "ep$e labels missing, stopping"; exit 1; }
  log "ep$e nominal AUC $($PY -c "import json;print('%.3f'%json.load(open('$G/ep$e/nominal.json'))['add_auc'])")"
done

# ---- 2. G -----------------------------------------------------------------------------------------------------------------
if [[ ! -f $G/g/$GN.npz ]]; then
  log "2. training $GN"
  timeout 7200 ./run_fc.sh 0 41 train_$GN -- python /env/gmd/train_g.py $GN --train $(for e in $TEACH; do printf 'ep%s ' $e; done) \
    --steps 100000 --root /out/v2 > $L/train_$GN.log 2>&1
  [[ -f $G/g/$GN.npz ]] || { log "G training failed (see train_$GN.log)"; exit 1; }
fi

# ---- 3. Gate 1 -------------------------------------------------------------------------------------------------------------
log "3. Gate 1"
for e in $TEACH; do
  [[ -f $G/g/${GN}_for_ep$e.npz ]] || $PY -c "import numpy as np;p=dict(np.load('$G/g/$GN.npz'));np.savez('$G/g/${GN}_for_ep$e.npz',engage_timeline=np.load('$G/ep$e/engage_timeline.npy'),**p)"
done
gate() {  # <slot> <ep>
  local GPU=$(( $1 % 2 )) EP=$2
  [[ -f $G/ep$EP/fingers_g_$GN.json ]] || MEM=8g timeout 1800 ./run_fc.sh $GPU $EP gate$EP -- python /env/gmd/rollout.py \
    $(python3 episodes.py ckpt $EP) /out/v2/ep$EP fingers_g --g /out/v2/g/${GN}_for_ep$EP.npz --tag _$GN > $L/gate_$EP.log 2>&1
}
export -f gate
i=0; for e in $TEACH; do echo "$i $e"; i=$((i + 1)); done | xargs -P 4 -L 1 bash -c 'gate $(( $0 % 4 )) $1'
PASS=$($PY - <<EOF
import json
n = 0
out = open("$L/gate1.txt", "w")
for e in "$TEACH".split():
    try:
        r = json.load(open(f"$G/ep{e}/fingers_g_$GN.json"))["add_auc"] / json.load(open(f"$G/ep{e}/nominal.json"))["add_auc"]
    except FileNotFoundError:
        r = 0.0
    print(f"ep{e} gate1 {100 * r:.0f}%", file=out)
    n += r >= 0.9
print(n)
EOF
)
log "Gate 1: $PASS / 9 at >= 90% ($(tr '\n' ' ' < $L/gate1.txt))"
[[ $PASS -ge 7 ]] || { log "Gate 1 failed: stopping before M (decision needed)"; exit 1; }

# ---- 5. M runs -------------------------------------------------------------------------------------------------------------
log "5. M runs: $M_JOBS"
worker() {  # <gpu>: take the next unclaimed run until none is left
  local GPU=$1 R EP
  for R in $M_JOBS; do
    mkdir $L/claim_$R 2> /dev/null || continue
    [[ -f $G/m/$R/DONE ]] && continue
    EP=${R##*_ep}
    log "M $R starting on GPU $GPU"
    SUB=v2 BAND_AFTER=52428800 NOBEST_N=6 ./run_m.sh $EP $GPU $R $GN.npz > $L/run_m_$R.log 2>&1
    log "M $R finished: $(cat $G/m/$R/STOP 2>/dev/null || echo cap)"
  done
}
export -f worker log; export M_JOBS
worker 0 & sleep 90; worker 1 & wait
while docker ps --format '{{.Names}}' | grep -q "gmd_me_\|gmd_m_"; do sleep 30; done

# ---- 6. videos + summary ------------------------------------------------------------------------------------------------
log "6. videos + summary"
export GMD_ROOT=$G GMD_GNAME=$GN GMD_STOP="within 0.05 for 5 checkpoints and no new best in the last 6, after 52.4M"
export GMD_RUNS="A:seen:41 A:seen:26 A:seen:3 B:unseen:37 B:unseen:6 B:unseen:15 B:unseen:30 C:S0_weak:19 C:S0_weak:28"
export GMD_NOTE="Second run: v2 / best reconstructions, each episode's own mesh, best compatible S0 checkpoints, M trained with the feedforward VOC (S0 curves: old VOC)."
$PY summarize_full.py s0only >> $L/pipeline.log 2>&1
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
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP rs0_$EP -- "python /env/gmd/render_m.py evaluation.checkpoint=$(python3 episodes.py ckpt $EP) viewer.output_path=/out/v2/videos/ep${EP}_S0.mp4 $C $V" > $L/render_S0_$EP.log 2>&1
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP rmg_$EP -- "GMD_G=/out/v2/g/$GN.npz python /env/gmd/render_m.py evaluation.checkpoint=/out/v2/m/$R/run/policy_$BEST.safetensors viewer.output_path=/out/v2/videos/ep${EP}_MG.mp4 $C $V" > $L/render_MG_$EP.log 2>&1
  S0A=$($PY -c "import json;print('%.3f'%json.load(open('$G/summary_s0.json'))['$EP'])" 2>/dev/null || echo "?")
  MGA=$($PY -c "import json;r=[json.loads(l) for l in open('$G/m/$R/evals.jsonl')];b=max(r,key=lambda x:x['add_auc']);print('%.3f at %.1fM'%(b['add_auc'],b['step']/1e6))")
  ffmpeg -loglevel error -y -i $G/videos/ep${EP}_S0.mp4 -i $G/videos/ep${EP}_MG.mp4 -filter_complex \
    "[0]scale=640:360,drawtext=fontfile=$F:text='ep$EP S0 policy  AUC $S0A':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[a];[1]scale=640:360,drawtext=fontfile=$F:text='${R%%_*} M + frozen $GN  AUC $MGA':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[b];[a][b]hstack=2" \
    -c:v libx264 -crf 20 -pix_fmt yuv420p $G/videos/ep${EP}_M_vs_S0.mp4
}
(for r in A_ep41 A_ep3 B_ep6 B_ep30 C_ep28; do render 0 $r; done) & (for r in B_ep37 A_ep26 B_ep15 C_ep19; do render 1 $r; done); wait
$PY summarize_full.py >> $L/pipeline.log 2>&1
log "finished"
