#!/usr/bin/env bash
# GMD run 6: run 5's recipe (first-version G+M: plain behaviour cloning of G; M with the S0 recipe, feedforward VOC, stop
# >= 0.9 x3 or converged + no new best in 6 after 52.4M, cap 96.5M; per-world v1/v2 meshes where they differ) with 5 more
# teachers moved from run 5's unseen set (38 30 6 32 15) -> 17 teachers; 6 unseen episodes left for M.
#   0. meshmix specs: run 5's (outputs/gmd/v5/meshmix)
#   1. teachers: run 5's 12 recordings linked (same checkpoints); 38 / 30 / 6 / 32 / 15 recorded (38 30 15 mixed meshes) + labelled
#   2. train G (g17) on the 17 teachers                    3. Gate 1 (teacher wrists + G), reported, not blocking
#   4. M on the 6 unseen episodes (37 5 16 19 20 13), both GPUs pulling from one queue (2048-world episodes first)
#   5. summary, curves, videos (S0 vs M+G, real time; M+G scored and rendered on the episode's own v2 mesh)
# Outputs: outputs/gmd/v6/ (ep<N>/, g/, m/<run>/, videos/, logs/pipeline.log). Resumable: finished pieces are skipped.
set -uo pipefail
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v5 GMD_VOC=ff                                   # run 5's episode table (same S0 checkpoints)
O=/usr1/e2s/v2d_challenge_data/outputs; G=$O/gmd/v6; V5=$O/gmd/v5; L=$G/logs; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
GN=g17; mkdir -p $L $G/g $G/videos
log() { echo "$(date -u '+%F %T') $*" | tee -a $L/pipeline.log; }
TEACH="35 36 41 40 18 2 3 26 8 1 14 28 38 30 6 32 15"
REUSE="35 36 41 40 18 2 3 26 8 1 14 28"
NEWT="38 30 6 32 15"
MIX="14 15 16 19 20 30 38"
M_JOBS="B_ep37 C_ep19 C_ep16 C_ep5 C_ep20 C_ep13"
S="nominal perturb dart0.1 dart0.2 perturb_dart0.1 perturb_dart0.2 fingers_open"
mix() { [[ " $MIX " == *" $1 "* ]] && echo "GMD_MESHMIX=/out/v5/meshmix/ep$1.npz"; }
export O G V5 L PY GN S MIX
export -f mix log

# CUDA must initialise (2026-11-03: fragmented host RAM made cuInit fail machine-wide; fix = drop caches + compact memory)
python3 -c "import ctypes,sys;sys.exit(ctypes.CDLL('libcuda.so.1').cuInit(0))" || { log "cuInit fails on the host: not starting (see /proc/buddyinfo)"; exit 1; }

# ---- 0. meshmix specs -----------------------------------------------------------------------------------------------------
for e in $MIX; do [[ -f $V5/meshmix/ep$e.npz ]] || { log "run 5's meshmix spec for ep$e is missing"; exit 1; }; done

# ---- 1. teachers ----------------------------------------------------------------------------------------------------------
log "1. teachers: linking run 5's recordings of $REUSE; recording $NEWT"
for e in $REUSE; do
  mkdir -p $G/ep$e
  for f in meta.npz engage_timeline.npy labels_summary.txt; do [[ -f $G/ep$e/$f ]] || cp $V5/ep$e/$f $G/ep$e/; done
  for s in $S; do for f in $s.npz $s.json ${s}_data.npz; do
    [[ -e $V5/ep$e/$f && ! -e $G/ep$e/$f ]] && ln -s ../../v5/ep$e/$f $G/ep$e/$f
  done; done
done
rec() {  # <slot> <ep>
  local GPU=$(( $1 % 2 )) EP=$2
  [[ -f $G/ep$EP/fingers_open.json ]] || MEM=10g timeout 3600 ./run_fc.sh $GPU $EP rec$EP -- \
    "$(mix $EP) python /env/gmd/rollout.py $(python3 episodes.py ckpt $EP) /out/v6/ep$EP $S --worlds 64" > $L/rec$EP.log 2>&1
  [[ -f $G/ep$EP/fingers_open.json ]] || { log "ep$EP recording FAILED (rec$EP.log)"; return; }
  [[ -f $G/ep$EP/labels_summary.txt ]] || nice $PY label.py $G/ep$EP > $L/label$EP.log 2>&1
  $PY meshmix_report.py $G/ep$EP >> $L/pipeline.log 2>&1
}
export -f rec
i=0; for e in $NEWT; do echo "$i $e"; i=$((i + 1)); done | xargs -P 2 -L 1 bash -c 'rec $0 $1'
for e in $TEACH; do
  [[ -f $G/ep$e/labels_summary.txt ]] || { log "ep$e labels missing, stopping"; exit 1; }
  log "ep$e teacher nominal AUC $($PY -c "import json;print('%.3f'%json.load(open('$G/ep$e/nominal.json'))['add_auc'])")"
done

# ---- 2. G -----------------------------------------------------------------------------------------------------------------
if [[ ! -f $G/g/$GN.npz ]]; then
  log "2. training $GN (plain behaviour cloning, 100k steps) on $TEACH"
  timeout 10800 ./run_fc.sh 0 41 train_$GN -- python /env/gmd/train_g.py $GN --train $(for e in $TEACH; do printf 'ep%s ' $e; done) \
    --steps 100000 --root /out/v6 > $L/train_$GN.log 2>&1
  [[ -f $G/g/$GN.npz ]] || { log "G training failed (see train_$GN.log)"; exit 1; }
fi

# ---- 3. Gate 1 (reported) -------------------------------------------------------------------------------------------------
log "3. Gate 1"
for e in $TEACH; do
  [[ -f $G/g/${GN}_for_ep$e.npz ]] || $PY -c "import numpy as np;p=dict(np.load('$G/g/$GN.npz'));np.savez('$G/g/${GN}_for_ep$e.npz',engage_timeline=np.load('$G/ep$e/engage_timeline.npy'),**p)"
done
gate() {  # <slot> <ep>
  local GPU=$(( $1 % 2 )) EP=$2
  [[ -f $G/ep$EP/fingers_g_$GN.json ]] || MEM=8g timeout 1800 ./run_fc.sh $GPU $EP gate$EP -- "$(mix $EP) python /env/gmd/rollout.py \
    $(python3 episodes.py ckpt $EP) /out/v6/ep$EP fingers_g --g /out/v6/g/${GN}_for_ep$EP.npz --tag _$GN" > $L/gate_$EP.log 2>&1
}
export -f gate
i=0; for e in $TEACH; do echo "$i $e"; i=$((i + 1)); done | xargs -P 2 -L 1 bash -c 'gate $(( $0 % 2 )) $1'
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
log "Gate 1: $PASS / 17 at >= 90% ($(tr '\n' ' ' < $L/gate1.txt)) - reported, M runs continue"

# ---- 4. M runs ------------------------------------------------------------------------------------------------------------
log "4. M runs: $M_JOBS"
worker() {  # <gpu>: take the next unclaimed run until none is left
  local GPU=$1 R EP
  for R in $M_JOBS; do
    mkdir $L/claim_$R 2> /dev/null || continue
    [[ -f $G/m/$R/DONE ]] && continue
    EP=${R##*_ep}
    log "M $R starting on GPU $GPU $(mix $EP)"
    env $(mix $EP) SUB=v6 BAND_AFTER=52428800 NOBEST_N=6 ./run_m.sh $EP $GPU $R $GN.npz > $L/run_m_$R.log 2>&1
    log "M $R finished: $(cat $G/m/$R/STOP 2>/dev/null || echo cap)"
  done
}
export -f worker; export M_JOBS
worker 0 & sleep 90; worker 1 & wait
while docker ps --format '{{.Names}}' | grep -q "gmd_me_\|gmd_m_"; do sleep 30; done

# ---- 5. summary + videos --------------------------------------------------------------------------------------------------
log "5. summary + videos"
export GMD_ROOT=$G GMD_GNAME=$GN GMD_TEACH="${TEACH// /, }" GMD_STOP="within 0.05 for 5 checkpoints and no new best in the last 6, after 52.4M"
export GMD_RUNS="$(for r in $M_JOBS; do g=${r%%_*}; e=${r##*_ep}; case $g in A) d=seen;; B) d=unseen;; C) d=S0_weak;; esac; printf '%s:%s:%s ' $g $d $e; done)"
export GMD_NOTE="Run 6: run 5's recipe with 17 teachers (38, 30, 6, 32, 15 moved in from run 5's unseen set); M on the 6 unseen episodes; per-world v1/v2 meshes (50/50) on ${MIX// /, } (scored on v2)."
$PY summarize_full.py s0only >> $L/pipeline.log 2>&1
$PY summarize_full.py >> $L/pipeline.log 2>&1
hostpq() {  # host parquet dir of an episode
  local EP=$1 RC; RC=$(python3 episodes.py recon $EP)
  case $RC in
    v2recon) echo $O/ego2sim_recon_v2_pkg/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave ;;
    v2clean) echo $O/ego2sim_recon_v2clean_pkg/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave ;;
    v2clean2) echo $O/ego2sim_recon_v2clean2_pkg/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave ;;
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
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP rs0_$EP -- "python /env/gmd/render_m.py evaluation.checkpoint=$(python3 episodes.py ckpt $EP) viewer.output_path=/out/v6/videos/ep${EP}_S0.mp4 $C $V" > $L/render_S0_$EP.log 2>&1
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP rmg_$EP -- "GMD_G=/out/v6/g/$GN.npz python /env/gmd/render_m.py evaluation.checkpoint=/out/v6/m/$R/run/policy_$BEST.safetensors viewer.output_path=/out/v6/videos/ep${EP}_MG.mp4 $C $V" > $L/render_MG_$EP.log 2>&1
  S0A=$($PY -c "import json;print('%.3f'%json.load(open('$G/summary_s0.json'))['$EP'])" 2>/dev/null || echo "?")
  MGA=$($PY -c "import json;r=[json.loads(l) for l in open('$G/m/$R/evals.jsonl')];b=max(r,key=lambda x:x['add_auc']);print('%.3f at %.1fM'%(b['add_auc'],b['step']/1e6))")
  ffmpeg -loglevel error -y -i $G/videos/ep${EP}_S0.mp4 -i $G/videos/ep${EP}_MG.mp4 -filter_complex \
    "[0]scale=640:360,drawtext=fontfile=$F:text='ep$EP S0 policy  AUC $S0A':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[a];[1]scale=640:360,drawtext=fontfile=$F:text='${R%%_*} M + frozen $GN  AUC $MGA':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[b];[a][b]hstack=2" \
    -c:v libx264 -crf 20 -pix_fmt yuv420p $G/videos/ep${EP}_M_vs_S0.mp4
}
i=0; for r in $M_JOBS; do echo "$i $r"; i=$((i + 1)); done > $L/render_jobs.txt
(awk '$1 % 2 == 0 {print $2}' $L/render_jobs.txt | while read -r r; do render 0 $r; done) &
(awk '$1 % 2 == 1 {print $2}' $L/render_jobs.txt | while read -r r; do render 1 $r; done); wait
log "finished"
