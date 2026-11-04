#!/usr/bin/env bash
# gmd_ft jobs on one GPU (outputs: outputs/gmd_ft/<name>/). Every job uses the episode's v4 run (outputs/gmd/v4/m/<run>) and
# its best checkpoint by 64-world ADD-AUC; G = v4's g9gated. S0 recipe overrides per episode as gmd/run_m.sh.
#   ft_job.sh parity <gpu> <run>                      v4 best M + G through FTEnv with R = 0 -> parity/<run>.json
#   ft_job.sh train_r <gpu> <run> <name> <K>          train R (M frozen, wrist perturbation), 64-world eval of every checkpoint,
#                                                     then robustness evals of the best R (perturbed, with and without R)
#   ft_job.sh resume_m <gpu> <run> <name> <r_name>    resume M from its v4 best training state with the best R of <r_name> frozen,
#                                                     64-world eval of every checkpoint, then the best M with R switched off
#   ft_job.sh harsh <gpu> <run>                       grip test at plan-(b) strength (M frozen, best R of r_<run>_k3 loaded for the
#                                                     environment): R off unshaken x3 (noise floor), R off and R on under harsh
#                                                     shaking (1.0-1.8x wrist speed, 1-3 cm at 1-3 Hz, sudden pushes) -> harsh/<run>_*.json
#   ft_job.sh joint <gpu> <run> <name>                plan (a): M and finger deviations trained jointly from scratch (jenv.py), same
#                                                     recipe and 81.8M cap as v4, fingers unfrozen from 50M; 64-world eval of every checkpoint
# env: FT_STEPS (default 25165824 = 12 x 2097152), FT_DELTA (0.05), FT_PEN (500); joint: FJ_UNFREEZE, FJ_RAMP, FJ_SCALE, FJ_PEN
set -uo pipefail
cd /usr1/e2s/gmd_ft
export GMD_SET=v2
JOB=$1; GPU=$2; RUN=$3
O=/usr1/e2s/v2d_challenge_data/outputs; V4=$O/gmd/v4/m/$RUN; FO=$O/gmd_ft
EP=${RUN##*_ep}; SEQ=$(printf 'episode_%06d' $EP); G=/gmd/v4/g/g9gated.npz
STEPS=${FT_STEPS:-25165824}; DELTA=${FT_DELTA:-0.05}; PEN=${FT_PEN:-500}
best() { python3 -c "import json,sys;r=[json.loads(l) for l in open(sys.argv[1])];x=max(r,key=lambda d:d['add_auc']);print(x['step'])" "$1"; }
BEST=$(best $V4/evals.jsonl); MCK=/gmd/v4/m/$RUN/run/policy_$BEST.safetensors
GM=$(cd /usr1/e2s/retarget_env/gmd && python3 episodes.py s0dir $EP)
OV=$(cat $O/rl_train_req/training_s0/$GM/overrides.txt)
PQ=$(cd /usr1/e2s/retarget_env/gmd && python3 episodes.py parquet $EP); WORLDS=$(cd /usr1/e2s/retarget_env/gmd && python3 episodes.py worlds $EP)
COMMON="experiment=sharpa_flash_sac \"task.parquet='$PQ'\" hydra.searchpath=[file:///env/cfg,file:///ft/cfg] observation=policy_s0 \
  observation.extras_path=/feat/$SEQ.npz training.replay.observation_storage_dtype=float16 logging.mode=online \
  logging.upload_checkpoints=false scene.world_count=$WORLDS task.motion_speed=1.0 $OV training.replay.capacity=2097152 \
  training.replay.minimum_size=204800 training.checkpoint.save_interval_environment_steps=2097152"
TENV="GMD_VOC=ff S0_EMA=0.001 S0_VOC_GATE=wrench GMD_G=$G FT_DELTA=$DELTA FT_PEN=$PEN"

train_and_watch() {  # <name> <train env> <eval env> <extra hydra overrides>
  local NAME=$1 ENV=$2 EENV=$3 EXTRA=$4 D=$FO/$1
  mkdir -p $D; rm -f $D/DONE
  echo "run $RUN best $BEST episode $EP worlds $WORLDS steps $STEPS | train env: $ENV | eval env: $EENV | extra: $EXTRA | S0 overrides: $OV" > $D/config.txt
  ./ft_fc.sh $GPU $EP $NAME -- "$ENV METRICS_JSONL=/out/$NAME/metrics.jsonl python /ft/train_ft.py $COMMON hydra.run.dir=/out/$NAME/run $EXTRA" > $D/train.log 2>&1 &
  local TP=$!
  sleep 60
  MEM=10g ./ft_fc.sh $GPU $EP e_$NAME -- "GMD_G=$G $EENV python /ft/ft_eval.py watch /out/$NAME" > $D/eval.log 2>&1 &
  local EPID=$!
  wait $TP; local RC=$?
  touch $D/DONE; wait $EPID
  echo "train exit $RC, finished $(date -u)" >> $D/train.log
  return $RC
}

case $JOB in
  parity)
    mkdir -p $FO/parity
    MEM=10g ./ft_fc.sh $GPU $EP par_$RUN -- "GMD_G=$G FT_MODE=m FT_DELTA=$DELTA python /ft/ft_eval.py once $MCK /out/parity/$RUN.json" > $FO/parity/$RUN.log 2>&1
    ;;
  train_r)
    NAME=$4; K=$5
    train_and_watch $NAME "$TENV FT_MODE=r FT_K=$K FT_M_CKPT=$MCK FT_PERTURB=1" "FT_MODE=r FT_K=$K FT_M_CKPT=$MCK FT_DELTA=$DELTA" \
      "training.total_environment_steps=$STEPS curriculum=ft_final" || exit 1
    RB=$(best $FO/$NAME/evals.jsonl); RCK=/out/$NAME/run/policy_$RB.safetensors; echo $RB > $FO/$NAME/best_checkpoint.txt
    for Z in 0 1; do      # robustness: full-strength perturbation, best R vs the same environment with R switched off
      MEM=10g ./ft_fc.sh $GPU $EP rob${Z}_$NAME -- "GMD_G=$G FT_MODE=r FT_K=$K FT_M_CKPT=$MCK FT_DELTA=$DELTA FT_PERTURB=1 FT_RAMP=1 FT_ZERO_R=$Z python /ft/ft_eval.py once $RCK /out/$NAME/perturbed_zero_r$Z.json" > $FO/$NAME/rob$Z.log 2>&1
    done
    ;;
  resume_m)
    NAME=$4; RN=$5; RB=$(cat $FO/$RN/best_checkpoint.txt); K=$(sed -n 's/.*FT_K=\([0-9]\).*/\1/p' $FO/$RN/config.txt | head -1)
    RCK=/out/$RN/run/policy_$RB.safetensors
    train_and_watch $NAME "$TENV FT_MODE=m FT_K=$K FT_R_CKPT=$RCK" "FT_MODE=m FT_K=$K FT_R_CKPT=$RCK FT_DELTA=$DELTA" \
      "training.total_environment_steps=$((BEST + STEPS)) curriculum=tracking_96m_pickup_ray5decay training.resume=true training.checkpoint_path=/gmd/v4/m/$RUN/run/state_$BEST.safetensors" || exit 1
    MB=$(best $FO/$NAME/evals.jsonl); echo $MB > $FO/$NAME/best_checkpoint.txt
    MEM=10g ./ft_fc.sh $GPU $EP roff_$NAME -- "GMD_G=$G FT_MODE=m FT_DELTA=$DELTA python /ft/ft_eval.py once /out/$NAME/run/policy_$MB.safetensors /out/$NAME/best_r_off.json" > $FO/$NAME/r_off.log 2>&1
    ;;
  harsh)
    RN=r_${RUN}_k3; RCK=/out/$RN/run/policy_$(cat $FO/$RN/best_checkpoint.txt).safetensors; mkdir -p $FO/harsh
    BASE="GMD_G=$G FT_MODE=r FT_K=3 FT_M_CKPT=$MCK FT_DELTA=$DELTA"
    HARSH="FT_PERTURB=1 FT_RAMP=1 FT_SCALE_MAX=1.8 FT_AMP_MIN=0.01 FT_AMP_MAX=0.03 FT_BURST_P=0.02 FT_BURST_A=0.6"
    for REP in 1 2 3; do
      MEM=10g ./ft_fc.sh $GPU $EP h_${RUN}_n$REP -- "$BASE FT_ZERO_R=1 FT_SEED=$REP python /ft/ft_eval.py once $RCK /out/harsh/${RUN}_normal_r_off_rep$REP.json" > $FO/harsh/${RUN}_normal_rep$REP.log 2>&1
    done
    for Z in 1 0; do
      MEM=10g ./ft_fc.sh $GPU $EP h_${RUN}_z$Z -- "$BASE $HARSH FT_ZERO_R=$Z python /ft/ft_eval.py once $RCK /out/harsh/${RUN}_harsh_zero_r$Z.json" > $FO/harsh/${RUN}_harsh_z$Z.log 2>&1
    done
    ;;
  joint)
    NAME=$4
    JV="FT_ENV=joint FJ_UNFREEZE=${FJ_UNFREEZE:-50000000} FJ_RAMP=${FJ_RAMP:-5000000} FJ_SCALE=${FJ_SCALE:-0.5} FJ_PEN=${FJ_PEN:-2000}"
    train_and_watch $NAME "$TENV $JV" "$JV" "training.total_environment_steps=81788928 curriculum=tracking_96m_pickup_ray5decay" || exit 1
    best $FO/$NAME/evals.jsonl > $FO/$NAME/best_checkpoint.txt
    ;;
esac
