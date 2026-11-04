#!/usr/bin/env bash
# Train M (frozen G in the loop) on one episode with the S0 recipe, score checkpoints as they appear, stop early by rule.
#   run_m.sh <episode> <gpu> <name> <g_file>      outputs: outputs/gmd/$SUB/m/<name>/ (SUB default: full)
# env: TOTAL (default 96468992 = 46 x 2097152), CKPT (2097152), WORLDS (default: the episode's S0 run), BAND_AFTER (33554432),
#      NOBEST_N (0 = off; converged also needs no new best in the last N checkpoints), GMD_VOC (ff = feedforward VOC),
#      GMD_SET (episode table, see episodes.py; also picks the reconstruction in run_fc.sh), EXTRA_OV (more hydra overrides
#      for training, e.g. training.exploration.target_sigma=0.2), GMD_GRIP_LIB / GMD_GRIP_W (grip reward, see genv.py),
#      GMD_MESHMIX (container path of a meshmix_prep.py spec: per-world v1/v2 meshes in training only; adds collision.ground=false;
#      the scoring of checkpoints stays on the episode's own v2 mesh)
set -uo pipefail
EP=$1; GPU=$2; NAME=$3; GF=$4; cd /usr1/e2s/retarget_env/gmd
SUB=${SUB:-full}; O=/usr1/e2s/v2d_challenge_data/outputs/gmd/$SUB/m/$NAME; mkdir -p $O; rm -f $O/STOP $O/DONE
H=/usr1/e2s/v2d_challenge_data/outputs/gmd/$SUB/logs/hold_gpu$GPU      # hold file: the next run on this GPU waits (frees it for renders)
while [[ -f $H ]]; do sleep 20; done
TOTAL=${TOTAL:-96468992}; CKPT=${CKPT:-2097152}; SEQ=$(printf 'episode_%06d' $EP)
OV=$(cat /usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0/$(python3 episodes.py s0dir $EP)/overrides.txt)
PQ=$(python3 episodes.py parquet $EP); WORLDS=${WORLDS:-$(python3 episodes.py worlds $EP)}
[[ -n ${GMD_MESHMIX:-} ]] && EXTRA_OV="${EXTRA_OV:-} collision.ground=false"
echo "episode $EP set ${GMD_SET:-v1} meshmix ${GMD_MESHMIX:-none} recon $(python3 episodes.py recon $EP) voc ${GMD_VOC:-old} grip_lib ${GMD_GRIP_LIB:-none} grip_w ${GMD_GRIP_W:-10} kernel ${GMD_GRIP_KERNEL:-narrow} extra ${EXTRA_OV:-none} g $GF total $TOTAL ckpt $CKPT worlds $WORLDS curriculum tracking_96m_pickup_ray5decay; S0 overrides: $OV" > $O/config.txt
./run_fc.sh $GPU $EP m_$NAME -- "GMD_MESHMIX=${GMD_MESHMIX:-} GMD_VOC=${GMD_VOC:-} GMD_GRIP_LIB=${GMD_GRIP_LIB:-} GMD_GRIP_W=${GMD_GRIP_W:-10} GMD_GRIP_KERNEL=${GMD_GRIP_KERNEL:-narrow} S0_EMA=0.001 S0_VOC_GATE=wrench GMD_G=/out/$SUB/g/$GF METRICS_JSONL=/out/$SUB/m/$NAME/metrics.jsonl \
  python /env/gmd/train_m.py experiment=sharpa_flash_sac \"task.parquet='$PQ'\" hydra.searchpath=[file:///env/cfg] \
  observation=policy_s0 observation.extras_path=/feat/$SEQ.npz training.replay.observation_storage_dtype=float16 \
  logging.mode=online logging.upload_checkpoints=false hydra.run.dir=/out/$SUB/m/$NAME/run scene.world_count=$WORLDS \
  task.motion_speed=1.0 $OV training.replay.capacity=2097152 training.replay.minimum_size=204800 \
  training.checkpoint.save_interval_environment_steps=$CKPT training.total_environment_steps=$TOTAL \
  curriculum=tracking_96m_pickup_ray5decay ${EXTRA_OV:-}" > $O/train.log 2>&1 &
TP=$!
sleep 60
MEM=10g ./run_fc.sh $GPU $EP me_$NAME -- "GMD_G=/out/$SUB/g/$GF python /env/gmd/eval_m.py /out/$SUB/m/$NAME --band_after ${BAND_AFTER:-33554432} --nobest_n ${NOBEST_N:-0}" > $O/eval.log 2>&1 &
EPID=$!
while kill -0 $TP 2>/dev/null; do
  if [[ -f $O/STOP ]]; then docker stop -t 30 gmd_m_$NAME > /dev/null 2>&1; echo "stopped by rule: $(cat $O/STOP)" >> $O/train.log; break; fi
  sleep 20
done
wait $TP; touch $O/DONE
wait $EPID
echo "finished $(date -u)" >> $O/train.log
