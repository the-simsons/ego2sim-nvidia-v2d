#!/usr/bin/env bash
# VOC fix check on ep26 (public teammate recon) and ep37 (v2recon): old VOC replay, feedforward VOC replay, S0 best policy
# (trained with the old VOC, run without VOC), same camera, real time; plus a 16-world nominal S0 rollout for per-frame errors.
cd /usr1/e2s/retarget_env/gmd
O=/usr1/e2s/v2d_challenge_data/outputs/gmd/full/debug/vocfix; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python; mkdir -p $O
V="viewer.backend=mp4 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra"
one() {  # <gpu> <ep> <recon> <host parquet dir> <container parquet dir> <checkpoint>
  local GPU=$1 EP=$2 RC=$3 HPQ=$4 PQ=$5 CK=$6 CP CT NF
  read -r CP CT NF < <($PY cam_parquet.py $HPQ 0.5 0.25)
  local C="'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]'"
  for M in old ff; do
    RECON=$RC MEM=10g timeout 900 ./run_fc.sh $GPU $EP vf_${M}_$EP -- "VOC_MODE=$M REPLAY_JSON=/out/full/debug/vocfix/ep${EP}_$M.json python /env/voc/replay_voc.py \"task.parquet='$PQ'\" task.motion_speed=1.0 viewer.output_path=/out/full/debug/vocfix/ep${EP}_$M.mp4 viewer.num_frames=$(( (NF - 1) * 5 / 2 )) viewer.video_fps=50 $C $V" > $O/ep${EP}_$M.log 2>&1
  done
  RECON=$RC MEM=10g timeout 900 ./run_fc.sh $GPU $EP vf_pol_$EP -- "python /env/gmd/render_m.py evaluation.checkpoint=$CK viewer.output_path=/out/full/debug/vocfix/ep${EP}_policy.mp4 viewer.num_frames=$((NF - 1)) viewer.video_fps=20 $C $V" > $O/ep${EP}_policy.log 2>&1
  RECON=$RC MEM=12g timeout 900 ./run_fc.sh $GPU $EP vf_ro_$EP -- "python /env/gmd/rollout.py $CK /out/full/debug/vocfix/ro_ep$EP nominal --worlds 16" > $O/ep${EP}_rollout.log 2>&1
}
P=/usr1/e2s/v2d_challenge_data/outputs
one 0 26 v1 $P/rl_train_req/assets/human_motion_data/ego_recon/processed/sequence_id=episode_000026_teammate/robot_name=sharpa_wave \
  /v2d/assets/human_motion_data/ego_recon/processed/sequence_id=episode_000026_teammate/robot_name=sharpa_wave /ckpt/ep26_v1_rayreward5decay/p1_96M/policy_79691776.safetensors &
one 1 37 v2recon $P/ego2sim_recon_v2_pkg/sim_reference/processed/sequence_id=ep37/robot_name=sharpa_wave \
  /pkg/sim_reference/processed/sequence_id=ep37/robot_name=sharpa_wave /ckpt/eval/ep37_v2recon_ray5decay/p1_96M/policy_71303168.safetensors &
wait
echo "done $(date -u)" >> $O/done.log
