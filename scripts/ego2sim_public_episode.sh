#!/usr/bin/env bash
# Locked Ego2Sim recipe (NOTES "Phase 5h") for one episode, up to a FlashCHORD-ready multi-object partition.
#   ego2sim_public_episode.sh EP OBJ1[,OBJ2...] [GPU]
# Needs ~/v2d/ego2sim/epEP/{obj_prompts.json,hand_prompts.json} (SAM2 prompts; object_id i = i-th name in OBJS; hand
# ids free, sides are voted from WiLoR). Every stage is skipped when its output exists (delete to redo).
# Env: SPLIT=public|evaluation (default public), MESH_SRC=scan (public scans; validation) or a dir with <obj>.glb/.obj,
#      STOP_BEFORE_FP=1: run only the mesh-independent stages (stereo, SAM2, MoGe, VO, hands).
#      DEPTH=stereo (SGBM, default) | stereo_fs (FoundationStereo, computed into epEP/stereo_fs/ if missing): the depth
#      FoundationPose tracks with (VO, hands and the bundle keep SGBM stereo/).
#      FP_KW: extra run_video_to_poses kwargs, e.g. "n_particles=16" (NOTES Phase 7d).
#      F0_K: static-start frame-0 smoothing over the first F0_K frames (default 5; 0 = off; NOTES Phase 8d).
#      TAG: suffix for this variant's tracks (epEP/fp_<obj>TAG), hand gap fill (hands/objfillTAG) and output
#      (train/epEPTAG), so variants do not overwrite each other. The shared bundles/loaded/processed/ per-object
#      intermediates and the pinned repo's ego_recon assets (by object name) are still rewritten by each run.
# Output: ~/v2d/ego2sim/train/epEP$TAG/{processed/sequence_id=epEP/robot_name=sharpa_wave, reconstructed_stage}
set -euo pipefail
EP=${1:?episode}; OBJS=${2:?objects}; GPU=${3:-0}
SPLIT=${SPLIT:-public}; MESH_SRC=${MESH_SRC:-scan}; DEPTH=${DEPTH:-stereo}; FP_KW=${FP_KW:-}; TAG=${TAG:-}
source ~/v2d/env.sh
export CUDA_VISIBLE_DEVICES=$GPU
E=~/v2d/ego2sim/ep$EP; S=$E/stereo; W=~/v2d/ego2sim; T=$W/train/ep$EP$TAG
DATA=~/v2d/v2d_data/track_3/$SPLIT; RECON=~/v2d/vtd_pinned/reconstruction
HOST=~/v2d/envs/recon_host/.venv/bin/python; CV=~/v2d/envs/v2d_sam2/.venv/bin/python
FPY=~/v2d/envs/v2d_foundation_pose/.venv/bin/python; RGPY=~/v2d/envs/rg_retarget/.venv/bin/python
IFS=',' read -ra NAMES <<< "$OBJS"
mesh_of() {  # never fails (set -e): scans are <obj>.glb or <obj>_visual.glb; MESH_SRC dirs hold <obj>.glb/.obj
  local c
  if [ "$MESH_SRC" = scan ]; then for c in "$DATA/mesh/$1/$1.glb" "$DATA/mesh/$1/$1_visual.glb"; do [ -f "$c" ] && { echo "$c"; return 0; }; done
  else for c in "$MESH_SRC/$1.glb" "$MESH_SRC/$1.obj"; do [ -f "$c" ] && { echo "$c"; return 0; }; done; fi
  echo "missing mesh for $1" >&2; return 0; }
t0=$SECONDS; lap() { echo "  [$((SECONDS - t0))s] $*"; }

CALIB=~/v2d/v2d_data/track_3/public/meta/camera_calibration.json  # the evaluation split ships no meta (same rig)
[ -f "$S/cam_a_undist.mp4" ] || { mkdir -p "$S"; $FPY ~/v2d/tools/ego_stereo_depth.py "$CALIB" \
  ~/v2d/tests_recon/stereo/extr_ep12.json "$DATA/videos/chunk-000" "$EP" "$S" 2>&1 | grep -v -i warn | tail -1; }
lap "stereo"
cd "$RECON"
for k in obj hand; do
  out=$E/masks; [ $k = hand ] && out=$E/hand_masks
  [ -d "$out" ] && [ -n "$(ls "$out")" ] && continue
  $HOST -c "
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
run_video_to_masks('$S/cam_a_undist.mp4', '$E/${k}_prompts.json', '$out', 'data/weights/sam2')" > "$E/sam2_$k.log" 2>&1
done
lap "SAM2 masks: objects $(ls $E/masks | tr '\n' ' '), hands $(ls $E/hand_masks | tr '\n' ' ')"
[ -d "$E/moge_depth" ] || $HOST -c "
from v2d.moge.docker.run_video_to_depth import run_video_to_depth
run_video_to_depth('$S/cam_a_undist.mp4', '$E/moge_depth', '$E/moge_intr', 'data/weights/moge', batch_size=8,
                   input_intrinsics_path='$S/intrinsics_undist.json')" > "$E/moge.log" 2>&1
[ -d "$E/fused_depth" ] || $CV ~/v2d/tools/fuse_depth.py "$S/stereo_depth" "$E/moge_depth" "$E/fused_depth"
lap "MoGe + fused depth"
excl=$(ls -d $E/masks/* $E/hand_masks/* | tr '\n' ',' | sed 's/,$//')
[ -f "$E/camera_vo.npz" ] || $CV ~/v2d/tools/ego_vo.py "$S" "$E/camera_vo.npz" --masks "$excl" 2>&1 | tail -1
lap "camera VO"
hand_ids=$(ls $E/hand_masks | tr '\n' ',' | sed 's/,$//')
objm=$(for i in "${!NAMES[@]}"; do echo -n "$E/masks/$i,"; done | sed 's/,$//')
[ -d "$E/hands/hands_by_side" ] || $HOST ~/v2d/tools/ego_hands.py "$S" "$E/hands" --hands "$hand_ids" \
  --hand-masks "$E/hand_masks" --object-masks "$objm" --depth-dir "$E/fused_depth" --gpu "$GPU" 2>&1 \
  | grep -E "votes|WARNING|real \+|^track" || true
for side in left right; do
  [ -d "$E/hands/clean/$side" ] || $CV ~/v2d/tools/clean_hand_tracks.py "$E/camera_vo.npz" \
    "$E/hands/hands_by_side/$side" "$E/hands/clean/$side"
done
lap "hands (MANO tracks)"
[ "${STOP_BEFORE_FP:-0}" = 1 ] && { lap "stopped before FoundationPose (STOP_BEFORE_FP=1)"; exit 0; }
if [ "$DEPTH" = stereo_fs ] && [ ! -f "$E/stereo_fs/stereo_stats.json" ]; then
  ~/v2d/envs/foundation_stereo/.venv/bin/python ~/v2d/tools/ego_stereo_depth.py "$CALIB" ~/v2d/tests_recon/stereo/extr_ep12.json \
    "$DATA/videos/chunk-000" "$EP" "$E/stereo_fs" --method fs 2>&1 | grep -v -i -E "warn|autocast" | tail -1
  rm -f "$E/stereo_fs/cam_a_undist.mp4"  # identical to stereo/cam_a_undist.mp4
  lap "FoundationStereo depth"
fi
for i in "${!NAMES[@]}"; do
  n=${NAMES[$i]}; [ -d "$E/fp_$n$TAG" ] && continue
  $HOST -c "
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
run_video_to_poses('$S/cam_a_undist.mp4', '$E/$DEPTH/stereo_depth', '$E/masks/$i', '$S/intrinsics_undist.json', '$(mesh_of "$n")',
                   '$E/fp_$n$TAG', 'data/weights/foundation_pose', backend='nvlabs_pytorch', target_width=1014, target_height=760,
                   mask_depth=True, reregister_iou_thresh=0.3${FP_KW:+, $FP_KW})" > "$E/fp_$n$TAG.log" 2>&1
done
lap "FoundationPose ($DEPTH ${FP_KW:-defaults}): $(for n in "${NAMES[@]}"; do echo -n "$n $(ls $E/fp_$n$TAG/*.json | wc -l) "; done)"
if [ "${F0_K:-5}" -gt 1 ]; then  # static-start frame 0 (the scorer aligns on frame 0; NOTES Phase 8d); F0_K=0 disables
  $CV ~/v2d/tools/fp_static_start.py "$E/camera_vo.npz" $(for n in "${NAMES[@]}"; do echo -n "$E/fp_$n$TAG "; done) \
    --k "${F0_K:-5}" 2>&1 | grep -v -i warn
  lap "frame-0 static start (k ${F0_K:-5})"
fi

fps_dirs=$(for n in "${NAMES[@]}"; do echo -n "$E/fp_$n$TAG=$(mesh_of "$n"),"; done | sed 's/,$//')
nfr=$(ls "$E/hands/frames" | wc -l)
for side in left right; do  # long gaps while gripping (WiLoR loses occluded hands): carry the hand with the object
  [ -d "$E/hands/objfill$TAG/$side" ] || ~/v2d/envs/v2d_hamer/.venv/bin/python ~/v2d/tools/fill_hand_gaps_object.py \
    "$E/camera_vo.npz" "$E/hands/clean/$side" "$E/hands/objfill$TAG/$side" --objects "$fps_dirs" --frames "$nfr" 2>&1 \
    | grep -v -i warn || true
done
lap "hands"
objargs=(); for i in "${!NAMES[@]}"; do n=${NAMES[$i]}; objargs+=("$n=$E/fp_$n$TAG=$(mesh_of "$n")"); done
hand_dir() { [ -n "$(ls "$E/hands/objfill$TAG/$1" 2>/dev/null)" ] && echo "$E/hands/objfill$TAG/$1" || true; }  # empty -> no hand
LEFT_HAND=$(hand_dir left) RIGHT_HAND=$(hand_dir right) RETARGET_ARGS="--smooth_reference --adaptive_offset" \
  DEVICE=${DEVICE:-cpu} GPU=$GPU bash ~/v2d/scripts/ego2sim_episode.sh "ep$EP" "$S" "$E/camera_vo.npz" "$excl" "${objargs[@]}" \
  2>&1 | grep -E "Loaded|table top|rror" || true
lap "bundle -> loader -> retarget -> support"
specs=(); for n in "${NAMES[@]}"; do
  specs+=("ep${EP}_$n::$(ls $W/processed/sequence_id=ep${EP}_$n/robot_name=sharpa_wave/*.parquet)::$W/bundles/ep${EP}_$n/outputs/result")
done
rm -rf "$T"; mkdir -p "$T/processed" "$T/reconstructed_stage"
(cd ~/v2d/vtd_pinned/robotic_grounding && $RGPY ~/v2d/tools/merge_sharpa_objects.py "$T/processed" "ep$EP" \
  "$W/reconstructed_stage" "${specs[@]}" 2>&1 | grep -E "merged|support:")
cp "$W/reconstructed_stage/ep${EP}_sharpa_wave_support.usda" "$T/reconstructed_stage/"
$RGPY ~/v2d/tools/check_reference_contacts.py "$T/processed/sequence_id=ep$EP/robot_name=sharpa_wave" --every 100000 \
  2>&1 | grep -v -i warn | grep -E "jumps|frames within"
lap "ready: $T/processed/sequence_id=ep$EP/robot_name=sharpa_wave"
