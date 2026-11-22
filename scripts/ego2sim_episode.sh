#!/usr/bin/env bash
# Ego2Sim pipeline for one episode, one object at a time, from existing stereo + VO + FP tracks to a FlashCHORD-ready
# floating-Sharpa reference:
#   bundle (tools/build_ego_bundle.py) -> loader (MANO FK) -> rigid URDFs -> retarget -> support surfaces
#   ego2sim_episode.sh CLIP STEREO_DIR CAM_NPZ EXCLUDE_MASKS OBJECT=FP_DIR=MESH [OBJECT=FP_DIR=MESH ...]
# Env: TABLE_Z (default 0.75; scene lifted so the table top is a support disk, as in the bundled example),
#      LEFT_HAND / RIGHT_HAND (hand-track dirs; empty = placeholder zero hands), GPU (default 0),
#      DEVICE (default cuda:0; "cpu" when the GPUs are busy training) for the loader + retarget,
#      RETARGET_ARGS (extra ego_recon_to_sharpa.py flags, e.g. "--surface_project"),
#      TRAJ_SOURCE (default video), MOCAP_USED_FOR (comma list), DEPTH_SOURCE (default stereo): the provenance record
#      (e2s/provenance.py) written into each bundle and copied into each retargeted partition, where
#      scripts/train_episode.sh reads it (FINAL=1 refuses anything but video),
#      REF_FPS (default 20.0): written into the retargeted parquet's fps (the ego_recon registry says 30). With FlashSAC
#      task.motion_speed=1.0 and sim fps 20 Hz this gives ONE evaluator step per video frame, which the packer needs
#      (the evaluator numbers rows reference_frame = step + start_frame). True video rate is 20.25 fps (1.2% slower).
# Output: ~/v2d/ego2sim/{bundles,loaded,processed,reconstructed_stage}; URDFs/meshes go to the pinned repo's
# human_motion_data/ego_recon/processed (where generate_rigid_urdfs.py writes them).
set -euo pipefail
CLIP=${1:?clip}; S=${2:?stereo dir}; CAM=${3:?cam npz}; EXCL=${4:?exclude masks}; shift 4
source ~/v2d/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
W=~/v2d/ego2sim; RG=~/v2d/vtd_pinned/robotic_grounding
LOADER=~/v2d/envs/v2d_task_library_loader/.venv/bin/python; RGPY=~/v2d/envs/rg_retarget/.venv/bin/python
objargs=(); for o in "$@"; do objargs+=(--object "$o"); done
hands=(); [ -n "${LEFT_HAND:-}" ] && hands+=(--left-hand "$LEFT_HAND"); [ -n "${RIGHT_HAND:-}" ] && hands+=(--right-hand "$RIGHT_HAND")
echo "== bundle"
~/v2d/envs/recon_host/.venv/bin/python ~/v2d/tools/build_ego_bundle.py --stereo-dir "$S" --cam-npz "$CAM" \
  --out-root "$W/bundles" --clip "$CLIP" "${objargs[@]}" --exclude-masks "$EXCL" "${hands[@]}"   --trajectory-source "${TRAJ_SOURCE:-video}" --mocap-used-for "${MOCAP_USED_FOR:-}" --depth-source "${DEPTH_SOURCE:-stereo}" 2>&1 | grep -v -i warn
for o in "$@"; do
  name=${o%%=*}; seq=${CLIP}_${name}
  echo "== $seq: loader"
  IN=$W/loader_in/$seq; rm -rf "$IN"; mkdir -p "$IN"; ln -sfn "$W/bundles/$seq" "$IN/$seq"
  rm -rf "$W/loaded/sequence_id=$seq" "$W/processed/sequence_id=$seq"
  (cd /tmp && $LOADER -m v2d.task_library_loader.lib.run_loader --dataset ego_recon --dataset_root "$IN" \
    --output_dir "$W/loaded" --object_name "$name" --no_ground_align --ground_z_offset "${TABLE_Z:-0.75}" \
    --mano_model_dir ~/v2d/mano --device "${DEVICE:-cuda:0}" --save 2>&1 | grep -v -iE "warn|deprecat|it/s" | tail -3)
done
cd "$RG"
echo "== rigid URDFs"
$RGPY scripts/generate_rigid_urdfs.py --dataset ego_recon 2>&1 | grep -E "Generated|Error" || true
echo "== retarget"
$RGPY scripts/retarget/ego_recon_to_sharpa.py --input_dir "$W/loaded" --output_dir "$W/processed" --save --device "${DEVICE:-cuda:0}" \
  --sequence_pattern "^${CLIP}_" ${RETARGET_ARGS:-} 2>&1 | grep -v -iE "warn|it/s\]" | tail -3
echo "== set reference fps to ${REF_FPS:-20.0}"
$RGPY - "$W/processed" "${CLIP}" "${REF_FPS:-20.0}" <<'PY'
import glob, sys
import pyarrow as pa, pyarrow.parquet as pq
root, clip, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])
for p in glob.glob(f"{root}/sequence_id={clip}_*/robot_name=*/*.parquet"):
    t = pq.read_table(p)
    i = t.column_names.index("fps")
    t = t.set_column(i, t.schema.field(i), pa.array([fps] * t.num_rows, type=t.schema.field(i).type))
    pq.write_table(t, p)
    print(f"  {p.split('/')[-3]}: fps -> {fps}")
PY
mkdir -p "$W/reconstructed_stage"
for o in "$@"; do
  name=${o%%=*}; seq=${CLIP}_${name}
  for part in "$W/processed/sequence_id=$seq"/robot_name=*; do cp "$W/bundles/$seq/provenance.json" "$part/"; done
  echo "== $seq: support"
  $RGPY scripts/reconstruct_support_surfaces.py --input_dir "$W/processed" --dataset ego_recon --robot_name sharpa_wave \
    --sequence_id "$seq" --output "$W/reconstructed_stage/${seq}_sharpa_wave_support.usda" 2>&1 | grep -v -i warn | tail -4
  echo "== $seq: table support (stereo table plane)"
  $RGPY ~/v2d/tools/make_table_support.py "$W/bundles/$seq/outputs/result"     "$(ls "$W/processed/sequence_id=$seq/robot_name=sharpa_wave/"*.parquet | head -1)"     "$W/bundles/$seq/outputs/result/mesh.obj" "$W/reconstructed_stage/${seq}_sharpa_wave_support.usda"
done
