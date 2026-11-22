#!/usr/bin/env bash
# Video-only object trajectory for one monocular clip (our phone recordings), the benchmark's best chain as of 2026-11-22
# (plans/RECONSTRUCTION_PLAN.md 5.1a / P4; bench preset video_default):
#   Depth Pro 2 depth (+ its focal estimate) -> FoundationPose -> Pi3X camera scaled to that depth
#   -> joint camera/object pose graph (resting object, per-10-frame scale) -> fp_repair
#   mono_clip_recon.sh CLIP [GPU]        CLIP = a manifest name in bench/clips/ (inputs only are read; never ground_truth)
# Env: E2S_DATA (data root), F_PX (known focal length in px; default: Depth Pro 2's own estimate), MONO_DEPTH=dp2|moge.
# Outputs, in the manifest's outputs_dir: dp2_depth_est/, dp2/intrinsics.json, fp_dp2/, camera_vo_pi3dp2.npz,
# camera_vo_pi3dp2_joint.npz, fp_dp2_rep/, provenance.json. Every stage is skipped when its output exists.
# The GPU must be completely free (lab rule); a CUDA start-up failure (seen intermittently on riddle) is retried.
set -uo pipefail
CLIP=${1:?clip}; GPU=${2:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
B=/usr1/e2s/rborah_v2d; A=${E2S_DATA:?set E2S_DATA}; RECON=~/v2d/vtd_pinned/reconstruction
HOST=~/v2d/envs/recon_host/.venv/bin/python; SAM=~/v2d/envs/v2d_sam2/.venv/bin/python
PI3=${PI3_PY:-$B/envs/pi3/bin/python}; DP2=${DP2_PY:-$B/envs/dp2/bin/python}
export HF_HOME=${HF_HOME_E2S:-$B/hf} HF_HUB_OFFLINE=1
eval "$($SAM - "$REPO" "$CLIP" <<'PY'
import sys, shlex
sys.path.insert(0, sys.argv[1])
from bench import manifest as mf
m = mf.inputs_only(mf.load(sys.argv[2]))
i = m["inputs"]
name = next(iter(i["object_masks"]))
mesh = i.get("object_meshes", {}).get(name, {}).get("mesh")
for k, v in {"OUT": m["outputs_dir"], "FRAMES": i["frames"], "VIDEO": i.get("video_undist", ""),
             "OBJ_MASKS": i["object_masks"][name], "HAND_MASKS": i["hand_masks"], "MESH": mesh or ""}.items():
    print(f"{k}={shlex.quote(v)}")
PY
)"
OUT=$A/$OUT; FRAMES=$A/$FRAMES; OBJ_MASKS=$A/$OBJ_MASKS; HAND_MASKS=$A/$HAND_MASKS; MESH=$A/$MESH
[ -n "$VIDEO" ] && VIDEO=$A/$VIDEO
mkdir -p "$OUT/dp2"
if [ -z "$(ls "$OUT/dp2_depth_est" 2>/dev/null)" ] || [ -z "$(ls "$OUT/fp_dp2" 2>/dev/null)" ] || [ ! -f "$OUT/camera_vo_pi3dp2.npz" ]; then
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$used" -gt 600 ] || [ "$apps" -gt 0 ]; then echo "GPU $GPU is not free (${used} MiB, ${apps} processes)"; exit 3; fi
fi
export CUDA_VISIBLE_DEVICES=$GPU
retry() { for t in 1 2 3; do "$@" && return 0; echo "  attempt $t failed; retrying in 60 s"; sleep 60; done; return 1; }
t0=$SECONDS; lap() { echo "[$CLIP $((SECONDS - t0)) s] $*"; }

# 1. depth + focal length
if [ -z "$(ls "$OUT/dp2_depth_est" 2>/dev/null)" ]; then
  retry $DP2 -W ignore "$REPO/tools/dp2_depth.py" --frames "$FRAMES" --out "$OUT/dp2_depth_est" ${F_PX:+--f-px $F_PX} > "$OUT/dp2.log" 2>&1 \
    || { echo "Depth Pro 2 failed"; tail -5 "$OUT/dp2.log"; exit 1; }
fi
F=$($SAM -c "import json; print(json.load(open('$OUT/dp2_depth_est/dp2.json'))['focal_px_used_fullres'])")
read W H < <($SAM -c "import cv2, glob; h, w = cv2.imread(sorted(glob.glob('$FRAMES/*.jpg'))[0]).shape[:2]; print(w, h)")
echo "{\"fx\": $F, \"fy\": $F, \"cx\": $((W / 2)).0, \"cy\": $((H / 2)).0, \"width\": $W, \"height\": $H}" > "$OUT/dp2/intrinsics.json"
lap "Depth Pro 2 depth, f = $F px"

# 2. object track
if [ -z "$VIDEO" ]; then
  VIDEO=$OUT/dp2/frames.mp4
  [ -f "$VIDEO" ] || ffmpeg -v error -y -framerate 30 -i "$FRAMES/%06d.jpg" -c:v libx264 -crf 12 -pix_fmt yuv420p "$VIDEO"
fi
if [ -z "$(ls "$OUT/fp_dp2" 2>/dev/null)" ]; then
  fp() { (cd "$RECON" && $HOST -c "
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
run_video_to_poses('$VIDEO', '$OUT/dp2_depth_est', '$OBJ_MASKS', '$OUT/dp2/intrinsics.json', '$MESH', '$OUT/fp_dp2',
                   'data/weights/foundation_pose', backend='nvlabs_pytorch', target_width=960, target_height=540,
                   mask_depth=True, reregister_iou_thresh=0.3, n_particles=16)" > "$OUT/fp_dp2.log" 2>&1); }
  retry fp || { echo "FoundationPose failed"; tail -8 "$OUT/fp_dp2.log"; exit 1; }
fi
lap "FoundationPose $(ls "$OUT/fp_dp2"/*.json | wc -l) frames"

# 3. camera (a short pause first: Pi3X right after FoundationPose has hit "CUDA driver initialization failed" on riddle)
[ -f "$OUT/camera_vo_pi3dp2.npz" ] || sleep 20
[ -f "$OUT/camera_vo_pi3dp2.npz" ] || retry $PI3 -W ignore "$REPO/tools/pi3_camera.py" --frames "$FRAMES" \
  --out "$OUT/camera_vo_pi3dp2.npz" --scale-depth "$OUT/dp2_depth_est" --exclude "$OBJ_MASKS,$HAND_MASKS" > "$OUT/pi3.log" 2>&1 \
  || { echo "Pi3X failed"; tail -5 "$OUT/pi3.log"; exit 1; }
lap "Pi3X camera"

# 4-5. joint refinement and jump repair (CPU)
export CUDA_VISIBLE_DEVICES=
K="{\"fx\":$F,\"fy\":$F,\"cx\":$((W / 2)),\"cy\":$((H / 2))}"
$SAM -W ignore "$REPO/tools/camera_pose_graph.py" --vo "$OUT/camera_vo_pi3dp2.npz" --poses "$OUT/fp_dp2" --mesh "$MESH" \
  --depth "$OUT/dp2_depth_est" --intrinsics "$K" --object-masks "$OBJ_MASKS" --hand-masks "$HAND_MASKS" \
  --out "$OUT/camera_vo_pi3dp2_joint.npz" --no-table --scale-every 10 | cut -c1-160
rm -rf "$OUT/fp_dp2_rep"
$SAM -W ignore "$REPO/tools/fp_repair.py" --poses "$OUT/fp_dp2" --camera "$OUT/camera_vo_pi3dp2_joint.npz" --mesh "$MESH" \
  --intrinsics "$K" --object-masks "$OBJ_MASKS" --hand-masks "$HAND_MASKS" --out "$OUT/fp_dp2_rep" | cut -c1-160
$SAM -c "
import sys; sys.path.insert(0, '$REPO')
from e2s import provenance
provenance.write('$OUT', provenance.make('video', depth_source='monocular_depthpro2', builder='scripts/mono_clip_recon.sh',
                 focal_px=$F, camera='camera_vo_pi3dp2_joint.npz', object='fp_dp2_rep'))"
lap "done: camera $OUT/camera_vo_pi3dp2_joint.npz, object $OUT/fp_dp2_rep"
