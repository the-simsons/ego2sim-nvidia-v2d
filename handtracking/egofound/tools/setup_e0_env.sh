#!/usr/bin/env bash
# Standalone E0 env (no Docker): mirrors the upstream v2d_hamer / v2d_wilor / v2d_moge / v2d_sam2 / v2d_hand_alignment
# Dockerfiles at video_to_data@8c778f9. Run from handtracking/egofound/. Code and pip packages only; model weights are read from
# /usr1/e2s/video_to_data/reconstruction/data/weights (already on disk).
set -euo pipefail
cd "$(dirname "$0")/../methods"
UP=v2d_upstream_8c778f9/reconstruction/modules
# read-only export of the upstream modules at 8c778f9 (never edits /usr1/e2s/video_to_data)
mkdir -p v2d_upstream_8c778f9
for m in v2d_common v2d_docker v2d_hamer v2d_hand_alignment v2d_moge v2d_mv v2d_pipelines v2d_sam2 v2d_wilor; do
  [ -d $UP/$m ] || git -c safe.directory=/usr1/e2s/video_to_data -C /usr1/e2s/video_to_data archive 8c778f9 \
    reconstruction/modules/$m | tar -x -C v2d_upstream_8c778f9
done
# HaMeR crop/alignment options that the cached Track 3 E0 run used (uncommitted in the shared video_to_data tree, taken
# read-only as a diff against 8c778f9): WiLoR-box crops, invalid-depth filtering, hand-scale mode/clamp. Rebuilt from the
# diff into its own folder; run_e0_standalone.py runs these two files as scripts when one of those options is set.
P=v2d_hamer_lib_patched
if [ ! -f $P/align_hands.py ]; then
  mkdir -p $P && cp $UP/v2d_hamer/lib/{align_hands,masks_to_hands}.py $P/
  patch -p5 -d $P < ../tools/upstream_patches/v2d_hamer_wilor_crop_valid_depth_scale_clamp.diff
fi
PY=.venv-e0/bin/python
[ -x $PY ] || uv venv --python 3.11 .venv-e0
# torch 2.5.1 as in the Dockerfiles; WiLoR-mini (torch<=2.5) then resolves it to 2.5.0, as pip would in the image
uv pip install --python $PY torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python $PY "numpy<2" Pillow opencv-python tqdm pyrender trimesh pyopengl smplx yacs hydra-core omegaconf \
  pytorch-lightning timm einops webdataset braceexpand dill av joblib loguru scikit-image scipy \
  huggingface_hub transformers safetensors ultralytics setuptools wheel pip  # pip: chumpy setup.py imports it
uv pip install --python $PY --no-build-isolation git+https://github.com/mattloper/chumpy.git git+https://github.com/lixiny/manotorch.git
[ -d hamer ] || { git clone --recursive https://github.com/geopavlakos/hamer.git hamer && git -C hamer checkout 3a01849f4148352e9260b69bf28b65d1671a4905 && git -C hamer submodule update --init --recursive; }
uv pip install --python $PY --no-deps -e hamer
uv pip install --python $PY git+https://github.com/warmshao/WiLoR-mini.git
uv pip install --python $PY "moge @ git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030"
uv pip install --python $PY "numpy<2" h5py flask decord -e $UP/v2d_common -e $UP/v2d_hamer/lib -e $UP/v2d_wilor/lib \
  -e $UP/v2d_moge/lib -e $UP/v2d_hand_alignment/lib -e $UP/v2d_mv
# SAM2 (v2d_sam2 Dockerfile installs it unpinned from git): same commit as handbench's SAM2 env. --no-deps, else uv
# upgrades torch to satisfy SAM-2's torch>=2.5.1 (upstream runs SAM2 in its own image with 2.5.1); runs fine on 2.5.0.
uv pip install --python $PY --no-deps "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4" \
  -e $UP/v2d_sam2/lib
PYOPENGL_PLATFORM=egl PYTHONPATH=hamer $PY -c "
import torch, numpy; print('torch', torch.__version__, torch.version.cuda, 'numpy', numpy.__version__)
from hamer.models import load_hamer; from hamer.datasets.utils import generate_image_patch_cv2; print('hamer imports OK')
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; print('wilor_mini OK')
import moge; print('moge OK')
import pyrender; print('pyrender OK')
import sam2, v2d.sam2.lib.video_to_masks; print('sam2 OK')
"
echo E0_ENV_DONE
