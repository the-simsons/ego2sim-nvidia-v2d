#!/bin/bash
# Build the HaWoR environment (E2) without Docker: handtracking/methods/HaWoR + .venv-hawor, CUDA extensions compiled
# for sm_89 (RTX 4090) with the system CUDA 12.6 toolkit. Weights from the official sources in HaWoR's README.
# MANO: right = the readable WiLoR copy; left = the mirrored-right stand-in (no official MANO_LEFT on this machine).
set -euo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd)
M=$H/methods/HaWoR
PY=$H/methods/.venv-hawor/bin/python
export CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=4
cd $M
[ -x $PY ] || uv venv -q --python 3.10 $H/methods/.venv-hawor
uv pip install -q --python $PY torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -q --python $PY -r requirements.txt || true
uv pip install -q --python $PY "numpy<2" setuptools wheel ninja gdown pytorch-lightning==2.2.4 lightning-utilities torchmetrics==1.4.0 "matplotlib<3.9" smplx pyrender trimesh einops timm yacs omegaconf hydra-core joblib natsort av imageio-ffmpeg
(cd thirdparty/DROID-SLAM && $PY setup.py install > $H/logs/hawor_droid_build.log 2>&1)
mkdir -p weights/external weights/hawor/checkpoints thirdparty/Metric3D/weights _DATA/data/mano _DATA/data_left/mano_left
W=/usr1/e2s/video_to_data/reconstruction/data/weights/wilor/pretrained_models
[ -f weights/external/detector.pt ] || cp $W/detector.pt weights/external/ 2>/dev/null || wget -q -c https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P weights/external/
for f in checkpoints/hawor.ckpt checkpoints/infiller.pt model_config.yaml; do
  [ -f weights/hawor/$f ] || wget -q -c https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/$f -O weights/hawor/$f
done
[ -f weights/external/droid.pth ] || $H/methods/.venv-hawor/bin/gdown -q 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O weights/external/droid.pth
[ -f thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth ] || $H/methods/.venv-hawor/bin/gdown -q 1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr -O thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth
cp -n $W/MANO_RIGHT.pkl _DATA/data/mano/MANO_RIGHT.pkl
cp -n /usr1/e2s/retarget_env/mano_standin/models/MANO_LEFT.pkl _DATA/data_left/mano_left/MANO_LEFT.pkl
du -sh weights thirdparty/Metric3D/weights
$PY -c "import torch, droid_backends; print('torch', torch.__version__, 'droid_backends OK')"
echo HAWOR SETUP DONE
