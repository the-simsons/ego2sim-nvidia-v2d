#!/usr/bin/env bash
# Render run 3's best checkpoints with debounce K (GPU 1, 2 at a time) and compose [k=0 | k=K] per episode.
cd /usr1/e2s/retarget_env/gmd
export GMD_SET=v2
O=/usr1/e2s/v2d_challenge_data/outputs; G=$O/gmd/v3; D=$G/rescore; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
mkdir -p $G/videos/gated
eval "$(sed -n '/^hostpq()/,/^}/p' v2_pipeline.sh)"
V="viewer.backend=mp4 viewer.video_fps=20 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra"
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
one() {  # <run> <K> <gpu>
  local R=$1 K=$2 GPU=$3 EP=${1##*_ep} B CK CP CT NF K0V A0 AK
  B=$($PY -c "import json;r=[json.loads(l) for l in open('$G/m/$R/evals.jsonl')];print(max(r,key=lambda x:x['add_auc'])['step'])")
  CK=/out/v3/m/$R/run/policy_$B.safetensors; K0V=$G/videos/ep${EP}_MG3.mp4
  if [[ $R == A_ep26 ]]; then CK=/out/v2/m/X2B_ep26/run/policy_$B.safetensors; K0V=$O/gmd/v2/exp/videos/ep26_X2B.mp4; fi
  read -r CP CT NF < <($PY cam_parquet.py $(hostpq $EP) 0.5 0.25)
  MEM=12g timeout 900 ./run_fc.sh $GPU $EP rgt_$EP -- "GMD_G=/out/v3/g/g9gated.npz GMD_DEBOUNCE=$K python /env/gmd/render_m.py evaluation.checkpoint=$CK viewer.output_path=/out/v3/videos/gated/ep${EP}_k$K.mp4 'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' viewer.num_frames=$((NF - 1)) $V" > $G/videos/gated/r_$EP.log 2>&1
  A0=$($PY -c "import json;print('%.3f'%json.load(open('$D/${R}_g9v2negB_k0.json'))['add_auc'])"); AK=$($PY -c "import json;print('%.3f'%json.load(open('$D/${R}_g9gated_k$K.json'))['add_auc'])")
  ffmpeg -loglevel error -y -i $K0V -i $G/videos/gated/ep${EP}_k$K.mp4 -filter_complex \
    "[0]fps=20,scale=640:360,drawtext=fontfile=$F:text='ep$EP run 3 M + new G (g9v2negB)  AUC $A0':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[a];[1]fps=20,scale=640:360,drawtext=fontfile=$F:text='same M + gated G  AUC $AK':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[b];[a][b]hstack=2:shortest=1" \
    -c:v libx264 -crf 20 -pix_fmt yuv420p $G/videos/gated/ep${EP}_negB_vs_gated.mp4
}
(one A_ep41 0 0; one A_ep3 0 0; one B_ep6 0 0; one B_ep30 0 0; one C_ep28 0 0) & (one A_ep26 0 1; one B_ep37 0 1; one B_ep15 0 1; one C_ep19 0 1) & wait
echo done >> $G/videos/gated/done.log
