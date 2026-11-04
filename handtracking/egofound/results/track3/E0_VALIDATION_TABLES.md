## Primary metrics (set v2d, cam_b/cam_c held out; each method on its own scored hand-frames)

| method | Tri MPJPE abc (mm) | depth (mm) | lateral (mm) | depth bias (mm) | Tri MPJPE ab (mm) | cam_b kp (px) | cam_b PCK@10 | cam_c kp (px) | cam_b IoU |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 241.5 | 239.3 | 27.3 | +237.5 | 230.4 | 26.59 | **0.115** | **42.74** | 0.333 |
| E0-standalone | **240.7** (-0.8) | **238.5** (-0.8) | **27.0** (-0.2) | +236.7 | **229.8** (-0.6) | **26.40** (-0.19) | 0.110 (-0.006) | 45.84 (+3.10) | **0.335** (+0.002) |
| E0-standalone-cached-settings | 241.3 (-0.2) | 239.1 (-0.1) | 27.1 (-0.2) | +237.4 | 230.3 (-0.1) | 26.46 (-0.13) | 0.112 (-0.004) | 42.87 (+0.13) | 0.334 (+0.001) |

## Secondary metrics

| method | cam_a IoU | cam_a kp (px) | centred MPJPE (mm) | stereo resid (mm)* | stereo bias (mm)* | jitter (m/s²) | % wrist >2 m/s | bone std (mm) | betas std | swaps/min | coverage | % infilled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 0.228 | 46.48 | 40.8 | **188.1** | +188.0 | **12.62** | **0.67%** | **1.03** | **0.000** | **0.00** | **100.0%** | 0.0% |
| E0-standalone | 0.222 (-0.005) | **46.31** (-0.18) | 44.0 (+3.2) | 189.8 (+1.7) | +189.7 | 32.59 (+19.97) | 5.37% (+4.7pp) | 1.17 (+0.14) | **0.000** (+0.000) | **0.00** (+0.00) | **100.0%** (+0.0pp) | 0.0% |
| E0-standalone-cached-settings | **0.230** (+0.003) | 46.44 (-0.05) | **40.4** (-0.4) | 188.7 (+0.5) | +188.6 | 13.58 (+0.95) | 1.34% (+0.7pp) | 1.04 (+0.02) | **0.000** (+0.000) | **0.00** (+0.00) | **100.0%** (+0.0pp) | 0.0% |

\* stereo residual uses SGBM depth from cam_b+cam_c: not a held-out metric, and circular for methods that consumed stereo depth.

## Common hand-frames (254 hand-frames scored by every method with a tri-abc reference)

| method | Tri MPJPE abc (mm) | depth (mm) | lateral (mm) | depth bias (mm) | Tri MPJPE ab (mm) | cam_b kp (px) | cam_b PCK@10 | cam_a IoU | cam_a kp (px) | centred MPJPE (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 241.5 | 239.3 | 27.3 | +237.5 | 231.6 | 26.83 | **0.102** | **0.241** | 47.80 | 40.8 |
| E0-standalone | **240.7** (-0.8) | **238.5** (-0.8) | **27.0** (-0.2) | +236.7 | **231.0** (-0.6) | **26.34** (-0.49) | 0.097 (-0.005) | 0.231 (-0.010) | **47.24** (-0.57) | 44.0 (+3.2) |
| E0-standalone-cached-settings | 241.3 (-0.2) | 239.1 (-0.1) | 27.1 (-0.2) | +237.4 | 231.5 (-0.1) | 26.66 (-0.17) | 0.099 (-0.003) | 0.240 (-0.001) | 47.31 (-0.50) | **40.4** (-0.4) |

## Runs

| method | version | checkpoint | consumed stereo depth | GPU | wall time (s) | fps | scored hand-frames |
|---|---|---|---|---|---|---|---|
| E0 | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) | HaMeR (v2d_hamer weights) + WiLoR detector | no | n/a (cached run; adapter on CPU) | – | – | 598 |
| E0-standalone | standalone E0: video_to_data@8c778f9 (read-only export, /usr1/e2s/pbasq-ego2sim-nvidia-v2d-clone/handtracking-egofound/methods/v2d_upstream_8c778f9) | {"moge": "/usr1/e2s/video_to_data/reconstruction/data/weights/moge/model.pt", "wilor": "/usr1/e2s/video_to_data/reconstruction/data/weights/wilor", "sam2": "/usr1/e2s/video_to_data/reconstruction/data/weights/sam2", "hamer": "/usr1/e2s/pbasq-ego2sim-nvidia-v2d-clone/handtracking-egofound/methods/weights-e0/hamer (symlinks)"} | no | NVIDIA GeForce RTX 4090 | 529.8 | – | 598 |
| E0-standalone-cached-settings | standalone E0: video_to_data@8c778f9 (read-only export, /usr1/e2s/pbasq-ego2sim-nvidia-v2d-clone/handtracking-egofound/methods/v2d_upstream_8c778f9) | {"moge": "/usr1/e2s/video_to_data/reconstruction/data/weights/moge/model.pt", "wilor": "/usr1/e2s/video_to_data/reconstruction/data/weights/wilor", "sam2": "/usr1/e2s/video_to_data/reconstruction/data/weights/sam2", "hamer": "/usr1/e2s/pbasq-ego2sim-nvidia-v2d-clone/handtracking-egofound/methods/weights-e0/hamer (symlinks)"} | no | NVIDIA GeForce RTX 4090 | 113.5 | – | 598 |
