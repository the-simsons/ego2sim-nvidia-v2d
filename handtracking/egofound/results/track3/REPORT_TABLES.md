## Primary metrics (set v2d, cam_b/cam_c held out; each method on its own scored hand-frames)

| method | Tri MPJPE abc (mm) | depth (mm) | lateral (mm) | depth bias (mm) | Tri MPJPE ab (mm) | cam_b kp (px) | cam_b PCK@10 | cam_c kp (px) | cam_c PCK@10 | cam_b IoU | cam_c IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 263.4 | 261.3 | 26.4 | +259.6 | 258.8 | 25.32 | 0.138 | 31.11 | 0.110 | 0.363 | 0.365 |
| F0 | 119.2 (-144.2) | 115.3 (-146.0) | 21.0 (-5.4) | +107.7 | 113.4 (-145.4) | 27.76 (+2.44) | 0.166 (+0.028) | 31.26 (+0.15) | 0.137 (+0.027) | 0.394 (+0.031) | 0.399 (+0.034) |
| F2 | 62.3 (-201.1) | 57.9 (-203.4) | 14.0 (-12.4) | +20.3 | **37.8** (-221.0) | 25.50 (+0.19) | 0.270 (+0.132) | 30.22 (-0.89) | 0.229 (+0.119) | 0.416 (+0.052) | 0.416 (+0.051) |
| F3 | 62.5 (-200.9) | 58.1 (-203.2) | 14.2 (-12.2) | +21.1 | 42.6 (-216.2) | 24.87 (-0.44) | 0.269 (+0.131) | 29.23 (-1.88) | 0.239 (+0.130) | 0.419 (+0.055) | 0.419 (+0.054) |
| F0-known-K | 105.4 (-158.0) | 102.6 (-158.7) | 15.2 (-11.2) | +93.2 | 101.2 (-157.6) | 23.72 (-1.60) | 0.265 (+0.127) | 26.90 (-4.21) | 0.236 (+0.126) | **0.442** (+0.079) | **0.443** (+0.078) |
| F2-known-K | 62.1 (-201.3) | 57.8 (-203.5) | 13.9 (-12.6) | +20.1 | 37.8 (-221.0) | 24.85 (-0.46) | 0.277 (+0.138) | 29.49 (-1.63) | 0.234 (+0.124) | 0.423 (+0.059) | 0.419 (+0.054) |
| F3-known-K | 62.2 (-201.1) | 57.9 (-203.4) | 14.0 (-12.4) | +20.8 | 42.6 (-216.3) | 24.24 (-1.08) | 0.275 (+0.137) | 28.53 (-2.58) | 0.245 (+0.135) | 0.425 (+0.062) | 0.422 (+0.057) |
| E1best-sgbm | 49.5 (-213.8) | **42.5** (-218.8) | 17.3 (-9.1) | +6.7 | 58.6 (-200.2) | 21.00 (-4.31) | 0.238 (+0.100) | 27.45 (-3.67) | 0.232 (+0.122) | 0.412 (+0.049) | 0.410 (+0.045) |
| E8fallback-e1best-smooth | **47.5** (-215.9) | 44.5 (-216.8) | **10.7** (-15.8) | +7.5 | 66.6 (-192.2) | **17.89** (-7.42) | **0.335** (+0.197) | **23.92** (-7.20) | **0.348** (+0.238) | 0.441 (+0.078) | 0.437 (+0.072) |

## Secondary metrics

| method | cam_a IoU | cam_a kp (px) | centred MPJPE (mm) | stereo resid (mm)* | stereo bias (mm)* | jitter (m/s²) | % wrist >2 m/s | bone std (mm) | betas std | swaps/min | coverage | % infilled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 0.284 | 42.20 | 42.4 | 200.9 | +195.4 | 13.40 | 1.89% | 1.05 | **0.000** | 0.19 | **100.0%** | 1.2% |
| F0 | 0.298 (+0.014) | 49.28 (+7.08) | 39.7 (-2.7) | 81.3 (-119.6) | +78.5 | 4.79 (-8.61) | **0.00%** (-1.9pp) | 3.23 (+2.19) | 0.179 (+0.179) | **0.00** (-0.19) | 50.0% (-50.0pp) | 49.3% |
| F2 | 0.287 (+0.003) | 47.44 (+5.24) | 39.7 (-2.7) | 54.8 (-146.2) | +17.0 | 13.99 (+0.59) | 2.67% (+0.8pp) | 3.23 (+2.19) | 0.179 (+0.179) | **0.00** (-0.19) | 50.0% (-50.0pp) | 49.3% |
| F3 | 0.288 (+0.004) | 46.43 (+4.23) | 39.6 (-2.8) | 52.8 (-148.2) | +16.8 | 5.26 (-8.14) | 1.22% (-0.7pp) | 2.97 (+1.92) | 0.179 (+0.179) | **0.00** (-0.19) | 50.0% (-50.0pp) | 49.3% |
| F0-known-K | **0.333** (+0.049) | 39.12 (-3.08) | 39.7 (-2.7) | 71.2 (-129.7) | +66.8 | 4.78 (-8.62) | **0.00%** (-1.9pp) | 3.25 (+2.20) | 0.179 (+0.179) | **0.00** (-0.19) | 50.1% (-49.9pp) | 49.3% |
| F2-known-K | 0.292 (+0.008) | 46.17 (+3.98) | 39.7 (-2.7) | 53.8 (-147.1) | +15.5 | 13.78 (+0.38) | 2.62% (+0.7pp) | 3.25 (+2.20) | 0.179 (+0.179) | **0.00** (-0.19) | 50.1% (-49.9pp) | 49.3% |
| F3-known-K | 0.293 (+0.009) | 45.22 (+3.03) | 39.6 (-2.8) | 51.3 (-149.6) | +15.8 | 5.16 (-8.24) | 0.86% (-1.0pp) | 2.98 (+1.94) | 0.179 (+0.179) | **0.00** (-0.19) | 50.1% (-49.9pp) | 49.3% |
| E1best-sgbm | 0.286 (+0.002) | 41.59 (-0.61) | 36.5 (-5.9) | **10.0** (-190.9) | +0.3 | **2.73** (-10.67) | 1.21% (-0.7pp) | **0.99** (-0.06) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E8fallback-e1best-smooth | 0.294 (+0.010) | **32.57** (-9.63) | **36.0** (-6.4) | 28.3 (-172.6) | -9.0 | 3.09 (-10.30) | 1.10% (-0.8pp) | 2.88 (+1.84) | 0.489 (+0.489) | **0.00** (-0.19) | 97.9% (-2.1pp) | 0.0% |

\* stereo residual uses SGBM depth from cam_b+cam_c: not a held-out metric, and circular for methods that consumed stereo depth.

## Common hand-frames (2767 hand-frames scored by every method with a tri-abc reference)

| method | Tri MPJPE abc (mm) | depth (mm) | lateral (mm) | depth bias (mm) | Tri MPJPE ab (mm) | cam_b kp (px) | cam_b PCK@10 | cam_a IoU | cam_a kp (px) | centred MPJPE (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 260.9 | 259.0 | 25.3 | +257.1 | 256.5 | 23.86 | 0.135 | 0.279 | 41.86 | 42.8 |
| F0 | 119.2 (-141.7) | 115.3 (-143.7) | 21.0 (-4.3) | +108.2 | 113.0 (-143.5) | 21.79 (-2.07) | 0.187 (+0.052) | 0.305 (+0.025) | 44.60 (+2.74) | 39.7 (-3.1) |
| F2 | 62.2 (-198.7) | 57.8 (-201.2) | 13.9 (-11.4) | +20.5 | **37.0** (-219.6) | 17.18 (-6.68) | 0.335 (+0.200) | 0.304 (+0.025) | 36.82 (-5.04) | 39.7 (-3.1) |
| F3 | 62.4 (-198.5) | 58.0 (-201.0) | 14.1 (-11.2) | +21.2 | 41.8 (-214.7) | 17.10 (-6.76) | 0.329 (+0.194) | 0.306 (+0.026) | 36.65 (-5.21) | 39.6 (-3.2) |
| F0-known-K | 105.4 (-155.5) | 102.6 (-156.4) | 15.2 (-10.1) | +93.7 | 100.7 (-155.8) | 17.34 (-6.52) | 0.305 (+0.170) | **0.342** (+0.063) | 33.56 (-8.31) | 39.7 (-3.1) |
| F2-known-K | 62.1 (-198.8) | 57.7 (-201.2) | 13.8 (-11.5) | +20.3 | **37.0** (-219.6) | 17.08 (-6.78) | 0.337 (+0.202) | 0.305 (+0.026) | 36.59 (-5.28) | 39.7 (-3.1) |
| F3-known-K | 62.2 (-198.7) | 57.9 (-201.1) | 14.0 (-11.4) | +21.0 | 41.8 (-214.8) | 16.93 (-6.93) | 0.331 (+0.197) | 0.307 (+0.028) | 36.27 (-5.59) | 39.6 (-3.2) |
| E1best-sgbm | 49.2 (-211.7) | **42.6** (-216.3) | 16.6 (-8.7) | +6.2 | 56.6 (-199.9) | 19.08 (-4.78) | 0.257 (+0.122) | 0.280 (+0.000) | 41.17 (-0.70) | 36.9 (-5.9) |
| E8fallback-e1best-smooth | **47.6** (-213.3) | 44.6 (-214.4) | **10.6** (-14.8) | +7.2 | 65.5 (-191.1) | **14.52** (-9.34) | **0.397** (+0.262) | 0.295 (+0.016) | **28.93** (-12.93) | **36.4** (-6.4) |

## Runs

| method | version | checkpoint | consumed stereo depth | GPU | wall time (s) | fps | scored hand-frames |
|---|---|---|---|---|---|---|---|
| E0 | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) | HaMeR (v2d_hamer weights) + WiLoR detector | no | n/a (cached run; adapter on CPU) | – | – | 9950 |
| F0 | EgoFound3R@264f4df9fd42536aae4dba982d18272cc3334077 | /usr1/e2s/pbasq-ego2sim-nvidia-v2d-clone/handtracking-egofound/weights-vggt/EgoFound3R.pt | no | NVIDIA GeForce RTX 4090 | 106.51294374465942 | – | 4978 |
| F2 | EgoFound3R@264f4df9fd42536aae4dba982d18272cc3334077 | /usr1/e2s/pbasq-ego2sim-nvidia-v2d-clone/handtracking-egofound/weights-vggt/EgoFound3R.pt | no | NVIDIA GeForce RTX 4090 | 106.51294374465942 | – | 4978 |
| F3 | EgoFound3R@264f4df9fd42536aae4dba982d18272cc3334077 | /usr1/e2s/pbasq-ego2sim-nvidia-v2d-clone/handtracking-egofound/weights-vggt/EgoFound3R.pt | no | NVIDIA GeForce RTX 4090 | 106.51294374465942 | – | 4978 |
| F0-known-K | EgoFound3R@264f4df9fd42536aae4dba982d18272cc3334077 | /usr1/e2s/pbasq-ego2sim-nvidia-v2d-clone/handtracking-egofound/weights-vggt/EgoFound3R.pt | no | NVIDIA GeForce RTX 4090 | 116.56893634796143 | – | 4994 |
| F2-known-K | EgoFound3R@264f4df9fd42536aae4dba982d18272cc3334077 | /usr1/e2s/pbasq-ego2sim-nvidia-v2d-clone/handtracking-egofound/weights-vggt/EgoFound3R.pt | no | NVIDIA GeForce RTX 4090 | 116.56893634796143 | – | 4994 |
| F3-known-K | EgoFound3R@264f4df9fd42536aae4dba982d18272cc3334077 | /usr1/e2s/pbasq-ego2sim-nvidia-v2d-clone/handtracking-egofound/weights-vggt/EgoFound3R.pt | no | NVIDIA GeForce RTX 4090 | 116.56893634796143 | – | 4994 |
| E1best-sgbm | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 betas size + E1 align dz + E1 smooth savgol | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E8fallback-e1best-smooth | HaWoR@66c7d41 (cam) + E7 tri(a+c)-shift + carry + sgbm + savgol; per-frame fallback: video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 betas size + E1 align dz + E1 smooth savgol | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | yes | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 9732 |
