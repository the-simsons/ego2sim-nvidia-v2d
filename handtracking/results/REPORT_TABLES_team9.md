## Primary metrics (set v2d; episodes 2,11,13,18,21,23,26,40,41 only, cam_b/cam_c held out; each method on its own scored hand-frames)

| method | Tri MPJPE abc (mm) | depth (mm) | lateral (mm) | depth bias (mm) | Tri MPJPE ab (mm) | cam_b kp (px) | cam_b PCK@10 | cam_c kp (px) | cam_b IoU |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 257.5 | 255.4 | 25.8 | +253.3 | 254.1 | 25.04 | 0.140 | 35.40 | 0.362 |
| TEAM-objfill_v2s | 72.2 (-185.3) | 65.3 (-190.0) | 20.9 (-4.9) | +45.0 | 78.4 (-175.7) | 23.82 (-1.22) | 0.223 (+0.083) | 35.76 (+0.36) | 0.422 (+0.060) |
| E1best-sgbm | 53.0 (-204.5) | 46.5 (-208.9) | 17.4 (-8.4) | +9.6 | 62.2 (-191.9) | 21.11 (-3.93) | 0.249 (+0.109) | 31.56 (-3.85) | 0.418 (+0.056) |
| E2-hawor-cam-stereo | **50.5** (-207.0) | **45.8** (-209.6) | 13.7 (-12.1) | +15.0 | **56.7** (-197.4) | 19.41 (-5.63) | 0.239 (+0.099) | 30.19 (-5.22) | **0.454** (+0.092) |
| E7tri-e0 | 59.0 (-198.5) | 53.4 (-202.0) | 16.9 (-8.9) | +13.0 | 76.7 (-177.4) | 28.43 (+3.39) | 0.177 (+0.037) | 40.23 (+4.82) | 0.362 (-0.000) |
| E7tri-hawor | 52.2 (-205.3) | 49.2 (-206.2) | 11.0 (-14.8) | +11.2 | 72.6 (-181.6) | 20.54 (-4.50) | 0.306 (+0.167) | 30.52 (-4.88) | 0.432 (+0.070) |
| E8fallback-e1best-smooth | 52.3 (-205.2) | 49.4 (-206.0) | **10.9** (-14.9) | +12.4 | 72.2 (-181.9) | **18.53** (-6.51) | **0.308** (+0.168) | **28.14** (-7.27) | 0.444 (+0.082) |

## Secondary metrics

| method | cam_a IoU | cam_a kp (px) | centred MPJPE (mm) | stereo resid (mm)* | stereo bias (mm)* | jitter (m/s²) | % wrist >2 m/s | bone std (mm) | betas std | swaps/min | coverage | % infilled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 0.284 | 42.52 | 44.4 | 193.7 | +193.0 | 14.11 | 2.11% | 1.06 | **0.000** | **0.00** | **100.0%** | 0.0% |
| TEAM-objfill_v2s | 0.288 (+0.004) | 47.29 (+4.77) | 38.7 (-5.8) | 30.8 (-162.9) | +28.7 | 4.56 (-9.54) | 0.21% (-1.9pp) | 1.45 (+0.39) | **0.000** (+0.000) | **0.00** (+0.00) | 90.5% (-9.5pp) | 13.2% |
| E1best-sgbm | 0.288 (+0.003) | 41.92 (-0.60) | 39.0 (-5.4) | 9.3 (-184.4) | +0.2 | **2.17** (-11.93) | 1.11% (-1.0pp) | **0.74** (-0.32) | **0.000** (+0.000) | **0.00** (+0.00) | **100.0%** (+0.0pp) | 0.0% |
| E2-hawor-cam-stereo | **0.307** (+0.022) | 35.07 (-7.45) | **37.4** (-7.1) | **7.9** (-185.7) | -0.0 | 2.76 (-11.35) | 0.13% (-2.0pp) | 2.04 (+0.98) | 0.412 (+0.412) | **0.00** (+0.00) | 93.9% (-6.1pp) | 0.0% |
| E7tri-e0 | 0.231 (-0.054) | 59.33 (+16.80) | 44.4 (-0.0) | 41.6 (-152.0) | -8.2 | 4.33 (-9.77) | 1.27% (-0.8pp) | 2.20 (+1.14) | **0.000** (+0.000) | **0.00** (+0.00) | **100.0%** (+0.0pp) | 0.0% |
| E7tri-hawor | 0.287 (+0.003) | 39.53 (-2.99) | 38.9 (-5.5) | 32.2 (-161.5) | -10.4 | 2.90 (-11.21) | **0.02%** (-2.1pp) | 1.95 (+0.90) | 0.412 (+0.412) | **0.00** (+0.00) | 93.9% (-6.1pp) | 0.0% |
| E8fallback-e1best-smooth | 0.298 (+0.013) | **32.92** (-9.61) | 38.9 (-5.6) | 28.2 (-165.5) | -6.9 | 2.57 (-11.53) | 0.69% (-1.4pp) | 2.67 (+1.62) | 0.498 (+0.498) | **0.00** (+0.00) | 97.6% (-2.4pp) | 0.0% |

\* stereo residual uses SGBM depth from cam_b+cam_c: not a held-out metric, and circular for methods that consumed stereo depth.

## Common hand-frames (2350 hand-frames scored by every method with a tri-abc reference)

| method | Tri MPJPE abc (mm) | depth (mm) | lateral (mm) | depth bias (mm) | Tri MPJPE ab (mm) | cam_b kp (px) | cam_b PCK@10 | cam_a IoU | cam_a kp (px) | centred MPJPE (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 259.0 | 256.9 | 25.7 | +255.1 | 255.8 | 24.00 | 0.131 | 0.285 | 42.86 | 44.2 |
| TEAM-objfill_v2s | 70.7 (-188.3) | 63.9 (-193.0) | 20.4 (-5.3) | +43.4 | 75.1 (-180.6) | 20.77 (-3.23) | 0.237 (+0.105) | 0.283 (-0.001) | 46.34 (+3.49) | 38.6 (-5.6) |
| E1best-sgbm | 52.6 (-206.4) | 46.1 (-210.8) | 17.3 (-8.4) | +9.7 | 60.2 (-195.6) | 19.36 (-4.63) | 0.270 (+0.139) | 0.285 (+0.001) | 42.40 (-0.46) | 38.8 (-5.5) |
| E2-hawor-cam-stereo | **50.5** (-208.5) | **45.8** (-211.1) | 13.6 (-12.1) | +15.1 | **55.0** (-200.7) | 17.08 (-6.91) | 0.262 (+0.130) | **0.307** (+0.023) | 33.85 (-9.01) | **37.3** (-6.9) |
| E7tri-e0 | 58.2 (-200.8) | 52.6 (-204.3) | 16.9 (-8.8) | +13.2 | 73.9 (-181.9) | 21.22 (-2.78) | 0.210 (+0.079) | 0.250 (-0.035) | 45.01 (+2.16) | 44.2 (-0.0) |
| E7tri-hawor | 52.3 (-206.7) | 49.3 (-207.6) | 11.0 (-14.7) | +11.6 | 70.9 (-184.9) | 15.42 (-8.57) | 0.362 (+0.230) | 0.303 (+0.018) | 30.39 (-12.47) | 38.9 (-5.3) |
| E8fallback-e1best-smooth | 52.1 (-206.9) | 49.1 (-207.8) | **10.8** (-14.9) | +12.2 | 70.3 (-185.4) | **14.98** (-9.01) | **0.365** (+0.234) | 0.304 (+0.020) | **29.43** (-13.43) | 38.8 (-5.4) |

## Runs

| method | version | checkpoint | consumed stereo depth | GPU | wall time (s) | fps | scored hand-frames |
|---|---|---|---|---|---|---|---|
| E0 | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) | HaMeR (v2d_hamer weights) + WiLoR detector | no | n/a (cached run; adapter on CPU) | – | – | 4322 |
| TEAM-objfill_v2s | team pipeline objfill_v2s (Rian): WiLoR + SAM2 + depth alignment + clean_hand_tracks + fill_hand_gaps_object | WiLoR | yes | n/a (team run) | – | – | 3965 |
| E1best-sgbm | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 betas size + E1 align dz + E1 smooth savgol | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 4322 |
| E2-hawor-cam-stereo | HaWoR@66c7d41 (cam) + E1 align scale | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | yes | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 4055 |
| E7tri-e0 | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E7 tri(a+c)-shift + carry + sgbm + savgol | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 4322 |
| E7tri-hawor | HaWoR@66c7d41 (cam) + E7 tri(a+c)-shift + carry + sgbm + savgol | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | yes | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 4055 |
| E8fallback-e1best-smooth | HaWoR@66c7d41 (cam) + E7 tri(a+c)-shift + carry + sgbm + savgol; per-frame fallback: video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 betas size + E1 align dz + E1 smooth savgol | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | yes | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 4212 |
