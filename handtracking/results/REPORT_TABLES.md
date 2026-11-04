## Primary metrics (set v2d, cam_b/cam_c held out; each method on its own scored hand-frames)

| method | Tri MPJPE abc (mm) | depth (mm) | lateral (mm) | depth bias (mm) | Tri MPJPE ab (mm) | cam_b kp (px) | cam_b PCK@10 | cam_c kp (px) | cam_b IoU |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 263.4 | 261.3 | 26.4 | +259.6 | 258.8 | 25.32 | 0.138 | 31.11 | 0.363 |
| E0-refined | 262.3 (-1.1) | 258.2 (-3.1) | 31.3 (+4.9) | +256.5 | 258.0 (-0.8) | 30.23 (+4.91) | 0.130 (-0.008) | 35.96 (+4.84) | 0.347 (-0.016) |
| E1a-betas | 51.0 (-212.3) | 43.2 (-218.1) | 18.7 (-7.7) | +7.1 | 59.6 (-199.2) | 27.63 (+2.31) | 0.222 (+0.083) | 27.73 (-3.38) | 0.388 (+0.025) |
| E1a-size-dz | 49.6 (-213.8) | 42.6 (-218.7) | 17.4 (-9.1) | +6.8 | 58.6 (-200.2) | 21.09 (-4.23) | 0.237 (+0.099) | 27.59 (-3.53) | 0.413 (+0.050) |
| E1a-size-dz-fs | 49.8 (-213.5) | 43.0 (-218.3) | 17.3 (-9.2) | +3.8 | 59.2 (-199.6) | 21.05 (-4.27) | 0.237 (+0.098) | 27.67 (-3.44) | 0.413 (+0.050) |
| E1b-dz | 58.2 (-205.2) | 48.5 (-212.8) | 23.6 (-2.8) | +21.9 | 65.4 (-193.4) | 26.13 (+0.81) | 0.125 (-0.013) | 32.95 (+1.83) | 0.355 (-0.008) |
| E1b-huber | 49.7 (-213.7) | 42.6 (-218.7) | 17.6 (-8.9) | +6.2 | 58.6 (-200.2) | 21.45 (-3.86) | 0.230 (+0.092) | 27.94 (-3.18) | 0.406 (+0.042) |
| E1b-scale | 49.7 (-213.7) | 42.6 (-218.7) | 17.5 (-8.9) | +6.7 | 58.6 (-200.2) | 21.21 (-4.10) | 0.234 (+0.096) | 27.73 (-3.38) | 0.410 (+0.047) |
| E1b-scale-fs | 50.0 (-213.4) | 43.1 (-218.2) | 17.4 (-9.0) | +3.8 | 59.3 (-199.5) | 21.15 (-4.16) | 0.234 (+0.095) | 27.82 (-3.30) | 0.411 (+0.048) |
| E1best-fs | 49.8 (-213.6) | 42.9 (-218.4) | 17.2 (-9.2) | +3.7 | 59.1 (-199.7) | 20.97 (-4.35) | 0.238 (+0.099) | 27.54 (-3.57) | 0.412 (+0.049) |
| E1best-sgbm | 49.5 (-213.8) | **42.5** (-218.8) | 17.3 (-9.1) | +6.7 | 58.6 (-200.2) | 21.00 (-4.31) | 0.238 (+0.100) | 27.45 (-3.67) | 0.412 (+0.049) |
| E1c-oneeuro | 51.2 (-212.2) | 43.0 (-218.3) | 20.2 (-6.3) | +5.7 | 60.4 (-198.4) | 30.21 (+4.90) | 0.153 (+0.015) | 35.93 (+4.81) | 0.356 (-0.007) |
| E1c-savgol | 49.6 (-213.7) | 42.6 (-218.7) | 17.4 (-9.0) | +6.7 | 58.6 (-200.2) | 21.11 (-4.21) | 0.235 (+0.097) | 27.58 (-3.54) | 0.410 (+0.047) |
| E1c-traj | 55.4 (-208.0) | 51.6 (-209.7) | 12.2 (-14.2) | -1.4 | 64.5 (-194.3) | **17.85** (-7.46) | 0.324 (+0.186) | 24.39 (-6.72) | 0.439 (+0.075) |
| E2-hawor | 121.1 (-142.2) | 115.3 (-146.0) | 21.6 (-4.9) | +109.3 | 114.4 (-144.4) | 28.48 (+3.17) | 0.211 (+0.073) | 29.90 (-1.21) | 0.415 (+0.052) |
| E2-hawor-cam | 121.2 (-142.2) | 115.3 (-146.0) | 21.6 (-4.9) | +109.3 | 114.5 (-144.4) | 28.55 (+3.23) | 0.211 (+0.073) | 29.97 (-1.14) | 0.415 (+0.052) |
| E2-hawor-cam-stereo | 53.2 (-210.2) | 44.7 (-216.6) | 19.1 (-7.4) | +15.8 | 57.1 (-201.7) | 26.51 (+1.19) | 0.248 (+0.110) | 29.58 (-1.54) | 0.438 (+0.075) |
| E2-hawor-cam-stereo-fs | 52.8 (-210.6) | 44.3 (-217.0) | 19.0 (-7.5) | +12.8 | **56.9** (-201.9) | 26.42 (+1.11) | 0.249 (+0.111) | 29.67 (-1.45) | 0.440 (+0.076) |
| E7tri-e0 | 53.9 (-209.5) | 48.0 (-213.3) | 16.9 (-9.5) | +8.5 | 72.4 (-186.4) | 27.93 (+2.62) | 0.180 (+0.042) | 35.45 (+4.33) | 0.361 (-0.002) |
| E7tri-hawor | 53.3 (-210.1) | 46.5 (-214.9) | 16.1 (-10.3) | +8.8 | 71.1 (-187.7) | 27.38 (+2.06) | 0.323 (+0.185) | 29.89 (-1.23) | 0.416 (+0.053) |
| E8fallback-e0 | 47.9 (-215.5) | 44.8 (-216.5) | 10.7 (-15.7) | +7.5 | 68.1 (-190.7) | 19.61 (-5.71) | 0.326 (+0.187) | 25.61 (-5.50) | 0.431 (+0.067) |
| E8fallback-e1best | 47.7 (-215.7) | 44.6 (-216.7) | 10.7 (-15.7) | +7.5 | 66.9 (-191.9) | 18.08 (-7.23) | 0.333 (+0.194) | 24.09 (-7.02) | 0.440 (+0.077) |
| E8fallback-e1best-replace | **47.5** (-215.9) | 44.5 (-216.8) | **10.5** (-15.9) | +7.3 | 66.8 (-192.1) | 17.96 (-7.36) | **0.337** (+0.198) | 24.06 (-7.06) | **0.443** (+0.079) |
| E8fallback-e1best-smooth | 47.5 (-215.9) | 44.5 (-216.8) | 10.7 (-15.8) | +7.5 | 66.6 (-192.2) | 17.89 (-7.42) | 0.335 (+0.197) | **23.92** (-7.20) | 0.441 (+0.078) |
| TEAM-objfill_v2s | 72.2 (-191.2) | 65.3 (-196.0) | 20.9 (-5.6) | +45.0 | 78.4 (-180.4) | 23.82 (-1.50) | 0.223 (+0.084) | 35.76 (+4.65) | 0.422 (+0.059) |

## Secondary metrics

| method | cam_a IoU | cam_a kp (px) | centred MPJPE (mm) | stereo resid (mm)* | stereo bias (mm)* | jitter (m/s²) | % wrist >2 m/s | bone std (mm) | betas std | swaps/min | coverage | % infilled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 0.284 | 42.20 | 42.4 | 200.9 | +195.4 | 13.40 | 1.89% | 1.05 | **0.000** | 0.19 | **100.0%** | 1.2% |
| E0-refined | 0.267 (-0.017) | 53.54 (+11.34) | 42.2 (-0.2) | 200.8 (-0.2) | +195.6 | 12.57 (-0.82) | 4.55% (+2.7pp) | 1.93 (+0.89) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E1a-betas | 0.282 (-0.002) | 49.39 (+7.19) | 37.0 (-5.4) | 8.7 (-192.2) | -0.0 | 7.80 (-5.60) | 1.56% (-0.3pp) | 3.69 (+2.64) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E1a-size-dz | 0.286 (+0.002) | 41.61 (-0.59) | 36.5 (-5.9) | 8.8 (-192.1) | -0.3 | 8.48 (-4.92) | 1.50% (-0.4pp) | 0.11 (-0.93) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E1a-size-dz-fs | 0.286 (+0.002) | 41.60 (-0.59) | 36.5 (-5.9) | 12.5 (-188.4) | +0.9 | 10.56 (-2.84) | 2.09% (+0.2pp) | **0.11** (-0.93) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E1b-dz | 0.226 (-0.058) | 54.61 (+12.41) | 42.5 (+0.1) | 12.8 (-188.1) | +5.1 | 8.41 (-4.98) | 1.64% (-0.2pp) | 0.93 (-0.11) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E1b-huber | 0.281 (-0.003) | 42.51 (+0.31) | 36.7 (-5.7) | 10.1 (-190.8) | +1.9 | 8.74 (-4.65) | 1.92% (+0.0pp) | 10.07 (+9.03) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E1b-scale | 0.283 (-0.001) | 41.97 (-0.23) | 36.6 (-5.8) | 8.8 (-192.2) | +0.0 | 7.80 (-5.60) | 1.60% (-0.3pp) | 3.69 (+2.64) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E1b-scale-fs | 0.283 (-0.001) | 41.97 (-0.23) | 36.6 (-5.8) | 12.8 (-188.1) | +1.2 | 10.50 (-2.89) | 2.18% (+0.3pp) | 4.61 (+3.56) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E1best-fs | 0.285 (+0.001) | 41.63 (-0.57) | 36.5 (-5.9) | 13.4 (-187.5) | +1.1 | 3.47 (-9.92) | 1.99% (+0.1pp) | 0.98 (-0.06) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E1best-sgbm | 0.286 (+0.002) | 41.59 (-0.61) | 36.5 (-5.9) | 10.0 (-190.9) | +0.3 | 2.73 (-10.67) | 1.21% (-0.7pp) | 0.99 (-0.06) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E1c-oneeuro | 0.253 (-0.031) | 54.31 (+12.11) | 36.7 (-5.7) | 15.9 (-185.0) | -0.9 | **2.29** (-11.10) | 0.76% (-1.1pp) | 3.42 (+2.37) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E1c-savgol | 0.283 (-0.001) | 41.93 (-0.27) | 36.6 (-5.8) | 9.9 (-191.0) | +0.4 | 2.62 (-10.77) | 1.34% (-0.5pp) | 3.56 (+2.51) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E1c-traj | **0.304** (+0.020) | **30.70** (-11.50) | 36.6 (-5.8) | 25.0 (-175.9) | -5.0 | 11.49 (-1.91) | 1.61% (-0.3pp) | 3.69 (+2.64) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E2-hawor | 0.296 (+0.012) | 44.74 (+2.55) | 36.3 (-6.1) | 78.2 (-122.8) | +77.8 | 9.88 (-3.51) | 0.37% (-1.5pp) | 1.43 (+0.39) | 0.419 (+0.419) | **0.00** (-0.19) | 93.7% (-6.3pp) | 7.7% |
| E2-hawor-cam | 0.295 (+0.011) | 44.90 (+2.70) | 36.3 (-6.1) | 78.2 (-122.8) | +77.8 | 9.84 (-3.56) | 0.43% (-1.5pp) | 1.43 (+0.39) | 0.419 (+0.419) | **0.00** (-0.19) | 93.7% (-6.3pp) | 0.0% |
| E2-hawor-cam-stereo | 0.297 (+0.013) | 44.90 (+2.70) | 34.9 (-7.5) | **7.7** (-193.2) | +0.0 | 4.12 (-9.27) | 0.32% (-1.6pp) | 2.05 (+1.00) | 0.419 (+0.419) | **0.00** (-0.19) | 93.7% (-6.3pp) | 0.0% |
| E2-hawor-cam-stereo-fs | 0.297 (+0.013) | 44.90 (+2.70) | **34.9** (-7.5) | 9.8 (-191.1) | -0.8 | 5.46 (-7.93) | 0.63% (-1.3pp) | 2.92 (+1.87) | 0.419 (+0.419) | **0.00** (-0.19) | 93.7% (-6.3pp) | 0.0% |
| E7tri-e0 | 0.235 (-0.049) | 56.28 (+14.08) | 42.4 (-0.0) | 40.9 (-160.1) | -12.8 | 4.16 (-9.24) | 0.83% (-1.1pp) | 2.29 (+1.25) | **0.000** (+0.000) | **0.00** (-0.19) | **100.0%** (+0.0pp) | 1.2% |
| E7tri-hawor | 0.275 (-0.009) | 67.06 (+24.86) | 36.3 (-6.1) | 31.7 (-169.2) | -13.1 | 3.20 (-10.20) | 0.28% (-1.6pp) | 1.83 (+0.79) | 0.419 (+0.419) | **0.00** (-0.19) | 93.7% (-6.3pp) | 0.0% |
| E8fallback-e0 | 0.286 (+0.002) | 37.54 (-4.66) | 36.3 (-6.1) | 35.2 (-165.7) | -13.5 | 5.45 (-7.95) | 1.04% (-0.8pp) | 2.78 (+1.73) | 0.478 (+0.478) | **0.00** (-0.19) | 97.0% (-3.0pp) | 0.0% |
| E8fallback-e1best | 0.294 (+0.010) | 32.90 (-9.30) | 36.0 (-6.4) | 28.5 (-172.5) | -8.9 | 6.28 (-7.12) | 1.32% (-0.6pp) | 2.71 (+1.67) | 0.489 (+0.489) | **0.00** (-0.19) | 97.9% (-2.1pp) | 0.0% |
| E8fallback-e1best-replace | 0.295 (+0.011) | 32.65 (-9.55) | 36.1 (-6.3) | 29.1 (-171.8) | -9.3 | 4.40 (-9.00) | 0.66% (-1.2pp) | 2.24 (+1.19) | 0.460 (+0.460) | **0.00** (-0.19) | 94.1% (-5.9pp) | 0.0% |
| E8fallback-e1best-smooth | 0.294 (+0.010) | 32.57 (-9.63) | 36.0 (-6.4) | 28.3 (-172.6) | -9.0 | 3.09 (-10.30) | 1.10% (-0.8pp) | 2.88 (+1.84) | 0.489 (+0.489) | **0.00** (-0.19) | 97.9% (-2.1pp) | 0.0% |
| TEAM-objfill_v2s | 0.288 (+0.004) | 47.29 (+5.10) | 38.7 (-3.8) | 30.8 (-170.2) | +28.7 | 4.56 (-8.83) | **0.21%** (-1.7pp) | 1.45 (+0.40) | **0.000** (+0.000) | **0.00** (-0.19) | 90.5% (-9.5pp) | 13.2% |

\* stereo residual uses SGBM depth from cam_b+cam_c: not a held-out metric, and circular for methods that consumed stereo depth.

## Common hand-frames (2349 hand-frames scored by every method with a tri-abc reference)

| method | Tri MPJPE abc (mm) | depth (mm) | lateral (mm) | depth bias (mm) | Tri MPJPE ab (mm) | cam_b kp (px) | cam_b PCK@10 | cam_a IoU | cam_a kp (px) | centred MPJPE (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 259.0 | 256.9 | 25.7 | +255.1 | 255.8 | 24.00 | 0.131 | 0.284 | 42.86 | 44.2 |
| E0-refined | 257.9 (-1.0) | 255.1 (-1.8) | 28.0 (+2.4) | +253.3 | 254.5 (-1.3) | 25.86 (+1.87) | 0.143 (+0.012) | 0.281 (-0.003) | 47.30 (+4.44) | 44.1 (-0.1) |
| E1a-betas | 52.5 (-206.5) | 46.0 (-210.9) | 17.3 (-8.3) | +9.8 | 60.1 (-195.7) | 19.48 (-4.51) | 0.264 (+0.133) | 0.284 (-0.000) | 42.63 (-0.22) | 38.8 (-5.4) |
| E1a-size-dz | 52.6 (-206.3) | 46.1 (-210.8) | 17.3 (-8.4) | +9.7 | 60.2 (-195.5) | 19.40 (-4.60) | 0.269 (+0.138) | 0.285 (+0.001) | 42.43 (-0.43) | 38.8 (-5.4) |
| E1a-size-dz-fs | 52.6 (-206.4) | 46.2 (-210.7) | 17.2 (-8.5) | +6.4 | 60.6 (-195.1) | 19.41 (-4.59) | 0.268 (+0.136) | 0.286 (+0.001) | 42.42 (-0.43) | 38.8 (-5.5) |
| E1b-dz | 61.2 (-197.7) | 52.2 (-204.6) | 23.7 (-2.0) | +24.9 | 67.3 (-188.4) | 24.72 (+0.72) | 0.121 (-0.011) | 0.225 (-0.059) | 55.48 (+12.63) | 44.3 (+0.1) |
| E1b-huber | 52.5 (-206.5) | 45.9 (-211.0) | 17.2 (-8.4) | +9.1 | 59.8 (-196.0) | 19.44 (-4.56) | 0.265 (+0.134) | 0.284 (-0.000) | 42.44 (-0.42) | 38.9 (-5.3) |
| E1b-scale | 52.7 (-206.3) | 46.1 (-210.8) | 17.3 (-8.3) | +9.6 | 60.2 (-195.6) | 19.49 (-4.50) | 0.264 (+0.133) | 0.284 (-0.000) | 42.63 (-0.22) | 38.8 (-5.4) |
| E1b-scale-fs | 52.6 (-206.4) | 46.2 (-210.7) | 17.2 (-8.4) | +6.3 | 60.6 (-195.2) | 19.51 (-4.49) | 0.263 (+0.132) | 0.284 (-0.000) | 42.63 (-0.22) | 38.8 (-5.4) |
| E1best-fs | 52.5 (-206.4) | 46.2 (-210.7) | 17.2 (-8.5) | +6.3 | 60.7 (-195.1) | 19.38 (-4.62) | 0.269 (+0.137) | 0.285 (+0.001) | 42.40 (-0.46) | 38.7 (-5.5) |
| E1best-sgbm | 52.6 (-206.4) | 46.1 (-210.8) | 17.3 (-8.4) | +9.7 | 60.2 (-195.6) | 19.36 (-4.63) | 0.270 (+0.139) | 0.285 (+0.001) | 42.40 (-0.46) | 38.8 (-5.5) |
| E1c-oneeuro | 54.1 (-204.9) | 46.3 (-210.5) | 20.2 (-5.5) | +8.7 | 61.8 (-193.9) | 23.97 (-0.02) | 0.200 (+0.068) | 0.268 (-0.017) | 49.85 (+7.00) | 38.9 (-5.3) |
| E1c-savgol | 52.6 (-206.4) | 46.1 (-210.8) | 17.3 (-8.3) | +9.6 | 60.1 (-195.6) | 19.45 (-4.55) | 0.265 (+0.134) | 0.284 (-0.000) | 42.59 (-0.26) | 38.8 (-5.4) |
| E1c-traj | 59.1 (-199.9) | 55.2 (-201.7) | 12.8 (-12.8) | +7.1 | 66.0 (-189.7) | 15.96 (-8.04) | **0.366** (+0.235) | 0.305 (+0.021) | 31.83 (-11.03) | 38.8 (-5.4) |
| E2-hawor | 126.9 (-132.1) | 124.7 (-132.2) | 16.3 (-9.4) | +118.5 | 121.6 (-134.1) | 19.66 (-4.33) | 0.213 (+0.082) | 0.305 (+0.021) | 33.86 (-9.00) | 38.9 (-5.3) |
| E2-hawor-cam | 126.9 (-132.1) | 124.7 (-132.2) | 16.3 (-9.4) | +118.5 | 121.6 (-134.1) | 19.66 (-4.33) | 0.213 (+0.082) | 0.305 (+0.021) | 33.86 (-9.00) | 38.9 (-5.3) |
| E2-hawor-cam-stereo | 50.5 (-208.5) | 45.8 (-211.1) | 13.6 (-12.1) | +15.1 | 55.0 (-200.7) | 17.08 (-6.91) | 0.262 (+0.130) | **0.307** (+0.023) | 33.86 (-9.00) | **37.3** (-6.9) |
| E2-hawor-cam-stereo-fs | **50.3** (-208.7) | **45.6** (-211.3) | 13.5 (-12.1) | +12.1 | **54.8** (-200.9) | 17.01 (-6.98) | 0.262 (+0.131) | 0.307 (+0.023) | 33.86 (-9.00) | 37.3 (-6.9) |
| E7tri-e0 | 58.2 (-200.8) | 52.6 (-204.3) | 16.8 (-8.8) | +13.2 | 73.9 (-181.9) | 21.22 (-2.78) | 0.210 (+0.079) | 0.250 (-0.035) | 44.96 (+2.11) | 44.2 (-0.0) |
| E7tri-hawor | 52.3 (-206.7) | 49.3 (-207.6) | 11.0 (-14.7) | +11.6 | 70.9 (-184.9) | 15.42 (-8.57) | 0.362 (+0.230) | 0.303 (+0.018) | 30.39 (-12.47) | 38.9 (-5.3) |
| E8fallback-e0 | 52.3 (-206.7) | 49.3 (-207.6) | 11.0 (-14.7) | +12.0 | 70.8 (-184.9) | 15.35 (-8.65) | 0.361 (+0.229) | 0.303 (+0.019) | 30.16 (-12.70) | 38.8 (-5.4) |
| E8fallback-e1best | 52.2 (-206.8) | 49.2 (-207.6) | 10.9 (-14.8) | +12.3 | 70.7 (-185.1) | 15.11 (-8.89) | 0.361 (+0.230) | 0.304 (+0.019) | 29.65 (-13.21) | 38.8 (-5.4) |
| E8fallback-e1best-replace | 52.2 (-206.8) | 49.2 (-207.6) | 10.9 (-14.8) | +12.3 | 70.7 (-185.1) | 15.11 (-8.89) | 0.362 (+0.230) | 0.304 (+0.019) | 29.65 (-13.21) | 38.8 (-5.4) |
| E8fallback-e1best-smooth | 52.1 (-206.9) | 49.1 (-207.7) | **10.8** (-14.9) | +12.3 | 70.3 (-185.4) | **14.98** (-9.01) | 0.365 (+0.234) | 0.304 (+0.020) | **29.43** (-13.43) | 38.8 (-5.4) |
| TEAM-objfill_v2s | 70.7 (-188.3) | 63.9 (-193.0) | 20.4 (-5.3) | +43.4 | 75.1 (-180.6) | 20.77 (-3.23) | 0.237 (+0.105) | 0.283 (-0.001) | 46.34 (+3.49) | 38.6 (-5.6) |

## Runs

| method | version | checkpoint | consumed stereo depth | GPU | wall time (s) | fps | scored hand-frames |
|---|---|---|---|---|---|---|---|
| E0 | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) | HaMeR (v2d_hamer weights) + WiLoR detector | no | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E0-refined | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) | HaMeR (v2d_hamer weights) + WiLoR detector | no | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E1a-betas | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 betas | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E1a-size-dz | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 betas size + E1 align dz | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E1a-size-dz-fs | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 betas size + E1 align dz | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E1b-dz | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align dz | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E1b-huber | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align huber | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E1b-scale | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E1b-scale-fs | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E1best-fs | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 betas size + E1 align dz + E1 smooth savgol | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E1best-sgbm | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 betas size + E1 align dz + E1 smooth savgol | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E1c-oneeuro | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 smooth oneeuro | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E1c-savgol | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 smooth savgol | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E1c-traj | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 smooth traj | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E2-hawor | HaWoR@66c7d41 (e2e) | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | no | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 9289 |
| E2-hawor-cam | HaWoR@66c7d41 (cam) | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | no | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 9289 |
| E2-hawor-cam-stereo | HaWoR@66c7d41 (cam) + E1 align scale | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | yes | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 9289 |
| E2-hawor-cam-stereo-fs | HaWoR@66c7d41 (cam) + E1 align scale | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | yes | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 9289 |
| E7tri-e0 | video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E7 tri(a+c)-shift + carry + sgbm + savgol | HaMeR (v2d_hamer weights) + WiLoR detector | yes | n/a (cached run; adapter on CPU) | – | – | 9950 |
| E7tri-hawor | HaWoR@66c7d41 (cam) + E7 tri(a+c)-shift + carry + sgbm + savgol | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | yes | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 9289 |
| E8fallback-e0 | HaWoR@66c7d41 (cam) + E7 tri(a+c)-shift + carry + sgbm + savgol; per-frame fallback: video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E7 tri(a+c)-shift + carry + sgbm + savgol | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | yes | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 9638 |
| E8fallback-e1best | HaWoR@66c7d41 (cam) + E7 tri(a+c)-shift + carry + sgbm + savgol; per-frame fallback: video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 betas size + E1 align dz + E1 smooth savgol | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | yes | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 9732 |
| E8fallback-e1best-replace | HaWoR@66c7d41 (cam) + E7 tri(a+c)-shift + carry + sgbm + savgol; per-frame fallback: video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 betas size + E1 align dz + E1 smooth savgol | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | yes | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 9320 |
| E8fallback-e1best-smooth | HaWoR@66c7d41 (cam) + E7 tri(a+c)-shift + carry + sgbm + savgol; per-frame fallback: video_to_data@8c778f9 run_ego_reconstruction --hand_tracking hamer (cached run outputs/track3_handfix) + E1 align scale + E1 betas size + E1 align dz + E1 smooth savgol | hawor.ckpt + infiller.pt (HF ThunderVVV/HaWoR), DROID-SLAM droid.pth, Metric3D vit-large | yes | NVIDIA GeForce RTX 4090 | 259.1936345100403 | 1.439 | 9732 |
| TEAM-objfill_v2s | team pipeline objfill_v2s (Rian): WiLoR + SAM2 + depth alignment + clean_hand_tracks + fill_hand_gaps_object | WiLoR | yes | n/a (team run) | – | – | 3965 |
