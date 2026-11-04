# Download log (budget: 3 GB for datasets and extra model checkpoints)

Total so far: **4.01 GB**. The user approved going over the 3 GB budget for AssemblyHands option B
(both val sequences, 31 Oct 2026). sha256 of every file: `data/assemblyhands/raw/manifest.tsv` (`tools/download_assemblyhands_val.sh`).

Excludes pip/conda packages, git clones of code, and the weights pre-placed in `weights-vggt/`.

| date | what | size | from | running total |
|---|---|---|---|---|
| 2026-10-31 | AssemblyHands `annotations/val/assemblyhands_val_ego_calib_v1-1.json` | 16.7 MB | Drive `11BXLwPQ9hZYu_doEUKDFR-s4zqzzlo8J` | 0.02 GB |
| 2026-10-31 | AssemblyHands `annotations/val/assemblyhands_val_ego_data_v1-1.json` | 84.1 MB | Drive `1-TNviWrqyf_0xBBiqYaGHkpVGuGiMSub` | 0.10 GB |
| 2026-10-31 | AssemblyHands `annotations/val/assemblyhands_val_joint_3d_v1-1.json` | 41.8 MB | Drive `1vvdFRwOynYdJVRWnlSFy5y1OphH7umiU` | 0.14 GB |
| 2026-10-31 | AssemblyHands `annotations/val/assemblyhands_val_exo_calib_v1-1.json` | 0.0 MB | Drive `1upansaUPxxxofabteTTXltSOKlAMJwCA` | 0.14 GB |
| 2026-10-31 | AssemblyHands `annotations/skeleton.txt` | 0.0 MB | Drive `1phxNZvJwEGqrxEwrD_AENZWr-nuzyCXe` | 0.14 GB |
| 2026-10-31 | AssemblyHands `annotations/README.txt` | 0.0 MB | Drive `1R2V1nhdPfNS9dKNNligqvwYtor0xfBM5` | 0.14 GB |
| 2026-10-31 | AssemblyHands `ego_images_rectified/val/nusar-2021_action_both_9081-c11b_9081_user_id_2021-02-12_161433.tar.gz` | 1,651.1 MB | Drive `1mryH9BQ7QQLqwAuhKMhDJ4-0bH1VWZGT` | 1.79 GB |
| 2026-10-31 | AssemblyHands `ego_images_rectified/val/nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345-008-001.tar.gz` | 2,215.8 MB | Drive `1jnNrtuvoYhEH0qQhu6GZ7VMAdyOl85cw` | 4.01 GB |

## Already on disk (not downloaded for this task)

| item | path | size | sha256 / note |
|---|---|---|---|
| VGGT-Ω 1B 512 | `weights-vggt/vggt_omega_1b_512.pt` | 4.58 GB | `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934` (matches HF metadata) |
| EgoFound3R | `weights-vggt/EgoFound3R.pt` | 424 MB | `02ba3135b030a23f88b91a986c173f6a3c600f6da45637605fd09914e9d78e66` (provided by the user, 31 Oct 2026; size matches HF) |
| WiLoR | `/usr1/e2s/video_to_data/reconstruction/data/weights/wilor` | 2.5 GB | upstream E0 weights |
| HaMeR | `/usr1/e2s/video_to_data/reconstruction/data/weights/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt` | 2.7 GB | `_DATA/data/{mano,models}` not readable (owned by another user); MANO from the WiLoR folder |
| MoGe-2 | `/usr1/e2s/video_to_data/reconstruction/data/weights/moge/model.pt` | 1.3 GB | |
| SAM2.1 large | `/usr1/e2s/video_to_data/reconstruction/data/weights/sam2` | 1.7 GB | |
| RTMPose hand | `../handtracking/assets/models/rtmpose` | | handbench |
| MANO right | `../handtracking/assets/mano_right.npz` | | handbench |
| HOT3D clips | `../handtracking/data/hot3d` | 834 MB | handbench (8 clips) |

## AssemblyHands Google Drive listing (no files downloaded)

`tools/list_gdrive.py` → `data/assemblyhands_listing/{annotations,ego_images}.tsv`.

- Annotations, val (public 3D labels): calib 16.7 MB + data 84.1 MB + joint_3d 41.8 MB = **143 MB**.
  test-iccv2023 has no public 3D labels (prediction template only); train annotations are 1.64 GB.
- Ego images, val: 2 sequences, one `.tar.gz` each: `9081-c11b` **1.5 GB**, `9012-c07c` **2.1 GB**.
  Test: 4 archives, 0.85–2.3 GB; train: 14 archives, 1.6–7.0 GB (46 GB in total).

## AssemblyHands val as downloaded (option B)

- 2 sequences / 2 subjects: `9012-c07c` and `9081-c11b`; 4 headset mono cameras each (636×480, rectified).
- Labelled hand-frames (≥ 10 valid joints): 16,348 (9012) + 13,262 (9081); 98 % are seen by ≥ 2 cameras, so F2/F3
  have a real second view (9012: HMC_84355350 + HMC_84358933; 9081: HMC_21110305 + HMC_21179183).
- Labels every 2nd frame; per-frame extrinsics (moving headset). Keypoints: 21 per hand, tip → base, wrist = 20
  (`annotations/skeleton.txt`).
- The `.tar.gz` archives are plain tar; extracted to `data/assemblyhands/ego_images_rectified/val/` (counts match the
  annotations per camera).
