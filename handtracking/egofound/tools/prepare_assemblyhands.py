"""AssemblyHands val (ego, rectified mono headset cameras) -> fixed-length clips for the monocular methods + ground truth.

Per sequence, the input camera and the second view are the two cameras labelled on (almost) every frame. Images are
released every 2nd frame of the 30 Hz capture, so clips play at 15 fps. Mono images are written as 3 identical
channels (same input for every method). The images are already rectified pinhole (calibration K); no undistortion.
Both input cameras are mounted sideways (the wearer's body is at the image's left), so frames are rotated 90 deg
counter-clockwise to an upright view (as the HOT3D frames were); K and the ground truth are rotated with them
(upright camera: x' = y, y' = -x).

Clips: N_CLIPS windows of CLIP_LEN consecutive released frames per sequence (no gap in the release), evenly spaced.
Visibility: the 3D and 2D `joint_valid` flags are 1 even for hands outside the view; a hand counts as visible in a
camera when the per-image annotation has a box for it (`bbox[side]` not null).

Ground truth (annotations/val/*joint_3d*: world coordinates, mm; *ego_calib*: per-frame world -> camera [R|t], mm)
is written in the input camera's frame, metres, in handbench's joint order (OpenPose: wrist, then thumb, index,
middle, ring, pinky, each base -> tip). AssemblyHands per hand: index 4f + (4 - k) is finger f's k-th joint from the
base (k = 4 is the tip), wrist = 20; right hand = 0..20, left = 21..41 (annotations/skeleton.txt).

  ../.venv-bench/bin/python tools/prepare_assemblyhands.py [--n-clips 4] [--clip-len 300]
Writes data/assemblyhands_input/<clip>.mp4, <clip>.json (K, size, fps, frames, cameras) and <clip>_gt.npz:
  joints_cam (T,2,21,3) input camera, m; valid (T,2,21) bool (3D label); side 0 = L, 1 = R
  visible (T,2), visible2 (T,2): hand has a 2D box in the input / second camera; box (T,2,4) input camera x0,y0,x1,y1
  T_cam2_from_cam (T,4,4) input camera -> second camera (m); K2 (3,3)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AH = ROOT / "data" / "assemblyhands"
ANN = AH / "raw" / "annotations" / "val"
FPS = 15.0  # 30 Hz capture, every 2nd frame released
# input camera, second view (the two labelled on almost every frame)
CAMERAS = {"nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345": ("HMC_84355350", "HMC_84358933"),
           "nusar-2021_action_both_9081-c11b_9081_user_id_2021-02-12_161433": ("HMC_21110305", "HMC_21179183")}
AH_TO_OPENPOSE = [20] + [4 * f + (4 - k) for f in range(5) for k in range(1, 5)]
R_UP = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])  # camera -> upright camera (image rotated 90 deg CCW)


def upright_K(K: np.ndarray, W: int) -> np.ndarray:
    return np.array([[K[1, 1], 0, K[1, 2]], [0, K[0, 0], W - 1 - K[0, 2]], [0, 0, 1.0]])


def upright_box(b, W: int):
    x0, y0, x1, y1 = b
    return [y0, W - 1 - x1, y1, W - 1 - x0]


def clip_name(seq: str, start: int) -> str:
    return f"{seq.split('_')[3]}_{start:06d}"  # e.g. 9012-c07c_000520


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--clip-len", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "data" / "assemblyhands_input"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = json.loads((ANN / "assemblyhands_val_ego_calib_v1-1.json").read_text())["calibration"]
    j3d = json.loads((ANN / "assemblyhands_val_joint_3d_v1-1.json").read_text())["annotations"]
    ego = json.loads((ANN / "assemblyhands_val_ego_data_v1-1.json").read_text())
    imgs = {i["id"]: i for i in ego["images"]}
    boxes = {(imgs[x["image_id"]]["seq_name"], imgs[x["image_id"]]["camera"], imgs[x["image_id"]]["frame_idx"]): x["bbox"]
             for x in ego["annotations"]}
    for seq, (cam, cam2) in CAMERAS.items():
        c1, c2 = cam + "_mono10bit", cam2 + "_mono10bit"
        img_dir = AH / "ego_images_rectified" / "val" / seq / c1
        frames = sorted(int(p.stem) for p in img_dir.glob("*.jpg"))
        ok_start = [i for i in range(len(frames) - a.clip_len + 1)
                    if frames[i + a.clip_len - 1] - frames[i] == 2 * (a.clip_len - 1)]
        K = np.array(calib[seq]["intrinsics"][c1], np.float64)
        K2 = np.array(calib[seq]["intrinsics"][c2], np.float64)
        want = np.linspace(0, len(frames) - a.clip_len, a.n_clips)
        starts = [min(ok_start, key=lambda i: abs(i - w)) for w in want]
        for s0 in starts:
            fr = frames[s0:s0 + a.clip_len]
            name = clip_name(seq, fr[0])
            T = len(fr)
            J = np.zeros((T, 2, 21, 3), np.float32)
            V = np.zeros((T, 2, 21), bool)
            T21 = np.tile(np.eye(4), (T, 1, 1))
            vis, vis2, box = np.zeros((T, 2), bool), np.zeros((T, 2), bool), np.full((T, 2, 4), np.nan, np.float32)
            for t, f in enumerate(fr):
                for c, vv in ((cam, vis), (cam2, vis2)):
                    b = boxes.get((seq, c, f))
                    if b is None:
                        continue
                    for s, side in ((0, "left"), (1, "right")):
                        if b.get(side) is not None:
                            vv[t, s] = True
                            if c == cam:
                                box[t, s] = b[side]
            for t, f in enumerate(fr):
                key = f"{f:06d}"
                ext = calib[seq]["extrinsics"].get(key, {})
                if key not in j3d[seq] or c1 not in ext:
                    continue
                E1 = np.vstack([np.array(ext[c1]), [0, 0, 0, 1]])
                Xw = np.array(j3d[seq][key]["world_coord"], np.float64)
                ok = np.array(j3d[seq][key]["joint_valid"], bool).reshape(-1)
                Xc = (Xw @ E1[:3, :3].T + E1[:3, 3]) / 1000.0
                for s, off in ((0, 21), (1, 0)):  # L = 21..41, R = 0..20
                    idx = [off + i for i in AH_TO_OPENPOSE]
                    J[t, s], V[t, s] = Xc[idx], ok[idx]
                if c2 in ext:
                    E2 = np.vstack([np.array(ext[c2]), [0, 0, 0, 1]])
                    T21[t] = E2 @ np.linalg.inv(E1)
                    T21[t, :3, 3] /= 1000.0
            W, H = cv2.imread(str(img_dir / f"{fr[0]:06d}.jpg")).shape[1::-1]
            J = (J @ R_UP.T).astype(np.float32)
            R4 = np.eye(4)
            R4[:3, :3] = R_UP
            T21 = T21 @ R4.T  # upright input camera -> second camera
            box = np.array([[upright_box(b, W) for b in bb] for bb in box], np.float32)
            Ku = upright_K(K, W)
            dst = out / f"{name}.mp4"
            ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{H}x{W}", "-r", f"{FPS}", "-i", "-", "-c:v", "libx264", "-crf", "12",
                                   "-preset", "medium", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
            for f in fr:
                g = cv2.rotate(cv2.imread(str(img_dir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE),
                               cv2.ROTATE_90_COUNTERCLOCKWISE)
                ff.stdin.write(np.ascontiguousarray(cv2.merge([g, g, g])).tobytes())
            ff.stdin.close()
            ff.wait()
            np.savez_compressed(out / f"{name}_gt.npz", joints_cam=J, valid=V, visible=vis, visible2=vis2, box=box,
                                T_cam2_from_cam=T21, K2=K2)
            hf = int(((V.sum(-1) >= 10) & vis).sum())
            json.dump({"clip": name, "sequence": seq, "camera": cam, "second_camera": cam2, "K": Ku.tolist(),
                       "K_rectified_sideways": K.tolist(), "rotation": "90 deg CCW (upright)", "size": [H, W], "fps": FPS, "n_frames": T, "frames": fr, "labelled_hand_frames": hf,
                       "source": str(img_dir)}, open(out / f"{name}.json", "w"), indent=1)
            print(f"{dst}: {T} frames, {hf} visible hand-frames with >= 10 valid joints")


if __name__ == "__main__":
    main()
