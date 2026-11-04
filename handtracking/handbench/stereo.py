"""SGBM stereo depth from the ego mono pair (left=cam_c, right=cam_b) with the refined rig, splatted into the
benchmark camera (undistorted cam_a pinhole). Same SGBM settings as tools/ego_stereo_depth.py.

NOTE (independence): this depth uses cam_b. Metrics built on it are not held-out-view metrics.
"""
from __future__ import annotations

import cv2
import numpy as np

from handbench.calib import Cam


class StereoDepth:
    def __init__(self, cams: dict, num_disp: int = 192, out_scale: float = 1.0):
        L, Rc = cams["ego_cam_c"], cams["ego_cam_b"]
        T_l_r = Rc.T_from_a @ np.linalg.inv(L.T_from_a)  # c frame -> b frame
        R, t = T_l_r[:3, :3], T_l_r[:3, 3].reshape(3, 1)
        size = L.size
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(L.K, L.dist, Rc.K, Rc.dist, size, R, t,
                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.mapl = cv2.initUndistortRectifyMap(L.K, L.dist, R1, P1, size, cv2.CV_32FC1)
        self.mapr = cv2.initUndistortRectifyMap(Rc.K, Rc.dist, R2, P2, size, cv2.CV_32FC1)
        self.f, self.baseline = P1[0, 0], -P2[0, 3] / P2[0, 0]
        W, H = size
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        self.rays = (np.linalg.inv(P1[:3, :3]) @ np.stack([uu.ravel(), vv.ravel(), np.ones(W * H)])).T
        # rectified-left -> cam_c -> cam_a
        T_c_from_a = L.T_from_a
        self.R_a_from_rect = T_c_from_a[:3, :3].T @ R1.T
        self.t_a_from_c = -T_c_from_a[:3, :3].T @ T_c_from_a[:3, 3]
        A = cams["ego_cam_a"]
        self.Ka = A.K.copy()
        self.Wa, self.Ha = A.size
        self.out_scale = out_scale
        bs = 5
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=bs, P1=8 * bs * bs,
                                          P2=32 * bs * bs, disp12MaxDiff=1, uniquenessRatio=10,
                                          speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def disparity(self, Lr: np.ndarray, Rr: np.ndarray) -> np.ndarray:
        """Rectified grey left/right -> disparity (px, rectified-left grid)."""
        return self.sgbm.compute(Lr, Rr).astype(np.float32) / 16.0

    def compute(self, img_c: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        """Grey/BGR cam_c + cam_b frames -> depth (m, 0 = invalid) in the benchmark cam_a pinhole, at out_scale."""
        g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if x.ndim == 3 else x
        Lr = cv2.remap(g(img_c), *self.mapl, cv2.INTER_LINEAR)
        Rr = cv2.remap(g(img_b), *self.mapr, cv2.INTER_LINEAR)
        disp = self.disparity(Lr, Rr).ravel()
        valid = disp > 1.0
        z = np.zeros_like(disp)
        z[valid] = self.f * self.baseline / disp[valid]
        valid &= (z > 0.15) & (z < 5.0)
        X = (self.rays[valid] * z[valid, None]) @ self.R_a_from_rect.T + self.t_a_from_c
        X = X[X[:, 2] > 0.1]
        s = self.out_scale
        W, H = int(round(self.Wa * s)), int(round(self.Ha * s))
        K = self.Ka.copy()
        K[:2] *= s
        K[:2, 2] = (self.Ka[:2, 2] + 0.5) * s - 0.5
        uv = X @ K.T
        u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
        v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
        ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        depth = np.full((H, W), np.inf, np.float32)
        np.minimum.at(depth, (v[ins], u[ins]), X[ins, 2].astype(np.float32))
        depth[~np.isfinite(depth)] = 0
        if s > 0.6:  # splat is sparser than the target grid: fill 1-px holes with a min filter
            big = np.where(depth > 0, depth, 1e6).astype(np.float32)
            er = cv2.erode(big, np.ones((3, 3), np.uint8))
            hole = (depth == 0) & (er < 1e5)
            depth[hole] = er[hole]
        return depth


FS_ONNX = "assets/models/foundation_stereo/deployable_foundationstereo_small_576x960_v2.0.onnx"
FS_W, FS_H = 960, 576
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], np.float32)


class FoundationStereoDepth(StereoDepth):
    """Same rectification and cam_a splat as StereoDepth, disparity from NVIDIA's deployable FoundationStereo (small,
    576x960 ONNX, a copy under handtracking/assets) on onnxruntime CUDA. Preprocessing follows upstream
    v2d_foundation_stereo/lib/trt_inference.py: aspect-preserving resize + replicate pad to 960x576, RGB, ImageNet
    mean/std; disparity cropped, divided by the resize scale and resized back to the rectified grid."""

    def __init__(self, cams: dict, out_scale: float = 1.0, onnx_path: str | None = None):
        super().__init__(cams, out_scale=out_scale)
        import onnxruntime as ort

        from handbench import paths

        path = onnx_path or str(paths.ROOT / FS_ONNX)
        if hasattr(ort, "preload_dlls"):  # cuDNN/cuBLAS come from the venv's nvidia-* pip wheels, not the system
            ort.preload_dlls()
        self.sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.provider = self.sess.get_providers()[0]

    def _prep(self, grey: np.ndarray):
        h, w = grey.shape[:2]
        s = min(FS_W / w, FS_H / h)
        nw, nh = int(w * s), int(h * s)
        im = cv2.resize(grey, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        im = cv2.copyMakeBorder(im, 0, FS_H - nh, 0, FS_W - nw, cv2.BORDER_REPLICATE)
        rgb = np.repeat(im[..., None], 3, axis=2).astype(np.float32)  # mono camera: same value in R, G, B
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), (s, nw, nh, w, h)

    def disparity(self, Lr: np.ndarray, Rr: np.ndarray) -> np.ndarray:
        L, (s, nw, nh, w, h) = self._prep(Lr)
        R, _ = self._prep(Rr)
        d = self.sess.run(["disparity"], {"left_image": L, "right_image": R})[0].reshape(FS_H, FS_W)
        d = d[:nh, :nw] / s
        return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def scaled_K(K: np.ndarray, s: float) -> np.ndarray:
    K2 = K.copy()
    K2[:2, :2] *= s
    K2[:2, 2] = (K[:2, 2] + 0.5) * s - 0.5
    return K2
