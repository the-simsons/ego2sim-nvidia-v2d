"""tools/sym_track.py on synthetic tracks: mid-clip look-alike flips are unwrapped, frame 0 comes from the still stretch."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import sym_track  # noqa: E402
from e2s.geometry import load_pose_dir, pose_to_dict, rotation_angle_deg  # noqa: E402

FPS = 30.0
BOX_OBJ = "".join(f"v {x} {y} {z}\n" for x in (-.15, .15) for y in (-.075, .075) for z in (0.0, .08))  # origin on the base


def world_track(n=150):
    T = np.tile(np.eye(4), (n, 1, 1))
    s = np.clip((np.arange(n) - 40) / 80.0, 0, 1)
    T[:, :3, :3] = Rotation.from_rotvec(np.outer(s, [0.3, 0.2, 1.2])).as_matrix()
    T[:, :3, 3] = np.c_[0.3 * s, 0.1 * s, 0.6 + 0.1 * np.sin(np.pi * s)]
    return T


class SymTrackTest(unittest.TestCase):
    def run_tool(self, Tcm, c2w, extra=()):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "in").mkdir()
            for i, T in enumerate(Tcm):
                json.dump(pose_to_dict(T), open(t / "in" / f"{i:06d}.json", "w"))
            np.savez(t / "cam.npz", camera_to_world=c2w)
            (t / "box.obj").write_text(BOX_OBJ)
            sym_track.main(["--poses", str(t / "in"), "--camera", str(t / "cam.npz"), "--mesh", str(t / "box.obj"),
                            "--symmetry", '{"flips_180": ["x", "y", "z"]}', "--fps", str(FPS), "--out", str(t / "out"), *extra])
            return load_pose_dir(t / "out", len(Tcm)), json.load(open(t / "out" / "sym_track.json"))

    def test_mid_clip_flips_are_unwrapped(self):
        W = world_track()
        c2w = np.tile(np.eye(4), (len(W), 1, 1))
        c2w[:, :3, 3] = np.c_[np.linspace(0, 0.4, len(W)), np.zeros(len(W)), np.zeros(len(W))]
        S = sym_track.twins({"flips_180": ["x", "y", "z"]}, np.array([0, 0, 0.04]))
        Wr = W.copy()
        Wr[60:90] = Wr[60:90] @ S[1]
        Wr[110:115] = Wr[110:115] @ S[3]
        out, st = self.run_tool(np.linalg.inv(c2w) @ Wr, c2w, ["--hold", "none"])
        Wo = c2w @ out
        self.assertLess(rotation_angle_deg(Wo[:, :3, :3], W[:, :3, :3]).max(), 1e-4)
        self.assertEqual(st["out_jumps_gt90deg"], 0)
        self.assertEqual(st["raw_jumps_gt90deg"], 4)

    def test_frame0_from_still_stretch(self):
        rng = np.random.default_rng(0)
        W = world_track()
        noisy = W.copy()
        noisy[:, :3, 3] += rng.normal(0, 0.003, (len(W), 3))
        noisy[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, np.radians(1.0), (len(W), 3))).as_matrix() @ W[:, :3, :3]
        c2w = np.tile(np.eye(4), (len(W), 1, 1))
        out, st = self.run_tool(noisy, c2w)
        self.assertEqual(len(st["held_still"]), 1)
        e_raw = np.linalg.norm(noisy[0, :3, 3] - W[0, :3, 3])
        e_out = np.linalg.norm(out[0, :3, 3] - W[0, :3, 3])
        self.assertLess(e_out, 0.0015)
        self.assertLess(e_out, e_raw + 1e-9)


if __name__ == "__main__":
    unittest.main()
