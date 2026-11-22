"""Unit tests for the benchmark metrics, provenance records and the Motive CSV loader (synthetic data only).
Run from the repository root: python -m unittest discover tests"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bench import metrics
from e2s import provenance
from e2s.geometry import box_symmetries

FPS = 30.0
BOX = np.array([[x, y, z] for x in (-.15, .15) for y in (-.075, .075) for z in (-.04, .04)])


def trajectory(n=180):
    """Still for 1 s, then a lift-and-turn, then still again (world <- mesh)."""
    T = np.tile(np.eye(4), (n, 1, 1))
    s = np.clip((np.arange(n) - 30) / 90.0, 0, 1)
    T[:, :3, :3] = Rotation.from_rotvec(np.outer(s, [0, 0, np.pi / 2])).as_matrix()
    T[:, :3, 3] = np.c_[0.3 * s, 0.1 * s, 0.75 + 0.1 * np.sin(np.pi * s)]
    return T


def rigid(rotvec, t):
    G = np.eye(4)
    G[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    G[:3, 3] = t
    return G


class MetricsTest(unittest.TestCase):
    def test_perfect_up_to_a_world_transform(self):
        gt = trajectory()
        rec = np.linalg.inv(rigid([0.3, -0.2, 1.0], [1, 2, 3])) @ gt
        s, _ = metrics.object_metrics(rec, gt, BOX, FPS)
        self.assertLess(s["pos_cm_max"], 1e-6)
        self.assertLess(s["rot_deg_max"], 1e-4)
        self.assertAlmostEqual(s["adds_auc"], 1.0, delta=0.01)

    def test_look_alike_start_is_absorbed(self):
        """A tracker that reports the box turned 180 deg throughout scores as perfect (bare cuboid symmetry)."""
        gt = trajectory()
        rec = gt @ box_symmetries()[2]
        s, _ = metrics.object_metrics(rec, gt, BOX, FPS)
        self.assertLess(s["rot_deg_max"], 1e-4)
        self.assertIn("symmetry", s["alignment"])

    def test_constant_offset_after_alignment_is_measured(self):
        gt = trajectory()
        rec = gt.copy()
        rec[60:, :3, 3] += [0.05, 0, 0]          # 5 cm error after the first still stretch
        s, pf = metrics.object_metrics(rec, gt, BOX, FPS)
        self.assertAlmostEqual(np.nanmedian(pf["pos_cm"][60:]), 5.0, places=3)
        self.assertLess(np.nanmax(pf["pos_cm"][:30]), 1e-6)

    def test_mid_clip_flip_is_flagged(self):
        gt = trajectory()
        rec = gt.copy()
        rec[100:140] = rec[100:140] @ box_symmetries()[1]
        s, pf = metrics.object_metrics(rec, gt, BOX, FPS)
        self.assertTrue(pf["flip"][100:140].all())
        self.assertFalse(pf["flip"][:100].any())
        self.assertAlmostEqual(s["flip_longest_s"], 40 / FPS, places=6)

    def test_missing_frames_lower_coverage(self):
        gt = trajectory()
        rec = gt.copy()
        rec[90:120] = np.nan
        s, _ = metrics.object_metrics(rec, gt, BOX, FPS)
        self.assertAlmostEqual(s["coverage"], 150 / 180)
        self.assertLess(s["pos_cm_max"], 1e-6)

    def test_auc(self):
        self.assertAlmostEqual(metrics.auc([0.0] * 10), 1.0, places=2)
        self.assertEqual(metrics.auc([20.0] * 10), 0.0)
        self.assertAlmostEqual(metrics.auc([5.0] * 10), 0.5, places=2)


class ProvenanceTest(unittest.TestCase):
    def test_rules(self):
        with self.assertRaises(ValueError):
            provenance.make("video", ["camera_path"])
        with self.assertRaises(ValueError):
            provenance.make("mocap", [])
        with self.assertRaises(ValueError):
            provenance.make("mixed", ["teleport"])

    def test_gate(self):
        vid = provenance.make("video", depth_source="stereo")
        moc = provenance.make("mocap", ["object_trajectory"])
        self.assertEqual(provenance.require_video(vid, final=True), "video")
        self.assertEqual(provenance.require_video(moc, final=False), "mocap")
        with self.assertRaises(RuntimeError):
            provenance.require_video(moc, final=True)
        with self.assertRaises(RuntimeError):
            provenance.require_video(None, final=True)

    def test_read_searches_parents(self):
        with tempfile.TemporaryDirectory() as t:
            provenance.write(t, provenance.make("mixed", ["sync"]))
            deep = Path(t) / "a" / "b"
            deep.mkdir(parents=True)
            self.assertEqual(provenance.read(deep)["trajectory_source"], "mixed")


MOTIVE = """Format Version,1.23,Take Name,t,Capture Frame Rate,120.000000,Export Frame Rate,120.000000,Rotation Type,Quaternion,Length Units,Millimeters,Coordinate Space,Global
,
Type,,Rigid Body,Rigid Body,Rigid Body,Rigid Body,Rigid Body,Rigid Body,Rigid Body
Name,,Box,Box,Box,Box,Box,Box,Box
ID,,1,1,1,1,1,1,1
,,Rotation,Rotation,Rotation,Rotation,Position,Position,Position
Frame,Time (Seconds),X,Y,Z,W,X,Y,Z
0,0.000000,0,0,0,1,100,200,750
1,0.008333,0,0,0.7071068,0.7071068,110,200,750
"""


class MotiveTest(unittest.TestCase):
    def test_load(self):
        from e2s.mocap import motive
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "take.csv"
            p.write_text(MOTIVE)
            d = motive.load(p)
        self.assertAlmostEqual(d["fps"], 120.0)
        b = d["bodies"]["Box"]
        np.testing.assert_allclose(b["pos"][0], [0.1, 0.2, 0.75])     # millimetres -> metres
        np.testing.assert_allclose(b["quat_xyzw"][1], [0, 0, 0.7071068, 0.7071068], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
