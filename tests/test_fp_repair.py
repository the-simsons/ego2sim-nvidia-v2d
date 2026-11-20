"""tools/fp_repair.repair on a synthetic track: a failure jump, a wrong plateau, and a re-registration jump back."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fp_repair  # noqa: E402
from e2s.geometry import rotation_angle_deg  # noqa: E402


class FpRepairTest(unittest.TestCase):
    def test_failure_stretch_is_carried_by_neighbours(self):
        N = 200
        ang = np.clip((np.arange(N) - 60) * 2.0, 0, 80)                    # the object turns 80 deg at 2 deg/frame
        truth = Rotation.from_rotvec(np.outer(np.radians(ang), [0, 0, 1])).as_matrix()
        err = Rotation.from_rotvec(np.radians(70) * np.array([1, 0, 0])).as_matrix()
        tracked = truth.copy()
        tracked[80:140] = truth[80:140] @ err                                  # lost at 80, re-registered at 140
        score = np.full(N, 0.85)
        score[80:140] = 0.55
        off, jumps, segs, info = fp_repair.repair(tracked, score)
        fixed = tracked @ off
        self.assertEqual(list(jumps), [80, 140])
        self.assertLess(rotation_angle_deg(fixed, truth).max(), 5.0)

    def test_clean_track_is_untouched(self):
        N = 120
        truth = Rotation.from_rotvec(np.outer(np.radians(np.linspace(0, 60, N)), [0.3, 0.2, 1])).as_matrix()
        off, jumps, segs, info = fp_repair.repair(truth, np.full(N, 0.8))
        self.assertEqual(len(jumps), 0)
        self.assertLess(rotation_angle_deg(off, np.repeat(np.eye(3)[None], N, 0)).max(), 1e-9)


if __name__ == "__main__":
    unittest.main()
