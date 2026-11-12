"""Regression checks that use captured motion and targeted perturbations."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark import error, inject
from bvh import BVH, matrices
from cli import process
from quality import measure
from refine import MODULES, config, detect, refine

DATA = Path(__file__).resolve().parent / "data"


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = config()
        cls.path = DATA / "J_ShortDance16_JazzWalk.bvh"
        cls.bvh = BVH.read(cls.path)
        cls.clean = cls.bvh.decode()

    def test_bvh_zero_is_byte_identical(self):
        for settings in (
            self.settings | {"strength": 0},
            self.settings | {key: {"enabled": True, "strength": 0} for key in MODULES},
            self.settings | {key: {"enabled": False, "strength": 100} for key in MODULES},
        ):
            with tempfile.TemporaryDirectory() as tmp:
                process(self.path, Path(tmp), settings)
                self.assertEqual((Path(tmp) / f"{self.path.stem}.refined.bvh").read_bytes(),
                                 self.path.read_bytes())

    def test_foot_only_strength_changes_constraint_and_keeps_timing(self):
        foot_only = self.settings | {key: self.settings[key] | {"enabled": False}
                                     for key in ("smooth", "torso", "speed")}
        low = refine(self.bvh, foot_only | {"strength": 15})
        high = refine(self.bvh, foot_only | {"strength": 100})
        np.testing.assert_array_equal(low[2].mask, high[2].mask)
        self.assertTrue(np.array_equal(high[0], self.clean[0]))
        key = "support_skate_m_s"
        self.assertLess(measure(self.bvh, high[:2], high[2])[key],
                        measure(self.bvh, low[:2], low[2])[key])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "round_trip.bvh"
            self.bvh.write(out, *high[:2])
            result = BVH.read(out)
            self.assertEqual(result.header, self.bvh.header)
            self.assertEqual(len(result.table), len(self.bvh.table))
            self.assertEqual(result.dt, self.bvh.dt)
            a = self.bvh.fk(*high[:2])[0]
            b = result.fk(*result.decode())[0]
            np.testing.assert_allclose(a, b, atol=1e-7, rtol=0)

    def test_rotation_and_root_only_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "root.bvh"
            root, local = self.clean
            shifted = root + [0, 0.1, 0]
            self.bvh.write(path, shifted, local)
            check_root, check_local = BVH.read(path).decode()
            np.testing.assert_allclose(check_root, shifted, atol=1e-9, rtol=0)
            np.testing.assert_allclose(check_local, local, atol=1e-9, rtol=0)

            path = Path(tmp) / "rotation.bvh"
            changed = local.copy()
            changed[20:28, 0] = changed[20:28, 0] @ matrices(np.tile([0, np.pi * 0.96, 0], (8, 1)))
            self.bvh.write(path, root, changed)
            np.testing.assert_allclose(BVH.read(path).decode()[1], changed, atol=1e-8, rtol=0)

    def test_airborne_core_keeps_leg_pose(self):
        root, local = self.clean
        lifted = root.copy()
        start = len(root) // 2
        lifted[start:start + 60, 1] += 0.30 * np.hanning(60)
        core = slice(start + 18, start + 42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "airborne.bvh"
            self.bvh.write(path, lifted, local)
            motion = BVH.read(path)
            foot_only = self.settings | {key: self.settings[key] | {"enabled": False}
                                         for key in ("smooth", "torso", "speed")}
            _, corrected, contacts = refine(motion, foot_only | {"strength": 100})
            self.assertFalse(contacts.mask[core].any())
            np.testing.assert_array_equal(corrected[core], motion.decode()[1][core])

    def test_joint_jitter_improves_against_clean_reference(self):
        motion = BVH.read(DATA / "J_Dance0_StepTouch.bvh")
        clean = motion.decode()
        ref = detect(motion, *motion.fk(*clean), self.settings)
        root, local, idx, mode, _ = inject(motion, clean, ref, "elbow_jitter")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jitter.bvh"
            motion.write(path, root, local)
            damaged = BVH.read(path)
            settings = self.settings | {key: self.settings[key] | {"enabled": False}
                                        for key in ("foot", "torso", "speed")}
            after = refine(damaged, settings)
            a = error(motion, clean, damaged.decode(), idx, mode)
            b = error(motion, clean, after[:2], idx, mode)
            self.assertLess(b, 0.85 * a)


if __name__ == "__main__":
    unittest.main()
