"""Step 6 tests for standard-DH extraction and MuJoCo FK validation."""

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from backend.dh import (
    dh_transform,
    factor_standard_dh,
    orientation_error,
    sample_qposes,
    validate_tables,
)
from backend.model_loader import compile_mjcf, model_summary


FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name):
    path = FIXTURES / name
    return compile_mjcf(path.read_bytes(), path.name)


class DhExtractionTests(unittest.TestCase):
    def test_factorization_recovers_standard_dh_transform(self):
        for theta, d, a, alpha in (
            (0.6, 0.2, 0.4, -1.1),
            (-0.8, -0.3, -0.25, 0.7),
            (np.pi, 0.0, 0.0, np.pi / 2),
        ):
            with self.subTest(theta=theta, alpha=alpha):
                expected = dh_transform(theta, d, a, alpha)
                actual = dh_transform(*factor_standard_dh(expected))
                np.testing.assert_allclose(actual, expected, atol=1e-12)
                self.assertLess(orientation_error(actual[:3, :3], expected[:3, :3]), 1e-7)

    def test_all_fixture_chains_have_one_verified_row_per_virtual_joint(self):
        expected = {
            "basic.xml": [2],
            "intermediate.xml": [3, 3],
            "complex.xml": [11, 11],
        }
        for name, counts in expected.items():
            with self.subTest(fixture=name):
                loaded = load_fixture(name)
                self.assertEqual([len(table.rows) for table in loaded.dh_tables.values()], counts)
                for chain in loaded.chains:
                    table = loaded.dh_tables[chain["id"]]
                    validation = table.validation
                    self.assertEqual(validation["status"], "verified")
                    self.assertEqual(validation["sample_count"], 13)
                    self.assertLessEqual(validation["max_fk_position_error_m"], validation["position_tolerance_m"])
                    self.assertLessEqual(validation["max_fk_orientation_error_rad"], validation["orientation_tolerance_rad"])
                    self.assertEqual([row["name"] for row in table.rows], [row["name"] for row in chain["virtual_joints"]])
                    for row in table.rows:
                        self.assertEqual(row["variable"], "theta" if row["kind"] == "revolute" else "d")
                summary = model_summary(loaded)
                self.assertEqual([entry["dh_table"]["validation"]["status"] for entry in summary["chains"]], ["verified"] * len(counts))

    def test_validation_poses_are_deterministic_and_move_complex_coordinates(self):
        loaded = load_fixture("complex.xml")
        first = sample_qposes(loaded.model, count=4)
        second = sample_qposes(loaded.model, count=4)
        self.assertEqual(len(first), 5)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(first[0], loaded.model.qpos0)
        self.assertTrue(any(np.linalg.norm(pose[:3] - first[0][:3]) > 0 for pose in first[1:]))
        self.assertTrue(any(np.linalg.norm(pose[3:7] - first[0][3:7]) > 0 for pose in first[1:]))
        self.assertTrue(any(np.linalg.norm(pose[7:11] - first[0][7:11]) > 0 for pose in first[1:]))
        for pose in first[1:]:
            self.assertAlmostEqual(float(np.linalg.norm(pose[3:7])), 1.0)
            self.assertAlmostEqual(float(np.linalg.norm(pose[7:11])), 1.0)

    def test_validation_flags_a_wrong_dh_parameter(self):
        loaded = load_fixture("basic.xml")
        tables = deepcopy(loaded.dh_tables)
        chain = loaded.chains[0]
        table = tables[chain["id"]]
        table.rows[0]["a"] += 0.1
        validate_tables(loaded.model, loaded.chains, tables, sample_qposes(loaded.model, count=3))
        self.assertEqual(table.validation["status"], "invalid")
        self.assertGreater(table.validation["max_fk_position_error_m"], table.validation["position_tolerance_m"])

    def test_jointless_model_has_no_dh_table(self):
        loaded = compile_mjcf(b'<mujoco><worldbody><body name="fixed"><geom type="sphere" size="0.1"/></body></worldbody></mujoco>', "fixed.xml")
        self.assertEqual(loaded.dh_tables, {})
        self.assertEqual(model_summary(loaded)["chains"], [])


if __name__ == "__main__":
    unittest.main()
