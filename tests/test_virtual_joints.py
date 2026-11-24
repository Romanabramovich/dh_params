"""Step 4 tests for virtual coordinates and MuJoCo pose reconstruction."""

from pathlib import Path
import unittest

import mujoco
import numpy as np

from backend.model_loader import compile_mjcf
from backend.virtual_joints import compose_virtual_motion, reconstruct_chain_tip, virtualize_chain, virtualize_joint


FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name):
    path = FIXTURES / name
    return compile_mjcf(path.read_bytes(), path.name)


def normalized(values):
    vector = np.asarray(values, dtype=np.float64)
    return vector / np.linalg.norm(vector)


class VirtualJointTests(unittest.TestCase):
    def assert_tip_matches_mujoco(self, loaded, chain):
        reconstructed = reconstruct_chain_tip(loaded.model, chain, loaded.data.qpos)
        tip_id = chain["tip_body_id"]
        np.testing.assert_allclose(reconstructed[:3, 3], loaded.data.xpos[tip_id], atol=1e-9)
        np.testing.assert_allclose(reconstructed[:3, :3], loaded.data.xmat[tip_id].reshape(3, 3), atol=1e-9)

    def test_fixture_expansion_names_axes_and_qpos_mapping(self):
        basic = load_fixture("basic.xml")
        self.assertEqual([row["name"] for row in basic.chains[0]["virtual_joints"]], ["shoulder", "extension"])

        intermediate = load_fixture("intermediate.xml")
        self.assertEqual([len(chain["virtual_joints"]) for chain in intermediate.chains], [3, 3])

        complex_model = load_fixture("complex.xml")
        for chain in complex_model.chains:
            rows = chain["virtual_joints"]
            self.assertEqual(len(rows), 11)
            self.assertEqual([row["name"] for row in rows[:9]], [
                "base_free.tx", "base_free.ty", "base_free.tz", "base_free.yaw", "base_free.pitch", "base_free.roll",
                "ball.yaw", "ball.pitch", "ball.roll",
            ])
            self.assertEqual([row["axis"] for row in rows[:6]], [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
            self.assertEqual(rows[2]["value"], 0.5)
            self.assertEqual(rows[3]["qpos_range"], [3, 7])
            self.assertEqual(rows[6]["qpos_range"], [7, 11])
            self.assertEqual([row["source_joint_id"] for row in rows[:9]], [0] * 6 + [1] * 3)

    def test_ball_quaternion_reconstructs_orientation_and_warns_near_singularity(self):
        loaded = load_fixture("complex.xml")
        for quaternion, expect_warning in (
            (normalized([0.7, 0.2, -0.3, 0.6]), False),
            ([np.cos((np.pi / 2 - 1e-5) / 2), 0, np.sin((np.pi / 2 - 1e-5) / 2), 0], True),
            ([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0], True),
            ([np.cos(np.pi / 4), 0, -np.sin(np.pi / 4), 0], True),
        ):
            with self.subTest(quaternion=quaternion):
                qpos = loaded.model.qpos0.copy()
                qpos[7:11] = quaternion
                rows, warnings = virtualize_joint(loaded.model, 1, qpos)
                actual = np.empty(9)
                mujoco.mju_quat2Mat(actual, normalized(quaternion))
                np.testing.assert_allclose(compose_virtual_motion(rows)[:3, :3], actual.reshape(3, 3), atol=1e-8)
                self.assertEqual(bool(warnings), expect_warning)
                if expect_warning:
                    self.assertEqual(rows[0]["value"], 0.0)

    def test_free_joint_reconstructs_world_pose(self):
        loaded = load_fixture("complex.xml")
        qpos = loaded.model.qpos0.copy()
        qpos[:3] = [0.2, -0.1, 0.7]
        qpos[3:7] = normalized([0.9, 0.1, 0.2, 0.3])
        loaded.data.qpos[:] = qpos
        mujoco.mj_forward(loaded.model, loaded.data)
        self.assert_tip_matches_mujoco(loaded, loaded.chains[0])
        rows, _ = virtualize_joint(loaded.model, 0, qpos)
        np.testing.assert_allclose(compose_virtual_motion(rows)[:3, 3], qpos[:3], atol=1e-9)

        rotated_base = b'''<mujoco><worldbody><body name="floating" pos="0.3 -0.2 0.4" quat="0.9238795325 0 0 0.3826834324">
          <freejoint name="free"/><geom type="sphere" size="0.1" mass="1"/>
          </body></worldbody></mujoco>'''
        loaded = compile_mjcf(rotated_base, "rotated_free.xml")
        self.assert_tip_matches_mujoco(loaded, loaded.chains[0])
        loaded.data.qpos[:3] = [-0.4, 0.5, 0.7]
        loaded.data.qpos[3:7] = normalized([0.6, 0.2, -0.3, 0.4])
        mujoco.mj_forward(loaded.model, loaded.data)
        self.assert_tip_matches_mujoco(loaded, loaded.chains[0])

    def test_mixed_fixtures_reconstruct_all_branch_tips_at_nonzero_pose(self):
        for name in ("basic.xml", "intermediate.xml", "complex.xml"):
            with self.subTest(name=name):
                loaded = load_fixture(name)
                qpos = loaded.model.qpos0.copy()
                if name == "basic.xml":
                    qpos[:] = [0.3, 0.12]
                elif name == "intermediate.xml":
                    qpos[:] = [0.4, 0.12, 0.2, -0.3]
                else:
                    qpos[:3] = [0.2, -0.1, 0.7]
                    qpos[3:7] = normalized([0.9, 0.1, 0.2, 0.3])
                    qpos[7:11] = normalized([0.8, -0.2, 0.3, 0.1])
                    qpos[11:] = [0.2, -0.3, 0.1, 0.4]
                loaded.data.qpos[:] = qpos
                mujoco.mj_forward(loaded.model, loaded.data)
                for chain in loaded.chains:
                    self.assert_tip_matches_mujoco(loaded, chain)

    def test_nonzero_pivot_body_orientation_and_scalar_reference(self):
        xml = b'''<mujoco><worldbody>
          <body name="mixed" pos="1 2 3" quat="0.9238795325 0 0.3826834324 0">
            <joint name="slider" type="slide" axis="1 0 0" ref="0.2"/>
            <joint name="orient" type="ball" pos="0.1 0.2 0.3"/>
            <geom type="sphere" size="0.1" mass="1"/>
          </body></worldbody></mujoco>'''
        loaded = compile_mjcf(xml, "mixed.xml")
        self.assert_tip_matches_mujoco(loaded, loaded.chains[0])
        loaded.data.qpos[0] = 0.35
        loaded.data.qpos[1:5] = normalized([0.8, 0.2, 0.3, -0.1])
        mujoco.mj_forward(loaded.model, loaded.data)
        rows, _ = virtualize_chain(loaded.model, loaded.chains[0], loaded.data.qpos)
        self.assertAlmostEqual(rows[0]["value"], 0.15)
        self.assert_tip_matches_mujoco(loaded, loaded.chains[0])

    def test_ball_before_slide_on_same_body_preserves_joint_order(self):
        xml = b'''<mujoco><worldbody><body name="mixed" pos="1 2 3" quat="0.9238795325 0 0.3826834324 0">
          <joint name="orient" type="ball" pos="0.1 0.2 0.3"/>
          <joint name="slider" type="slide" axis="1 0 0" ref="0.2"/>
          <geom type="sphere" size="0.1" mass="1"/>
          </body></worldbody></mujoco>'''
        loaded = compile_mjcf(xml, "reversed_mixed.xml")
        loaded.data.qpos[:4] = normalized([0.8, 0.2, 0.3, -0.1])
        loaded.data.qpos[4] = 0.35
        mujoco.mj_forward(loaded.model, loaded.data)
        rows, _ = virtualize_chain(loaded.model, loaded.chains[0], loaded.data.qpos)
        self.assertEqual([row["name"] for row in rows], ["orient.yaw", "orient.pitch", "orient.roll", "slider"])
        self.assert_tip_matches_mujoco(loaded, loaded.chains[0])

    def test_invalid_pose_and_quaternion_are_rejected(self):
        loaded = load_fixture("complex.xml")
        with self.assertRaises(ValueError):
            virtualize_chain(loaded.model, loaded.chains[0], np.zeros(loaded.model.nq - 1))
        bad = loaded.model.qpos0.copy()
        bad[7:11] = 0
        with self.assertRaises(ValueError):
            virtualize_chain(loaded.model, loaded.chains[0], bad)


if __name__ == "__main__":
    unittest.main()
