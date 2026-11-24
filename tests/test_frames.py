"""Step 5 tests for pair geometry and standard-DH frame attachments."""

from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from backend.frames import assign_chain_frames, evaluate_frame_transforms
from backend.model_loader import compile_mjcf
from backend.utils import assign_dh_pair
from backend.virtual_joints import trace_virtual_chain


FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name):
    path = FIXTURES / name
    return compile_mjcf(path.read_bytes(), path.name)


class PairGeometryTests(unittest.TestCase):
    def assert_right_handed(self, result):
        rotation = result["transform"][:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=10)

    def test_skew_intersecting_parallel_and_collinear(self):
        cases = (
            ("skew", [0, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0], 1.0),
            ("intersecting", [0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0], 0.0),
            ("parallel", [0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], 1.0),
            ("collinear", [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 1], 0.0),
        )
        for expected, p1, z1, p2, z2, distance in cases:
            with self.subTest(case=expected):
                result = assign_dh_pair(p1, z1, p2, z2)
                self.assertEqual(result["case"], expected)
                self.assertAlmostEqual(result["distance"], distance, places=8)
                self.assert_right_handed(result)
                self.assertLess(np.linalg.norm(np.cross(result["origin"] - p2, z2)), 1e-8)

    def test_parallel_projection_ignores_off_axis_previous_origin(self):
        result = assign_dh_pair(
            [0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1],
            previous_origin=[0.3, 0.4, 2.0],
        )
        self.assertEqual(result["case"], "parallel")
        np.testing.assert_allclose(result["foot_on_first"], [0, 0, 2], atol=1e-10)
        np.testing.assert_allclose(result["origin"], [1, 0, 2], atol=1e-10)
        np.testing.assert_allclose(result["x"], [1, 0, 0], atol=1e-10)

    def test_reversed_collinear_and_near_parallel_axes(self):
        reversed_axes = assign_dh_pair([0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, -1])
        self.assertEqual(reversed_axes["case"], "parallel")
        np.testing.assert_allclose(reversed_axes["x"], [1, 0, 0], atol=1e-10)
        self.assert_right_handed(reversed_axes)

        collinear = assign_dh_pair([0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, -1], previous_x=[0, 1, 0])
        self.assertEqual(collinear["case"], "collinear")
        np.testing.assert_allclose(collinear["x"], [0, 1, 0], atol=1e-10)

        near = assign_dh_pair([0, 0, 0], [0, 0, 1], [1, 0, 0], [1e-8, 0, 1], angular_tol=1e-7)
        self.assertEqual(near["case"], "parallel")
        self.assert_right_handed(near)
        self.assertLess(abs(float(np.dot(near["x"], near["z"]))), 1e-10)


class ChainFrameTests(unittest.TestCase):
    def test_fixture_reference_frames_are_repeatable_and_follow_joint_axes(self):
        for name in ("basic.xml", "intermediate.xml", "complex.xml"):
            with self.subTest(name=name):
                loaded = load_fixture(name)
                for chain in loaded.chains:
                    assignment = loaded.frame_assignments[chain["id"]]
                    self.assertEqual(len(assignment.reference_frames), len(chain["virtual_joints"]) + 1)
                    self.assertEqual(len(assignment.pairs), len(chain["virtual_joints"]) - 1)
                    for index, axis in enumerate(assignment.axes):
                        frame = assignment.reference_frames[index]
                        direction = np.asarray(axis["direction"])
                        np.testing.assert_allclose(frame[:3, 2], direction, atol=1e-9)
                        self.assertLess(np.linalg.norm(np.cross(frame[:3, 3] - axis["point"], direction)), 1e-8)
                        np.testing.assert_allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-9)
                        self.assertAlmostEqual(float(np.linalg.det(frame[:3, :3])), 1.0, places=9)
                        if chain["virtual_joints"][index]["source_joint_type"] in ("hinge", "slide"):
                            joint_id = axis["source_joint_id"]
                            np.testing.assert_allclose(axis["point"], loaded.data.xanchor[joint_id], atol=1e-9)
                            np.testing.assert_allclose(axis["direction"], loaded.data.xaxis[joint_id], atol=1e-9)
                    self.assertEqual(assignment.as_dict(), assign_chain_frames(loaded.model, loaded.data, chain).as_dict())

    def test_cached_frames_move_with_virtual_prefixes_without_reclassification(self):
        loaded = load_fixture("complex.xml")
        qpos = loaded.model.qpos0.copy()
        qpos[:3] = [0.2, -0.1, 0.7]
        qpos[3:7] = np.array([0.9, 0.1, 0.2, 0.3]) / np.linalg.norm([0.9, 0.1, 0.2, 0.3])
        qpos[7:11] = np.array([0.8, -0.2, 0.3, 0.1]) / np.linalg.norm([0.8, -0.2, 0.3, 0.1])
        qpos[11:] = [0.2, -0.3, 0.1, 0.4]
        loaded.data.qpos[:] = qpos
        mujoco.mj_forward(loaded.model, loaded.data)

        for chain in loaded.chains:
            assignment = loaded.frame_assignments[chain["id"]]
            with patch("backend.frames.assign_dh_pair", side_effect=AssertionError("pair geometry reran")):
                frames = evaluate_frame_transforms(loaded.model, chain, assignment, qpos)
            trace = trace_virtual_chain(loaded.model, chain, qpos)
            for index, frame in enumerate(frames[:-1]):
                direction = trace.axis_directions[index]
                np.testing.assert_allclose(frame[:3, 2], direction, atol=1e-9)
                self.assertLess(np.linalg.norm(np.cross(frame[:3, 3] - trace.axis_points[index], direction)), 1e-8)
            predicted_tip = frames[-1] @ assignment.tool_transform
            tip_id = chain["tip_body_id"]
            np.testing.assert_allclose(predicted_tip[:3, 3], loaded.data.xpos[tip_id], atol=1e-9)
            np.testing.assert_allclose(predicted_tip[:3, :3], loaded.data.xmat[tip_id].reshape(3, 3), atol=1e-9)

    def test_single_joint_chain_has_base_and_terminal_tool_frame(self):
        xml = b'''<mujoco><worldbody><body name="one" pos="0 0 0.2">
          <joint name="j" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.1" mass="1"/>
          </body></worldbody></mujoco>'''
        loaded = compile_mjcf(xml, "one.xml")
        chain = loaded.chains[0]
        assignment = loaded.frame_assignments[chain["id"]]
        self.assertEqual(len(assignment.reference_frames), 2)
        self.assertEqual(assignment.pairs, [])
        np.testing.assert_allclose(assignment.reference_frames[0], assignment.reference_frames[1], atol=1e-12)
        self.assertGreater(assignment.distance_tolerance, 0)

    def test_slide_anchor_offset_tracks_mujoco_axis_after_motion(self):
        xml = b'''<mujoco><worldbody><body name="base">
          <joint name="hinge" type="hinge" axis="0 0 1" pos="0.1 0 0"/>
          <geom type="sphere" size="0.1" mass="1"/>
          <body name="slider" pos="0.2 0 0">
            <joint name="slide" type="slide" axis="0 1 0" pos="0 0.05 0"/>
            <geom type="sphere" size="0.1" mass="1"/>
          </body></body></worldbody></mujoco>'''
        loaded = compile_mjcf(xml, "offset_slide.xml")
        chain = loaded.chains[0]
        assignment = loaded.frame_assignments[chain["id"]]
        loaded.data.qpos[:] = [0.3, 0.12]
        mujoco.mj_forward(loaded.model, loaded.data)
        frames = evaluate_frame_transforms(loaded.model, chain, assignment, loaded.data.qpos)
        direction = loaded.data.xaxis[1]
        np.testing.assert_allclose(frames[1][:3, 2], direction, atol=1e-9)
        self.assertLess(np.linalg.norm(np.cross(frames[1][:3, 3] - loaded.data.xanchor[1], direction)), 1e-8)
        predicted_tip = frames[-1] @ assignment.tool_transform
        np.testing.assert_allclose(predicted_tip[:3, 3], loaded.data.xpos[2], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
