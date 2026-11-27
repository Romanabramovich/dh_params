"""Step 7 tests for pose updates, moving overlays, and loop diagnostics."""

from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import mujoco
import numpy as np

from backend.main import app
from backend.model_loader import compile_mjcf
from backend.pose import PoseError, constraint_diagnostics, normalize_qpos
from backend.rendering import append_pose_overlays


FIXTURES = Path(__file__).parent / "fixtures"


class PoseLogicTests(unittest.TestCase):
    def test_invalid_pose_and_quaternion_normalization(self):
        path = FIXTURES / "complex.xml"
        loaded = compile_mjcf(path.read_bytes(), path.name)
        qpos = loaded.model.qpos0.tolist()
        for bad in (qpos[:-1], qpos + [0], [float("nan")] + qpos[1:], [True] + qpos[1:]):
            with self.subTest(bad=bad):
                with self.assertRaises(PoseError):
                    normalize_qpos(loaded.model, bad)
        zero_quaternion = qpos.copy()
        zero_quaternion[7:11] = [0, 0, 0, 0]
        with self.assertRaisesRegex(PoseError, "zero-length quaternion"):
            normalize_qpos(loaded.model, zero_quaternion)
        qpos[3:7] = [2, 0, 0, 0]
        qpos[7:11] = [0, 0, 0, 3]
        normalized = normalize_qpos(loaded.model, qpos)
        np.testing.assert_allclose(normalized[3:7], [1, 0, 0, 0])
        np.testing.assert_allclose(normalized[7:11], [0, 0, 0, 1])

    def test_joint_limits_are_enforced(self):
        xml = b'''<mujoco><compiler angle="radian"/><worldbody><body><joint type="hinge" range="-1 1" limited="true"/>
          <geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'''
        loaded = compile_mjcf(xml, "limited.xml")
        with self.assertRaisesRegex(PoseError, "stay in"):
            normalize_qpos(loaded.model, [1.5])
        np.testing.assert_allclose(normalize_qpos(loaded.model, [0.5]), [0.5])

    def test_constraint_gap_and_solver_residual_move_together(self):
        path = FIXTURES / "complex.xml"
        loaded = compile_mjcf(path.read_bytes(), path.name)
        initial = constraint_diagnostics(loaded.model, loaded.data)[0]
        self.assertEqual(initial["type"], "connect")
        self.assertAlmostEqual(initial["anchor_separation_m"], 0.0)
        loaded.data.qpos[12] = 0.5
        mujoco.mj_forward(loaded.model, loaded.data)
        moved = constraint_diagnostics(loaded.model, loaded.data)[0]
        self.assertGreater(moved["anchor_separation_m"], 0.01)
        self.assertAlmostEqual(moved["anchor_separation_m"], moved["residual_norm"], places=8)

    def test_overlay_geometry_tracks_frames_and_constraint_gap(self):
        path = FIXTURES / "complex.xml"
        loaded = compile_mjcf(path.read_bytes(), path.name)
        chain = loaded.chains[0]
        scene = mujoco.MjvScene(loaded.model, maxgeom=100)
        append_pose_overlays(scene, loaded, chain)
        self.assertEqual(scene.ngeom, 3 * 12 + 2)
        moving_frame_origin = scene.geoms[3].pos.copy()
        loaded.data.qpos[0] += 0.2
        loaded.data.qpos[12] = 0.5
        mujoco.mj_forward(loaded.model, loaded.data)
        loaded.pose_frames.clear()
        loaded.constraints = constraint_diagnostics(loaded.model, loaded.data)
        moved_scene = mujoco.MjvScene(loaded.model, maxgeom=100)
        append_pose_overlays(moved_scene, loaded, chain)
        self.assertEqual(moved_scene.ngeom, 3 * 12 + 3)
        self.assertGreater(np.linalg.norm(moved_scene.geoms[3].pos - moving_frame_origin), 0.01)
        with patch("backend.pose.evaluate_frame_transforms", side_effect=AssertionError("frames recomputed")):
            cached_scene = mujoco.MjvScene(loaded.model, maxgeom=100)
            append_pose_overlays(cached_scene, loaded, chain)
        self.assertEqual(cached_scene.ngeom, moved_scene.ngeom)


class PoseApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def upload(self, name):
        path = FIXTURES / name
        response = self.client.post("/api/model", files={"file": (name, path.read_bytes(), "application/xml")})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_pose_version_frame_header_and_cached_camera_render(self):
        summary = self.upload("basic.xml")
        chain_id = summary["chains"][0]["id"]
        initial = self.client.get("/api/frame", params={"chain_id": chain_id})
        self.assertEqual(initial.headers["x-pose-version"], "0")
        with patch("backend.rendering.mujoco.mj_forward", wraps=mujoco.mj_forward) as forward:
            result = self.client.put("/api/pose", json={"qpos": [0.4, 0.1], "chain_id": chain_id})
            self.assertEqual(forward.call_count, 1)
        self.assertEqual(result.status_code, 200, result.text)
        snapshot = result.json()
        self.assertEqual(snapshot["pose_version"], 1)
        self.assertEqual(snapshot["qpos"], [0.4, 0.1])
        self.assertEqual(len(snapshot["frames"]), 3)
        self.assertAlmostEqual(snapshot["dh_values"][0]["theta"] - summary["chains"][0]["dh_table"]["rows"][0]["theta_offset"], 0.4)
        self.assertAlmostEqual(snapshot["dh_values"][1]["d"] - summary["chains"][0]["dh_table"]["rows"][1]["d_offset"], 0.1)
        self.assertLess(snapshot["fk_position_error_m"], 1e-7)
        self.assertLess(snapshot["fk_orientation_error_rad"], 1e-5)
        with patch("backend.pose.evaluate_frame_transforms", side_effect=AssertionError("frames recomputed on camera move")), patch("backend.rendering.mujoco.mj_forward", side_effect=AssertionError("FK reran on camera move")):
            frame = self.client.get("/api/frame", params={"chain_id": chain_id, "azimuth": 45})
        self.assertEqual(frame.status_code, 200, frame.text[:200] if frame.status_code != 200 else "")
        self.assertEqual(frame.headers["x-pose-version"], "1")
        self.assertNotEqual(initial.content, frame.content)
        self.assertEqual(self.client.get("/api/pose", params={"chain_id": chain_id}).json()["pose_version"], 1)

    def test_invalid_pose_does_not_change_current_state(self):
        self.upload("complex.xml")
        before = self.client.get("/api/model").json()
        for payload in ({"qpos": [0]}, {"qpos": [0] * 15}, {"qpos": before["qpos"], "chain_id": "missing"}):
            response = self.client.put("/api/pose", json=payload)
            self.assertEqual(response.status_code, 422)
        after = self.client.get("/api/model").json()
        self.assertEqual(after["pose_version"], 0)
        self.assertEqual(after["qpos"], before["qpos"])
        self.assertEqual(self.client.get("/api/pose", params={"chain_id": "missing"}).status_code, 422)
        self.assertEqual(self.client.get("/api/frame", params={"chain_id": "missing"}).status_code, 422)

    def test_complex_pose_normalizes_quaternions_and_reports_loop_gap(self):
        summary = self.upload("complex.xml")
        qpos = summary["qpos"]
        qpos[3] = 2.0
        qpos[7] = 3.0
        qpos[12] = 0.5
        chain_id = summary["chains"][0]["id"]
        response = self.client.put("/api/pose", json={"qpos": qpos, "chain_id": chain_id})
        self.assertEqual(response.status_code, 200, response.text)
        snapshot = response.json()
        self.assertEqual(snapshot["qpos"][3], 1.0)
        self.assertEqual(snapshot["qpos"][7], 1.0)
        self.assertGreater(snapshot["constraints"][0]["anchor_separation_m"], 0.01)
        self.assertGreater(snapshot["constraints"][0]["residual_norm"], 0.01)
        self.assertLess(snapshot["fk_position_error_m"], 1e-7)
        self.assertEqual(self.client.get("/api/frame", params={"chain_id": chain_id}).headers["x-pose-version"], "1")


if __name__ == "__main__":
    unittest.main()
