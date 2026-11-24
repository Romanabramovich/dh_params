"""Step 1 tests for standalone MJCF import and compiled topology."""

from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from backend.main import app


FIXTURES = Path(__file__).parent / "fixtures"


class ModelUploadTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def upload_fixture(self, name):
        path = FIXTURES / name
        return self.client.post("/api/model", files={"file": (name, path.read_bytes(), "application/xml")})

    def test_three_fixture_levels_compile_and_keep_expected_topology(self):
        expected = {
            "basic.xml": ("basic_serial", 3, 2, 2, 0, {"free": 0, "ball": 0, "slide": 1, "hinge": 1}),
            "intermediate.xml": ("intermediate_branching", 7, 4, 4, 0, {"free": 0, "ball": 0, "slide": 1, "hinge": 3}),
            "complex.xml": ("complex_floating_loop", 7, 6, 15, 1, {"free": 1, "ball": 1, "slide": 0, "hinge": 4}),
        }
        for filename, (model_name, nbody, njnt, nq, neq, counts) in expected.items():
            with self.subTest(filename=filename):
                response = self.upload_fixture(filename)
                self.assertEqual(response.status_code, 200, response.text)
                summary = response.json()
                self.assertEqual(summary["model_name"], model_name)
                self.assertEqual((summary["body_count"], summary["joint_count"], summary["qpos_size"], summary["equality_count"]), (nbody, njnt, nq, neq))
                self.assertEqual(summary["joint_counts"], counts)
                self.assertEqual(summary["reference_qpos"], app.state.loaded_model.model.qpos0.tolist())
                self.assertGreater(summary["camera"]["distance"], 0)

    def test_intermediate_keeps_fixed_bodies_branching_and_same_body_joints(self):
        summary = self.upload_fixture("intermediate.xml").json()
        bodies = {body["name"]: body["id"] for body in summary["bodies"]}
        parents = {body["name"]: body["parent_id"] for body in summary["bodies"]}
        self.assertEqual(parents["fixed_spacer"], bodies["pedestal"])
        self.assertEqual(parents["carrier"], bodies["fixed_spacer"])
        self.assertEqual(parents["left_arm"], bodies["carrier"])
        self.assertEqual(parents["right_arm"], bodies["carrier"])
        joints = {joint["name"]: joint for joint in summary["joints"]}
        self.assertEqual(joints["carrier_yaw"]["body_id"], joints["carrier_slide"]["body_id"])
        self.assertLess(joints["carrier_yaw"]["qpos_address"], joints["carrier_slide"]["qpos_address"])

    def test_complex_keeps_quaternion_coordinates_and_loop_constraint(self):
        response = self.upload_fixture("complex.xml")
        summary = response.json()
        self.assertEqual(summary["equality_count"], 1)
        self.assertEqual(summary["reference_qpos"][3:7], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(summary["reference_qpos"][7:11], [1.0, 0.0, 0.0, 0.0])
        joints = {joint["name"]: joint for joint in summary["joints"]}
        self.assertEqual(joints["base_free"]["qpos_address"], 0)
        self.assertEqual(joints["ball"]["qpos_address"], 7)
        self.assertEqual(joints["axis_a"]["qpos_address"], 11)
        self.assertEqual(joints["axis_b"]["qpos_address"], 13)
        data = app.state.loaded_model.data
        self.assertAlmostEqual(float(data.xaxis[joints["axis_a"]["id"]] @ data.xaxis[joints["axis_a2"]["id"]]), 1.0)
        self.assertAlmostEqual(float(data.xaxis[joints["axis_a"]["id"]] @ data.xaxis[joints["axis_b"]["id"]]), 1.0, places=7)

    def test_invalid_xml_and_missing_external_asset_have_distinct_errors(self):
        malformed = self.client.post("/api/model", files={"file": ("bad.xml", b"<mujoco>", "application/xml")})
        self.assertEqual((malformed.status_code, malformed.json()["error"]["code"]), (400, "invalid_xml"))

        missing = self.client.post("/api/model", files={"file": ("mesh.xml", b'<mujoco><asset><mesh name="m" file="missing.stl"/></asset></mujoco>', "application/xml")})
        self.assertEqual((missing.status_code, missing.json()["error"]["code"]), (422, "external_file"))
        self.assertIn("missing.stl", missing.json()["error"]["message"])

    def test_valid_xml_that_mujoco_rejects_reports_compile_error(self):
        response = self.client.post("/api/model", files={"file": ("bad.xml", b"<mujoco><worldbody><joint/></worldbody></mujoco>", "application/xml")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "compile_error")

    def test_successful_replacement_and_failed_upload_state(self):
        self.upload_fixture("complex.xml")
        self.upload_fixture("basic.xml")
        current = self.client.get("/api/model").json()
        self.assertEqual(current["model_name"], "basic_serial")
        self.assertEqual(current["joint_count"], 2)

        self.client.post("/api/model", files={"file": ("bad.xml", b"<mujoco>", "application/xml")})
        self.assertEqual(self.client.get("/api/model").json()["model_name"], "basic_serial")

    def test_empty_and_wrong_file_type(self):
        empty = self.client.post("/api/model", files={"file": ("empty.xml", b"", "application/xml")})
        self.assertEqual(empty.json()["error"]["code"], "empty_file")
        wrong = self.client.post("/api/model", files={"file": ("robot.zip", b"contents", "application/zip")})
        self.assertEqual((wrong.status_code, wrong.json()["error"]["code"]), (415, "file_type"))

    def test_mjcf_extension_and_size_limit(self):
        path = FIXTURES / "basic.xml"
        accepted = self.client.post("/api/model", files={"file": ("basic.mjcf", path.read_bytes(), "application/xml")})
        self.assertEqual(accepted.status_code, 200)
        oversized = self.client.post("/api/model", files={"file": ("large.xml", b" " * (16 * 1024 * 1024 + 1), "application/xml")})
        self.assertEqual((oversized.status_code, oversized.json()["error"]["code"]), (413, "file_too_large"))

    def test_page_and_assets_are_served(self):
        self.assertIn("MJCF DH visualizer", self.client.get("/").text)
        script = self.client.get("/app.js")
        self.assertEqual(script.status_code, 200)
        self.assertEqual(script.headers["cache-control"], "no-store")
        self.assertEqual(self.client.get("/style.css").status_code, 200)


if __name__ == "__main__":
    unittest.main()
