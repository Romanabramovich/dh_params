"""Step 2 tests for camera geometry and rendered model frames."""

from io import BytesIO
import math
from pathlib import Path
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.rendering import CameraState, adjusted_camera


FIXTURES = Path(__file__).parent / "fixtures"


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.default = CameraState((1.0, 2.0, 3.0), 90.0, -45.0, 2.0)

    def test_orbit_wrap_and_elevation_limit(self):
        result = adjusted_camera(self.default, 450.0, 120.0, None)
        self.assertEqual(result.lookat, self.default.lookat)
        self.assertEqual(result.azimuth, 90.0)
        self.assertEqual(result.elevation, 89.0)

    def test_zoom_limits_follow_model_scale(self):
        self.assertEqual(adjusted_camera(self.default, None, None, 0.001).distance, 0.1)
        self.assertEqual(adjusted_camera(self.default, None, None, 1000).distance, 40.0)

    def test_invalid_camera_values(self):
        for azimuth, elevation, distance in [(math.nan, None, None), (None, math.inf, None), (None, None, 0), (None, None, -1)]:
            with self.subTest(values=(azimuth, elevation, distance)):
                with self.assertRaises(ValueError):
                    adjusted_camera(self.default, azimuth, elevation, distance)


class FrameEndpointTests(unittest.TestCase):
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

    def test_frame_without_model_and_bad_camera(self):
        missing = self.client.get("/api/frame")
        self.assertEqual((missing.status_code, missing.json()["error"]["code"]), (404, "no_model"))
        self.upload("basic.xml")
        bad = self.client.get("/api/frame?distance=0")
        self.assertEqual((bad.status_code, bad.json()["error"]["code"]), (422, "invalid_camera"))

    def test_all_fixture_levels_render_nonempty_jpeg(self):
        for name in ("basic.xml", "intermediate.xml", "complex.xml"):
            with self.subTest(name=name):
                self.upload(name)
                response = self.client.get("/api/frame")
                self.assertEqual(response.status_code, 200, response.text[:200] if response.status_code != 200 else "")
                self.assertEqual(response.headers["content-type"], "image/jpeg")
                image = Image.open(BytesIO(response.content))
                self.assertEqual(image.size, (800, 600))
                self.assertTrue(any(low < high for low, high in image.getextrema()))

    def test_orbit_changes_frame_and_model_replacement_recreates_renderer(self):
        summary = self.upload("basic.xml")
        initial = self.client.get("/api/frame")
        moved = self.client.get(f"/api/frame?azimuth={summary['camera']['azimuth'] + 90}")
        self.assertEqual((initial.status_code, moved.status_code), (200, 200))
        self.assertNotEqual(initial.content, moved.content)

        self.upload("complex.xml")
        replacement = self.client.get("/api/frame")
        self.assertEqual(replacement.status_code, 200, replacement.text[:200] if replacement.status_code != 200 else "")
        self.assertNotEqual(initial.content, replacement.content)


if __name__ == "__main__":
    unittest.main()
