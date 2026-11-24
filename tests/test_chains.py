"""Step 3 tests for compiled root-to-tip chain discovery."""

from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from backend.main import app
from backend.model_loader import compile_mjcf, model_summary


FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name):
    path = FIXTURES / name
    return compile_mjcf(path.read_bytes(), path.name)


class ChainDiscoveryTests(unittest.TestCase):
    def test_basic_serial_chain(self):
        chains = load_fixture("basic.xml").chains
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0]["id"], "tip-2")
        self.assertEqual(chains[0]["body_ids"], [0, 1, 2])
        self.assertEqual(chains[0]["joint_ids"], [0, 1])
        self.assertEqual(chains[0]["joint_names"], ["shoulder", "extension"])

    def test_intermediate_branches_share_ordered_joint_prefix(self):
        chains = load_fixture("intermediate.xml").chains
        self.assertEqual([chain["name"] for chain in chains], ["left_tool", "right_arm"])
        self.assertEqual([chain["joint_ids"] for chain in chains], [[0, 1, 2], [0, 1, 3]])
        self.assertEqual(chains[0]["body_names"], ["world", "pedestal", "fixed_spacer", "carrier", "left_arm", "left_tool"])
        self.assertEqual(chains[1]["body_names"], ["world", "pedestal", "fixed_spacer", "carrier", "right_arm"])
        self.assertEqual(chains[0]["joint_names"][:2], ["carrier_yaw", "carrier_slide"])

    def test_complex_chains_keep_free_and_ball_joints_before_branch(self):
        chains = load_fixture("complex.xml").chains
        self.assertEqual([chain["name"] for chain in chains], ["tip_a", "tip_b"])
        self.assertEqual([chain["joint_ids"] for chain in chains], [[0, 1, 2, 3], [0, 1, 4, 5]])
        self.assertEqual([chain["joint_names"][:2] for chain in chains], [["base_free", "ball"], ["base_free", "ball"]])

    def test_jointless_model_has_no_chains(self):
        xml = b'<mujoco model="jointless"><worldbody><body name="fixed"><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        loaded = compile_mjcf(xml, "jointless.xml")
        self.assertEqual(loaded.chains, [])
        self.assertEqual(model_summary(loaded)["chains"], [])

    def test_fixed_only_branch_is_not_reported_as_movable_chain(self):
        xml = b'''<mujoco><compiler fusestatic="false"/><worldbody>
          <body name="fixed"><geom type="sphere" size="0.1"/></body>
          <body name="moving"><joint name="j"/><geom type="sphere" size="0.1" mass="1"/></body>
          </worldbody></mujoco>'''
        chains = compile_mjcf(xml, "mixed.xml").chains
        self.assertEqual([chain["name"] for chain in chains], ["moving"])
        self.assertEqual(chains[0]["joint_names"], ["j"])

    def test_ids_and_order_are_stable_for_same_model(self):
        first = load_fixture("intermediate.xml").chains
        second = load_fixture("intermediate.xml").chains
        self.assertEqual(first, second)


class ChainApiTests(unittest.TestCase):
    def test_post_and_get_expose_the_same_chains(self):
        path = FIXTURES / "intermediate.xml"
        with TestClient(app) as client:
            posted = client.post("/api/model", files={"file": (path.name, path.read_bytes(), "application/xml")})
            self.assertEqual(posted.status_code, 200)
            current = client.get("/api/model")
            self.assertEqual(current.status_code, 200)
            self.assertEqual(posted.json()["chains"], current.json()["chains"])
            self.assertEqual(len(current.json()["chains"]), 2)


if __name__ == "__main__":
    unittest.main()
