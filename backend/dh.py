"""Extract standard-DH rows and verify their forward kinematics."""

from dataclasses import dataclass, field
import math

import mujoco
import numpy as np

from backend.frames import FrameAssignment
from backend.virtual_joints import virtualize_chain


ORIENTATION_TOLERANCE = 1e-5
VALIDATION_SAMPLES = 12
VALIDATION_SEED = 2026


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    cosine_theta, sine_theta = math.cos(theta), math.sin(theta)
    cosine_alpha, sine_alpha = math.cos(alpha), math.sin(alpha)
    return np.array([
        [cosine_theta, -sine_theta * cosine_alpha, sine_theta * sine_alpha, a * cosine_theta],
        [sine_theta, cosine_theta * cosine_alpha, -cosine_theta * sine_alpha, a * sine_theta],
        [0.0, sine_alpha, cosine_alpha, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def orientation_error(actual: np.ndarray, expected: np.ndarray) -> float:
    cosine = (float(np.trace(actual.T @ expected)) - 1.0) / 2.0
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def factor_standard_dh(relative: np.ndarray) -> tuple[float, float, float, float]:
    rotation = relative[:3, :3]
    position = relative[:3, 3]
    theta = math.atan2(rotation[1, 0], rotation[0, 0])
    alpha = math.atan2(rotation[2, 1], rotation[2, 2])
    d = float(position[2])
    a = float(position[0] * math.cos(theta) + position[1] * math.sin(theta))
    return theta, d, a, alpha


@dataclass
class DhTable:
    chain_id: str
    rows: list[dict]
    base_transform: np.ndarray
    tool_transform: np.ndarray
    reference_position_error: float
    reference_orientation_error: float
    validation: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "convention": "standard",
            "chain_id": self.chain_id,
            "rows": self.rows,
            "base_transform": self.base_transform.tolist(),
            "tool_transform": self.tool_transform.tolist(),
            "validation": self.validation,
        }


def extract_dh_table(chain: dict, assignment: FrameAssignment) -> DhTable:
    virtual_rows = chain["virtual_joints"]
    frames = assignment.reference_frames
    if len(frames) != len(virtual_rows) + 1:
        raise ValueError("Each virtual joint requires one standard-DH row.")

    rows = []
    maximum_position_error = 0.0
    maximum_orientation_error = 0.0
    for index, virtual in enumerate(virtual_rows):
        relative = np.linalg.inv(frames[index]) @ frames[index + 1]
        theta, d, a, alpha = factor_standard_dh(relative)
        rebuilt = dh_transform(theta, d, a, alpha)
        maximum_position_error = max(maximum_position_error, float(np.linalg.norm(relative[:3, 3] - rebuilt[:3, 3])))
        maximum_orientation_error = max(maximum_orientation_error, orientation_error(relative[:3, :3], rebuilt[:3, :3]))
        rows.append({
            "index": index + 1,
            "name": virtual["name"],
            "source_joint_id": virtual["source_joint_id"],
            "kind": virtual["kind"],
            "variable": "theta" if virtual["kind"] == "revolute" else "d",
            "reference_value": virtual["value"],
            "qpos_range": virtual["qpos_range"],
            "theta_offset": theta,
            "d_offset": d,
            "a": a,
            "alpha": alpha,
        })

    return DhTable(
        chain_id=chain["id"],
        rows=rows,
        base_transform=assignment.base_transform.copy(),
        tool_transform=assignment.tool_transform.copy(),
        reference_position_error=maximum_position_error,
        reference_orientation_error=maximum_orientation_error,
    )


def dh_forward_kinematics(table: DhTable, current_virtual_rows: list[dict]) -> np.ndarray:
    if len(current_virtual_rows) != len(table.rows):
        raise ValueError("Virtual pose does not match the DH table.")
    transform = table.base_transform.copy()
    for row, current in zip(table.rows, current_virtual_rows, strict=True):
        if row["name"] != current["name"] or row["kind"] != current["kind"]:
            raise ValueError("Virtual pose does not match the DH table.")
        delta = current["value"] - row["reference_value"]
        theta = row["theta_offset"] + (delta if row["variable"] == "theta" else 0.0)
        d = row["d_offset"] + (delta if row["variable"] == "d" else 0.0)
        transform = transform @ dh_transform(theta, d, row["a"], row["alpha"])
    return transform @ table.tool_transform


def sample_qposes(model: mujoco.MjModel, count: int = VALIDATION_SAMPLES, seed: int = VALIDATION_SEED) -> list[np.ndarray]:
    if count < 0:
        raise ValueError("Sample count must be nonnegative.")
    generator = np.random.default_rng(seed)
    samples = [model.qpos0.copy()]
    scale = max(float(model.stat.extent), 0.1)
    for _ in range(count):
        qpos = model.qpos0.copy()
        for joint_id in range(model.njnt):
            kind = int(model.jnt_type[joint_id])
            start = int(model.jnt_qposadr[joint_id])
            if kind in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                if model.jnt_limited[joint_id] and model.jnt_range[joint_id, 1] > model.jnt_range[joint_id, 0]:
                    qpos[start] = generator.uniform(*model.jnt_range[joint_id])
                else:
                    amplitude = 0.5 if kind == int(mujoco.mjtJoint.mjJNT_HINGE) else 0.2 * scale
                    qpos[start] += generator.uniform(-amplitude, amplitude)
            else:
                if kind == int(mujoco.mjtJoint.mjJNT_FREE):
                    qpos[start:start + 3] += generator.uniform(-0.2 * scale, 0.2 * scale, size=3)
                    quaternion_start = start + 3
                else:
                    quaternion_start = start
                quaternion = generator.normal(size=4)
                qpos[quaternion_start:quaternion_start + 4] = quaternion / np.linalg.norm(quaternion)
        samples.append(qpos)
    return samples


def validate_tables(model: mujoco.MjModel, chains: list[dict], tables: dict[str, DhTable], poses: list[np.ndarray]) -> None:
    if not poses:
        raise ValueError("At least one validation pose is required.")
    position_tolerance = 1e-6 * max(1.0, float(model.stat.extent))
    maxima = {chain["id"]: [0.0, 0.0] for chain in chains}
    data = mujoco.MjData(model)
    for qpos in poses:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for chain in chains:
            table = tables[chain["id"]]
            current_rows, _ = virtualize_chain(model, chain, qpos)
            predicted = dh_forward_kinematics(table, current_rows)
            tip_id = chain["tip_body_id"]
            position_error = float(np.linalg.norm(predicted[:3, 3] - data.xpos[tip_id]))
            rotation_error = orientation_error(predicted[:3, :3], data.xmat[tip_id].reshape(3, 3))
            maxima[chain["id"]][0] = max(maxima[chain["id"]][0], position_error)
            maxima[chain["id"]][1] = max(maxima[chain["id"]][1], rotation_error)

    for chain in chains:
        table = tables[chain["id"]]
        position_error, rotation_error = maxima[chain["id"]]
        reference_valid = table.reference_position_error <= position_tolerance and table.reference_orientation_error <= ORIENTATION_TOLERANCE
        fk_valid = position_error <= position_tolerance and rotation_error <= ORIENTATION_TOLERANCE
        table.validation = {
            "status": "verified" if reference_valid and fk_valid else "invalid",
            "scope": "unconstrained_tree_fk",
            "sample_count": len(poses),
            "position_tolerance_m": position_tolerance,
            "orientation_tolerance_rad": ORIENTATION_TOLERANCE,
            "max_reference_position_error_m": table.reference_position_error,
            "max_reference_orientation_error_rad": table.reference_orientation_error,
            "max_fk_position_error_m": position_error,
            "max_fk_orientation_error_rad": rotation_error,
        }


def build_validated_tables(model: mujoco.MjModel, chains: list[dict], assignments: dict[str, FrameAssignment]) -> dict[str, DhTable]:
    tables = {chain["id"]: extract_dh_table(chain, assignments[chain["id"]]) for chain in chains}
    if tables:
        validate_tables(model, chains, tables, sample_qposes(model))
    return tables
