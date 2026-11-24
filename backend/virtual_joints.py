"""Map MuJoCo joints to ordered one-degree-of-freedom virtual joints."""

import math
from dataclasses import dataclass

import mujoco
import numpy as np


X = (1.0, 0.0, 0.0)
Y = (0.0, 1.0, 0.0)
Z = (0.0, 0.0, 1.0)
SINGULAR_TOL = 1e-8
SINGULAR_WARNING_TOL = 1e-4


def quaternion_to_zyx(quaternion) -> tuple[float, float, float, bool]:
    """Return yaw, pitch, roll and a near-singularity flag.

    At the Z-Y-X singularity, yaw is fixed to zero and roll absorbs the
    remaining rotation. The returned angles still reconstruct the pose.
    """
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("Quaternion must contain four finite values.")
    magnitude = float(np.linalg.norm(quat))
    if magnitude < 1e-12:
        raise ValueError("Quaternion must have nonzero length.")
    quat = quat / magnitude

    flat_rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(flat_rotation, quat)
    rotation = flat_rotation.reshape(3, 3)

    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    cosine = abs(math.cos(pitch))
    if cosine <= SINGULAR_TOL:
        yaw = 0.0
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
    else:
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
    return yaw, pitch, roll, cosine < SINGULAR_WARNING_TOL


def _row(name, source_name, source_id, source_type, kind, component, axis, qpos_range, value):
    return {
        "name": name,
        "source_joint_name": source_name,
        "source_joint_id": source_id,
        "source_joint_type": source_type,
        "kind": kind,
        "component": component,
        "axis": list(axis),
        "qpos_range": list(qpos_range),
        "value": float(value),
        "unit": "rad" if kind == "revolute" else "m",
    }


def virtualize_joint(model: mujoco.MjModel, joint_id: int, qpos) -> tuple[list[dict], list[str]]:
    joint_type = int(model.jnt_type[joint_id])
    start = int(model.jnt_qposadr[joint_id])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"

    if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
        return [_row(name, name, joint_id, "hinge", "revolute", "angle", model.jnt_axis[joint_id], (start, start + 1), qpos[start] - model.qpos0[start])], []
    if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
        return [_row(name, name, joint_id, "slide", "prismatic", "distance", model.jnt_axis[joint_id], (start, start + 1), qpos[start] - model.qpos0[start])], []

    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        rows = [
            _row(f"{name}.t{axis_name}", name, joint_id, "free", "prismatic", f"t{axis_name}", axis, (start + index, start + index + 1), qpos[start + index])
            for index, (axis_name, axis) in enumerate((("x", X), ("y", Y), ("z", Z)))
        ]
        quaternion_start = start + 3
        source_type = "free"
    elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        rows = []
        quaternion_start = start
        source_type = "ball"
    else:
        raise ValueError(f"Unsupported MuJoCo joint type: {joint_type}.")

    yaw, pitch, roll, near_singular = quaternion_to_zyx(qpos[quaternion_start:quaternion_start + 4])
    for component, axis, value in (("yaw", Z, yaw), ("pitch", Y, pitch), ("roll", X, roll)):
        rows.append(_row(f"{name}.{component}", name, joint_id, source_type, "revolute", component, axis, (quaternion_start, quaternion_start + 4), value))
    warnings = [f"{name}: Z-Y-X orientation is near a singularity; virtual angles may change abruptly."] if near_singular else []
    return rows, warnings


def virtualize_chain(model: mujoco.MjModel, chain: dict, qpos) -> tuple[list[dict], list[str]]:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape != (model.nq,) or not np.all(np.isfinite(qpos)):
        raise ValueError(f"qpos must contain {model.nq} finite values.")
    rows = []
    warnings = []
    for joint_id in chain["joint_ids"]:
        joint_rows, joint_warnings = virtualize_joint(model, joint_id, qpos)
        rows.extend(joint_rows)
        warnings.extend(joint_warnings)
    return rows, warnings


def _translation(vector) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = vector
    return transform


def _quaternion_rotation(quaternion) -> np.ndarray:
    flat_rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(flat_rotation, np.asarray(quaternion, dtype=np.float64))
    return flat_rotation.reshape(3, 3)


def compose_virtual_motion(rows: list[dict]) -> np.ndarray:
    """Compose one source joint's virtual rows in their listed order."""
    result = np.eye(4)
    for row in rows:
        result = result @ virtual_row_transform(row)
    return result


def virtual_row_transform(row: dict) -> np.ndarray:
    axis = np.asarray(row["axis"], dtype=np.float64)
    value = row["value"]
    if row["kind"] == "prismatic":
        return _translation(axis * value)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + math.sin(value) * skew + (1 - math.cos(value)) * (skew @ skew)
    motion = np.eye(4)
    motion[:3, :3] = rotation
    return motion


@dataclass
class VirtualTrace:
    rows: list[dict]
    axis_points: list[np.ndarray]
    axis_directions: list[np.ndarray]
    prefix_transforms: list[np.ndarray]
    tip_transform: np.ndarray


def trace_virtual_chain(model: mujoco.MjModel, chain: dict, qpos) -> VirtualTrace:
    """Track virtual axes and their moving prefixes through the body tree."""
    rows, _ = virtualize_chain(model, chain, qpos)
    rows_by_joint = {}
    for row in rows:
        rows_by_joint.setdefault(row["source_joint_id"], []).append(row)

    transform = np.eye(4)
    axis_points = []
    axis_directions = []
    prefixes = []
    for body_id in chain["body_ids"][1:]:
        start = int(model.body_jntadr[body_id])
        count = int(model.body_jntnum[body_id])
        body_joint_ids = range(start, start + count)
        if count and int(model.jnt_type[start]) == int(mujoco.mjtJoint.mjJNT_FREE):
            transform = np.eye(4)
        else:
            body_offset = _translation(model.body_pos[body_id])
            body_offset[:3, :3] = _quaternion_rotation(model.body_quat[body_id])
            transform = transform @ body_offset
        for joint_id in body_joint_ids:
            joint_type = int(model.jnt_type[joint_id])
            if joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_BALL)):
                anchor = model.jnt_pos[joint_id]
                transform = transform @ _translation(anchor)
            for row in rows_by_joint[joint_id]:
                if not prefixes:
                    prefixes.append(transform.copy())
                axis_points.append(transform[:3, 3].copy())
                axis_directions.append(transform[:3, :3] @ np.asarray(row["axis"], dtype=np.float64))
                transform = transform @ virtual_row_transform(row)
                prefixes.append(transform.copy())
            if joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_BALL)):
                transform = transform @ _translation(-anchor)
    return VirtualTrace(rows, axis_points, axis_directions, prefixes, transform)


def reconstruct_chain_tip(model: mujoco.MjModel, chain: dict, qpos) -> np.ndarray:
    """Rebuild a chain tip pose from virtual joints and compiled body offsets."""
    return trace_virtual_chain(model, chain, qpos).tip_transform
