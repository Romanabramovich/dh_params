"""Validate pose input and report moving DH frames and equality constraints."""

import math

import mujoco
import numpy as np

from backend.dh import dh_forward_kinematics, orientation_error
from backend.frames import evaluate_frame_transforms
from backend.virtual_joints import virtualize_chain


class PoseError(ValueError):
    pass


def normalize_qpos(model: mujoco.MjModel, values) -> np.ndarray:
    if not isinstance(values, list) or len(values) != model.nq:
        raise PoseError(f"qpos must be a list of {model.nq} numbers.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise PoseError("qpos values must be finite numbers.")
    qpos = np.asarray(values, dtype=np.float64)
    for joint_id in range(model.njnt):
        kind = int(model.jnt_type[joint_id])
        start = int(model.jnt_qposadr[joint_id])
        if kind in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            if model.jnt_limited[joint_id]:
                lower, upper = model.jnt_range[joint_id]
                if not lower - 1e-9 <= qpos[start] <= upper + 1e-9:
                    raise PoseError(f"Joint {joint_id} must stay in [{lower:g}, {upper:g}].")
            continue
        quaternion_start = start + 3 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else start
        quaternion = qpos[quaternion_start:quaternion_start + 4]
        magnitude = float(np.linalg.norm(quaternion))
        if magnitude < 1e-12:
            raise PoseError(f"Joint {joint_id} has a zero-length quaternion.")
        qpos[quaternion_start:quaternion_start + 4] = quaternion / magnitude
    return qpos


def _spatial_anchors(model: mujoco.MjModel, data: mujoco.MjData, index: int):
    kind = int(model.eq_type[index])
    if kind not in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
        return None
    first, second = int(model.eq_obj1id[index]), int(model.eq_obj2id[index])
    if int(model.eq_objtype[index]) == int(mujoco.mjtObj.mjOBJ_SITE):
        return data.site_xpos[first].copy(), data.site_xpos[second].copy()
    values = model.eq_data[index]
    if kind == int(mujoco.mjtEq.mjEQ_CONNECT):
        first_local, second_local = values[:3], values[3:6]
    else:
        first_local, second_local = values[3:6], values[:3]
    first_world = data.xpos[first] + data.xmat[first].reshape(3, 3) @ first_local
    second_world = data.xpos[second] + data.xmat[second].reshape(3, 3) @ second_local
    return first_world, second_world


def constraint_diagnostics(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
    equality_rows = {}
    for row in range(data.nefc):
        if int(data.efc_type[row]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY):
            equality_rows.setdefault(int(data.efc_id[row]), []).append(float(data.efc_pos[row]))
    diagnostics = []
    names = {
        int(mujoco.mjtEq.mjEQ_CONNECT): "connect",
        int(mujoco.mjtEq.mjEQ_WELD): "weld",
        int(mujoco.mjtEq.mjEQ_JOINT): "joint",
        int(mujoco.mjtEq.mjEQ_TENDON): "tendon",
        int(mujoco.mjtEq.mjEQ_DISTANCE): "distance",
    }
    for index in range(model.neq):
        kind = int(model.eq_type[index])
        residuals = equality_rows.get(index, [])
        anchors = _spatial_anchors(model, data, index)
        diagnostics.append({
            "id": index,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index) or f"equality_{index}",
            "type": names.get(kind, "other"),
            "active": bool(data.eq_active[index]),
            "residual": residuals,
            "residual_norm": float(np.linalg.norm(residuals)) if residuals else None,
            "residual_unit": "m" if kind == int(mujoco.mjtEq.mjEQ_CONNECT) else "solver units",
            "anchors": [point.tolist() for point in anchors] if anchors else None,
            "anchor_separation_m": float(np.linalg.norm(anchors[0] - anchors[1])) if anchors else None,
        })
    return diagnostics


def selected_chain(model, chains: list[dict], chain_id: str | None) -> dict | None:
    if not chains:
        if chain_id:
            raise PoseError("This model has no kinematic chains.")
        return None
    if chain_id is None:
        return chains[0]
    for chain in chains:
        if chain["id"] == chain_id:
            return chain
    raise PoseError(f"Unknown chain: {chain_id}.")


def chain_frames(loaded, chain: dict) -> list[np.ndarray]:
    chain_id = chain["id"]
    if chain_id not in loaded.pose_frames:
        loaded.pose_frames[chain_id] = evaluate_frame_transforms(
            loaded.model, chain, loaded.frame_assignments[chain_id], loaded.data.qpos,
        )
    return loaded.pose_frames[chain_id]


def pose_snapshot(loaded, chain: dict | None) -> dict:
    result = {
        "pose_version": loaded.pose_version,
        "qpos": loaded.data.qpos.tolist(),
        "constraints": loaded.constraints,
        "chain_id": chain["id"] if chain else None,
        "frames": [],
        "virtual_joints": [],
        "dh_values": [],
    }
    if chain is None:
        return result
    frames = chain_frames(loaded, chain)
    virtual_rows, warnings = virtualize_chain(loaded.model, chain, loaded.data.qpos)
    table = loaded.dh_tables[chain["id"]]
    values = []
    for row, virtual in zip(table.rows, virtual_rows, strict=True):
        delta = virtual["value"] - row["reference_value"]
        values.append({
            "name": row["name"],
            "theta": row["theta_offset"] + (delta if row["variable"] == "theta" else 0.0),
            "d": row["d_offset"] + (delta if row["variable"] == "d" else 0.0),
        })
    predicted = dh_forward_kinematics(table, virtual_rows)
    tip_id = chain["tip_body_id"]
    result.update({
        "frames": [{
            "index": index,
            "origin": frame[:3, 3].tolist(),
            "x": frame[:3, 0].tolist(),
            "y": frame[:3, 1].tolist(),
            "z": frame[:3, 2].tolist(),
        } for index, frame in enumerate(frames)],
        "virtual_joints": virtual_rows,
        "virtual_warnings": warnings,
        "dh_values": values,
        "fk_position_error_m": float(np.linalg.norm(predicted[:3, 3] - loaded.data.xpos[tip_id])),
        "fk_orientation_error_rad": orientation_error(predicted[:3, :3], loaded.data.xmat[tip_id].reshape(3, 3)),
    })
    return result
