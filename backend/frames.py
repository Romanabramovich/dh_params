"""Assign reference standard-DH frames and cache their moving attachments."""

from dataclasses import dataclass

import mujoco
import numpy as np

from backend.utils import assign_dh_pair, normalize, perpendicular
from backend.virtual_joints import trace_virtual_chain


ANGULAR_TOL = 1e-7


def _frame(origin, z_axis, preferred_x) -> np.ndarray:
    z_axis = normalize(z_axis)
    x_axis = perpendicular(z_axis, preferred_x)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = origin
    return transform


@dataclass
class FrameAssignment:
    axes: list[dict]
    pairs: list[dict]
    reference_frames: list[np.ndarray]
    local_frames: list[np.ndarray]
    base_transform: np.ndarray
    tool_transform: np.ndarray
    distance_tolerance: float

    def as_dict(self) -> dict:
        frames = []
        for index, transform in enumerate(self.reference_frames):
            frames.append({
                "index": index,
                "origin": transform[:3, 3].tolist(),
                "x": transform[:3, 0].tolist(),
                "y": transform[:3, 1].tolist(),
                "z": transform[:3, 2].tolist(),
                "transform": transform.tolist(),
            })
        return {
            "convention": "standard",
            "angular_tolerance": ANGULAR_TOL,
            "distance_tolerance": self.distance_tolerance,
            "axes": self.axes,
            "pairs": self.pairs,
            "frames": frames,
            "base_transform": self.base_transform.tolist(),
            "tool_transform": self.tool_transform.tolist(),
        }


def assign_chain_frames(model: mujoco.MjModel, data: mujoco.MjData, chain: dict) -> FrameAssignment:
    """Build F0..Fn once at qpos0, with F_i.z following virtual axis i+1."""
    trace = trace_virtual_chain(model, chain, model.qpos0)
    if not trace.rows:
        raise ValueError("A DH chain must contain at least one virtual joint.")

    axes = []
    for index, row in enumerate(trace.rows):
        point = trace.axis_points[index]
        direction = trace.axis_directions[index]
        if row["source_joint_type"] in ("hinge", "slide"):
            joint_id = row["source_joint_id"]
            point = data.xanchor[joint_id]
            direction = data.xaxis[joint_id]
        axes.append({
            "name": row["name"],
            "kind": row["kind"],
            "source_joint_id": row["source_joint_id"],
            "point": np.asarray(point, dtype=np.float64).tolist(),
            "direction": normalize(direction).tolist(),
        })

    scale = max(float(model.stat.extent), 1e-6)
    distance_tolerance = max(1e-12, 1e-8 * scale)
    first_x = perpendicular(axes[0]["direction"])
    if len(axes) > 1:
        first_pair = assign_dh_pair(
            axes[0]["point"], axes[0]["direction"], axes[1]["point"], axes[1]["direction"],
            angular_tol=ANGULAR_TOL, distance_tol=distance_tolerance,
        )
        first_x = first_pair["x"]

    reference_frames = [_frame(axes[0]["point"], axes[0]["direction"], first_x)]
    pairs = []
    for index in range(len(axes) - 1):
        result = assign_dh_pair(
            axes[index]["point"], axes[index]["direction"],
            axes[index + 1]["point"], axes[index + 1]["direction"],
            previous_x=reference_frames[-1][:3, 0],
            previous_origin=reference_frames[-1][:3, 3],
            angular_tol=ANGULAR_TOL,
            distance_tol=distance_tolerance,
        )
        reference_frames.append(result["transform"])
        pairs.append({
            "from": axes[index]["name"],
            "to": axes[index + 1]["name"],
            "frame_index": index + 1,
            "case": result["case"],
            "separation": result["distance"],
            "angle_sine": result["angle_sine"],
        })

    # The terminal frame follows the final joint. A fixed tool transform
    # retains the authored tip pose without inventing a terminal joint axis.
    reference_frames.append(reference_frames[-1].copy())
    local_frames = [
        np.linalg.inv(prefix) @ frame
        for prefix, frame in zip(trace.prefix_transforms, reference_frames, strict=True)
    ]
    tool_transform = np.linalg.inv(reference_frames[-1]) @ trace.tip_transform
    return FrameAssignment(
        axes=axes,
        pairs=pairs,
        reference_frames=reference_frames,
        local_frames=local_frames,
        base_transform=reference_frames[0].copy(),
        tool_transform=tool_transform,
        distance_tolerance=distance_tolerance,
    )


def evaluate_frame_transforms(model: mujoco.MjModel, chain: dict, assignment: FrameAssignment, qpos) -> list[np.ndarray]:
    """Move cached frames with their virtual prefixes; do no axis reclassification."""
    trace = trace_virtual_chain(model, chain, qpos)
    return [prefix @ local for prefix, local in zip(trace.prefix_transforms, assignment.local_frames, strict=True)]
