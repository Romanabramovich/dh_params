"""Compile one self-contained MJCF file and expose its compiled topology."""

from dataclasses import dataclass
from xml.etree import ElementTree

import mujoco


JOINT_TYPES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


class ModelLoadError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


@dataclass
class LoadedModel:
    filename: str
    model_name: str
    model: mujoco.MjModel
    data: mujoco.MjData


def compile_mjcf(contents: bytes, filename: str) -> LoadedModel:
    if not contents.strip():
        raise ModelLoadError("empty_file", "The MJCF file is empty.")

    try:
        xml = contents.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ModelLoadError("invalid_xml", "The MJCF file must use UTF-8 text.") from exc

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise ModelLoadError("invalid_xml", f"Invalid XML: {exc}.") from exc

    # A file drop supplies one XML document, not a directory of relative files.
    for element in root.iter():
        reference = element.get("file")
        if reference is not None:
            raise ModelLoadError(
                "external_file",
                f"This demo accepts standalone MJCF files. <{element.tag}> references '{reference}'.",
                status_code=422,
            )

    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
    except (mujoco.FatalError, ValueError) as exc:
        raise ModelLoadError("compile_error", f"MuJoCo could not compile this MJCF: {exc}.") from exc

    return LoadedModel(filename=filename, model_name=root.get("model", "unnamed model"), model=model, data=data)


def model_summary(loaded: LoadedModel) -> dict:
    model = loaded.model
    bodies = []
    for body_id in range(model.nbody):
        bodies.append({
            "id": body_id,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}",
            "parent_id": None if body_id == 0 else int(model.body_parentid[body_id]),
        })

    joints = []
    joint_counts = {name: 0 for name in JOINT_TYPES.values()}
    for joint_id in range(model.njnt):
        joint_type = JOINT_TYPES[int(model.jnt_type[joint_id])]
        joint_counts[joint_type] += 1
        joints.append({
            "id": joint_id,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}",
            "type": joint_type,
            "body_id": int(model.jnt_bodyid[joint_id]),
            "qpos_address": int(model.jnt_qposadr[joint_id]),
        })

    return {
        "filename": loaded.filename,
        "model_name": loaded.model_name,
        "body_count": model.nbody,
        "joint_count": model.njnt,
        "equality_count": model.neq,
        "qpos_size": model.nq,
        "joint_counts": joint_counts,
        "reference_qpos": loaded.data.qpos.tolist(),
        "bodies": bodies,
        "joints": joints,
    }
