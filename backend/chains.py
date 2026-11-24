"""Discover movable root-to-tip paths in a compiled MuJoCo body tree."""

import mujoco


def discover_chains(model: mujoco.MjModel) -> list[dict]:
    """Return one chain per leaf with at least one joint on its path.

    The body path retains fixed bodies for later base/tool transforms, while
    the ordered joint path omits them. MuJoCo's compiled body and joint IDs
    provide stable ordering within a model.
    """
    children = [[] for _ in range(model.nbody)]
    for body_id in range(1, model.nbody):
        children[int(model.body_parentid[body_id])].append(body_id)

    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        for body_id in range(model.nbody)
    ]
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        for joint_id in range(model.njnt)
    ]

    chains = []
    for tip_body_id in range(1, model.nbody):
        if children[tip_body_id]:
            continue

        body_ids = []
        body_id = tip_body_id
        while body_id != 0:
            body_ids.append(body_id)
            body_id = int(model.body_parentid[body_id])
        body_ids.append(0)
        body_ids.reverse()

        joints = []
        for body_id in body_ids[1:]:
            first_joint = int(model.body_jntadr[body_id])
            joint_count = int(model.body_jntnum[body_id])
            joints.extend(range(first_joint, first_joint + joint_count))
        if not joints:
            continue

        chains.append({
            "id": f"tip-{tip_body_id}",
            "name": body_names[tip_body_id],
            "tip_body_id": tip_body_id,
            "body_ids": body_ids,
            "body_names": [body_names[body_id] for body_id in body_ids],
            "joint_ids": joints,
            "joint_names": [joint_names[joint_id] for joint_id in joints],
        })

    return chains
