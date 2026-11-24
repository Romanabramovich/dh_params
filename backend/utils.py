
import numpy as np


def normalize(vector, epsilon=1e-12):
    normal = np.linalg.norm(vector)
    if normal < epsilon:
        raise ValueError("Cannot normalize zero vector")
    return vector / normal


def perpendicular(v, preferred=None):
    """Deterministically find a unit vector perpendicular to v."""
    v = normalize(v)

    if preferred is not None:
        x = preferred - np.dot(preferred, v) * v
        if np.linalg.norm(x) > 1e-9:
            return normalize(x)

    basis = np.eye(3)
    candidate = basis[np.argmin(np.abs(basis @ v))]
    return normalize(candidate - np.dot(candidate, v) * v)


def assign_dh_pair(
    p1, z1, p2, z2,
    previous_x=None,
    previous_origin=None,
    angular_tol=1e-7,
    distance_tol=1e-8
):
    """
    Construct a candidate standard-DH frame associated
    with the second axis.

    Inputs are expressed in the same reference frame.
    """
    p1, p2 = np.asarray(p1), np.asarray(p2)
    u, v = normalize(z1), normalize(z2)

    w = p2 - p1
    b = np.clip(np.dot(u, v), -1.0, 1.0)
    cross = np.cross(u, v)

    # Parallel or collinear
    if np.linalg.norm(cross) < angular_tol:

        reference = (
            p1 if previous_origin is None
            else previous_origin
        )

        # Closest point on axis 2 to reference
        c2 = p2 + np.dot(reference - p2, v) * v

        # Difference perpendicular to the common axis
        delta = c2 - reference
        delta = delta - np.dot(delta, v) * v

        if np.linalg.norm(delta) < distance_tol:
            case = "collinear"
            x = perpendicular(v, previous_x)
        else:
            case = "parallel"
            x = normalize(delta)

    else:
        A = np.dot(u, w)
        B = np.dot(v, w)
        denom = 1.0 - b * b

        s = (A - b * B) / denom
        t = (b * A - B) / denom

        c1 = p1 + s * u
        c2 = p2 + t * v

        delta = c2 - c1

        if np.linalg.norm(delta) < distance_tol:
            case = "intersecting"
            x = normalize(cross)
        else:
            case = "skew"
            x = normalize(delta)

    z = v
    y = normalize(np.cross(z, x))
    x = normalize(np.cross(y, z))

    R = np.column_stack((x, y, z))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c2

    return {
        "case": case,
        "transform": T,
        "origin": c2,
        "x": x,
        "y": y,
        "z": z
    }
