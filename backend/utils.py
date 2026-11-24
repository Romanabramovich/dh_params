import numpy as np


def normalize(vector, epsilon=1e-12):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("Expected a finite 3D vector")
    normal = np.linalg.norm(vector)
    if normal < epsilon:
        raise ValueError("Cannot normalize zero vector")
    return vector / normal


def perpendicular(v, preferred=None):
    """Deterministically find a unit vector perpendicular to v."""
    v = normalize(v)

    if preferred is not None:
        preferred = np.asarray(preferred, dtype=np.float64)
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
    p1, p2 = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)
    if p1.shape != (3,) or p2.shape != (3,) or not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
        raise ValueError("Axis points must be finite 3D vectors")
    if angular_tol <= 0 or distance_tol <= 0:
        raise ValueError("Axis tolerances must be positive")
    u, v = normalize(z1), normalize(z2)

    w = p2 - p1
    b = np.clip(np.dot(u, v), -1.0, 1.0)
    cross = np.cross(u, v)

    # Parallel or collinear
    cross_norm = np.linalg.norm(cross)
    if cross_norm < angular_tol:
        reference = p1 if previous_origin is None else np.asarray(previous_origin, dtype=np.float64)
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise ValueError("Previous origin must be a finite 3D vector")
        # The previous frame origin need not lie on axis 1. Project it there
        # before measuring the separation between the two axes.
        c1 = p1 + np.dot(reference - p1, u) * u
        c2 = p2 + np.dot(c1 - p2, v) * v
        delta = c2 - c1

        if np.linalg.norm(delta) < distance_tol:
            case = "collinear"
            x = perpendicular(v, previous_x)
        else:
            case = "parallel"
            x = normalize(delta)

    else:
        A = np.dot(u, w)
        B = np.dot(v, w)
        denom = cross_norm * cross_norm

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
        "z": z,
        "foot_on_first": c1,
        "foot_on_second": c2,
        "distance": float(np.linalg.norm(delta)),
        "angle_sine": float(cross_norm),
    }
