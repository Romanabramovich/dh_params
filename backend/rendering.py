"""Offscreen MuJoCo rendering on one graphics-context thread."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import math

import mujoco
import numpy as np
from PIL import Image

from backend.model_loader import LoadedModel
from backend.pose import chain_frames, constraint_diagnostics, pose_snapshot, selected_chain


FRAME_WIDTH = 800
FRAME_HEIGHT = 600


def _add_line(scene, start, end, color, width=3.0) -> None:
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(np.asarray(end) - start) < 1e-10:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
        np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(start), np.asarray(end))
    scene.ngeom += 1


def _add_marker(scene, point, color, radius) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius]), np.asarray(point), np.eye(3).ravel(),
        np.asarray(color, dtype=np.float32),
    )
    scene.ngeom += 1


def append_pose_overlays(scene, loaded: LoadedModel, chain: dict | None) -> None:
    scale = max(0.03, 0.08 * float(loaded.model.stat.extent))
    if chain is not None:
        for frame in chain_frames(loaded, chain):
            origin = frame[:3, 3]
            for index, color in enumerate(((1, 0.25, 0.25, 1), (0.25, 1, 0.25, 1), (0.25, 0.55, 1, 1))):
                _add_line(scene, origin, origin + frame[:3, index] * scale, color, 3.0)
    for constraint in loaded.constraints:
        if not constraint["active"] or constraint["anchors"] is None:
            continue
        first, second = [np.asarray(point) for point in constraint["anchors"]]
        color = (1, 0.72, 0.15, 1) if constraint["anchor_separation_m"] < 1e-5 else (1, 0.3, 0.5, 1)
        _add_line(scene, first, second, color, 4.0)
        _add_marker(scene, first, color, scale * 0.12)
        _add_marker(scene, second, color, scale * 0.12)


@dataclass(frozen=True)
class CameraState:
    lookat: tuple[float, float, float]
    azimuth: float
    elevation: float
    distance: float

    def as_dict(self) -> dict:
        return {
            "lookat": list(self.lookat),
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "distance": self.distance,
        }


def default_camera(model: mujoco.MjModel) -> CameraState:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    return CameraState(
        lookat=tuple(float(value) for value in camera.lookat),
        azimuth=float(camera.azimuth),
        elevation=float(camera.elevation),
        distance=float(camera.distance),
    )


def adjusted_camera(
    default: CameraState,
    azimuth: float | None,
    elevation: float | None,
    distance: float | None,
) -> CameraState:
    values = (azimuth, elevation, distance)
    if any(value is not None and not math.isfinite(value) for value in values):
        raise ValueError("Camera values must be finite.")
    if distance is not None and distance <= 0:
        raise ValueError("Camera distance must be positive.")

    azimuth = default.azimuth if azimuth is None else (azimuth + 180) % 360 - 180
    elevation = default.elevation if elevation is None else max(-89.0, min(89.0, elevation))
    minimum = max(0.01, default.distance * 0.05)
    maximum = max(minimum, default.distance * 20.0)
    distance = default.distance if distance is None else max(minimum, min(maximum, distance))
    return CameraState(default.lookat, azimuth, elevation, distance)


class RenderService:
    """Own the renderer and all GL calls on a single worker thread."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco-render")
        self._loaded: LoadedModel | None = None
        self._renderer: mujoco.Renderer | None = None

    async def replace_model(self, loaded: LoadedModel) -> None:
        await asyncio.wrap_future(self._executor.submit(self._replace_model, loaded))

    def _replace_model(self, loaded: LoadedModel) -> None:
        self._close_renderer()
        self._loaded = loaded

    async def set_pose(self, loaded: LoadedModel, qpos: np.ndarray, chain_id: str | None) -> dict:
        return await asyncio.wrap_future(self._executor.submit(self._set_pose, loaded, qpos, chain_id))

    def _set_pose(self, loaded: LoadedModel, qpos: np.ndarray, chain_id: str | None) -> dict:
        if self._loaded is not loaded:
            raise RuntimeError("The loaded model changed. Try the pose again.")
        chain = selected_chain(loaded.model, loaded.chains, chain_id)
        loaded.data.qpos[:] = qpos
        mujoco.mj_forward(loaded.model, loaded.data)
        loaded.pose_version += 1
        loaded.pose_frames.clear()
        loaded.constraints = constraint_diagnostics(loaded.model, loaded.data)
        return pose_snapshot(loaded, chain)

    async def snapshot(self, loaded: LoadedModel, chain_id: str | None) -> dict:
        return await asyncio.wrap_future(self._executor.submit(self._snapshot, loaded, chain_id))

    def _snapshot(self, loaded: LoadedModel, chain_id: str | None) -> dict:
        if self._loaded is not loaded:
            raise RuntimeError("The loaded model changed. Try again.")
        return pose_snapshot(loaded, selected_chain(loaded.model, loaded.chains, chain_id))

    async def render(self, camera: CameraState, chain_id: str | None = None) -> tuple[bytes, int]:
        return await asyncio.wrap_future(self._executor.submit(self._render, camera, chain_id))

    def _render(self, camera: CameraState, chain_id: str | None) -> tuple[bytes, int]:
        if self._loaded is None:
            raise RuntimeError("No MJCF model is loaded.")
        if self._renderer is None:
            visual = self._loaded.model.vis.global_
            visual.offwidth = max(visual.offwidth, FRAME_WIDTH)
            visual.offheight = max(visual.offheight, FRAME_HEIGHT)
            self._renderer = mujoco.Renderer(self._loaded.model, width=FRAME_WIDTH, height=FRAME_HEIGHT)

        view = mujoco.MjvCamera()
        view.type = mujoco.mjtCamera.mjCAMERA_FREE
        view.lookat[:] = camera.lookat
        view.azimuth = camera.azimuth
        view.elevation = camera.elevation
        view.distance = camera.distance
        self._renderer.update_scene(self._loaded.data, camera=view)
        chain = selected_chain(self._loaded.model, self._loaded.chains, chain_id)
        append_pose_overlays(self._renderer.scene, self._loaded, chain)

        output = BytesIO()
        Image.fromarray(self._renderer.render()).save(output, format="JPEG", quality=85)
        return output.getvalue(), self._loaded.pose_version

    def _close_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    async def close(self) -> None:
        await asyncio.wrap_future(self._executor.submit(self._close_renderer))
        self._executor.shutdown(wait=True)
