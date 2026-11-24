"""Offscreen MuJoCo rendering on one graphics-context thread."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import math

import mujoco
from PIL import Image

from backend.model_loader import LoadedModel


FRAME_WIDTH = 800
FRAME_HEIGHT = 600


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

    async def render(self, camera: CameraState) -> bytes:
        return await asyncio.wrap_future(self._executor.submit(self._render, camera))

    def _render(self, camera: CameraState) -> bytes:
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

        output = BytesIO()
        Image.fromarray(self._renderer.render()).save(output, format="JPEG", quality=85)
        return output.getvalue()

    def _close_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    async def close(self) -> None:
        await asyncio.wrap_future(self._executor.submit(self._close_renderer))
        self._executor.shutdown(wait=True)
