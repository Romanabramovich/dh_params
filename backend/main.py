"""Local HTTP entry point for the MJCF visualizer demo."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

# EGL lets the local server render without a visible MuJoCo window on Linux.
# Set MUJOCO_GL explicitly to override this default.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from backend.model_loader import ModelLoadError, compile_mjcf, model_summary
from backend.pose import PoseError, normalize_qpos, selected_chain
from backend.rendering import RenderService, adjusted_camera, default_camera


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.loaded_model = None
    application.state.render_service = RenderService()
    try:
        yield
    finally:
        await application.state.render_service.close()


app = FastAPI(title="MJCF DH visualizer", lifespan=lifespan)


def summary_with_camera(loaded) -> dict:
    return {**model_summary(loaded), "camera": default_camera(loaded.model).as_dict()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/app.js")
def javascript() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})


@app.get("/style.css")
def stylesheet() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@app.get("/api/model")
def current_model() -> JSONResponse:
    loaded = app.state.loaded_model
    if loaded is None:
        return JSONResponse(status_code=404, content={"error": {"code": "no_model", "message": "No MJCF model is loaded."}})
    return JSONResponse(summary_with_camera(loaded))


@app.post("/api/model")
async def upload_model(file: UploadFile = File(...)) -> JSONResponse:
    filename = file.filename or ""
    if Path(filename).suffix.lower() not in {".xml", ".mjcf"}:
        return JSONResponse(status_code=415, content={"error": {"code": "file_type", "message": "Choose an MJCF .xml or .mjcf file."}})

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={"error": {"code": "file_too_large", "message": "MJCF files must be 16 MB or smaller."}})

    try:
        loaded = compile_mjcf(contents, filename)
    except ModelLoadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})

    await app.state.render_service.replace_model(loaded)
    app.state.loaded_model = loaded
    return JSONResponse(summary_with_camera(loaded))


@app.get("/api/pose")
async def current_pose(chain_id: str | None = None) -> JSONResponse:
    loaded = app.state.loaded_model
    if loaded is None:
        return JSONResponse(status_code=404, content={"error": {"code": "no_model", "message": "No MJCF model is loaded."}})
    try:
        snapshot = await app.state.render_service.snapshot(loaded, chain_id)
    except PoseError as exc:
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_chain", "message": str(exc)}})
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"error": {"code": "model_changed", "message": str(exc)}})
    return JSONResponse(snapshot)


@app.put("/api/pose")
async def update_pose(payload: dict) -> JSONResponse:
    loaded = app.state.loaded_model
    if loaded is None:
        return JSONResponse(status_code=404, content={"error": {"code": "no_model", "message": "No MJCF model is loaded."}})
    try:
        chain_id = payload.get("chain_id")
        if chain_id is not None and not isinstance(chain_id, str):
            raise PoseError("chain_id must be a string.")
        selected_chain(loaded.model, loaded.chains, chain_id)
        qpos = normalize_qpos(loaded.model, payload.get("qpos"))
        snapshot = await app.state.render_service.set_pose(loaded, qpos, chain_id)
    except PoseError as exc:
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_pose", "message": str(exc)}})
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"error": {"code": "model_changed", "message": str(exc)}})
    return JSONResponse(snapshot)


@app.get("/api/frame")
async def frame(
    azimuth: float | None = None,
    elevation: float | None = None,
    distance: float | None = None,
    chain_id: str | None = None,
) -> Response:
    loaded = app.state.loaded_model
    if loaded is None:
        return JSONResponse(status_code=404, content={"error": {"code": "no_model", "message": "No MJCF model is loaded."}})

    try:
        camera = adjusted_camera(default_camera(loaded.model), azimuth, elevation, distance)
        selected_chain(loaded.model, loaded.chains, chain_id)
    except PoseError as exc:
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_chain", "message": str(exc)}})
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_camera", "message": str(exc)}})

    try:
        jpeg, pose_version = await app.state.render_service.render(camera, chain_id)
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": {"code": "render_error", "message": f"MuJoCo could not render this model: {exc}"}})
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store", "X-Pose-Version": str(pose_version)})
