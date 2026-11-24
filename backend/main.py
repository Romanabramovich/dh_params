"""Local HTTP entry point for the MJCF visualizer demo."""

from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend.model_loader import ModelLoadError, compile_mjcf, model_summary


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

app = FastAPI(title="MJCF DH visualizer")
app.state.loaded_model = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/app.js")
def javascript() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.js", media_type="text/javascript")


@app.get("/style.css")
def stylesheet() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@app.get("/api/model")
def current_model() -> JSONResponse:
    loaded = app.state.loaded_model
    if loaded is None:
        return JSONResponse(status_code=404, content={"error": {"code": "no_model", "message": "No MJCF model is loaded."}})
    return JSONResponse(model_summary(loaded))


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

    app.state.loaded_model = loaded
    return JSONResponse(model_summary(loaded))
