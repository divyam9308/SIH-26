from pathlib import Path

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import APP_NAME, APP_VERSION, FRONTEND_DIR
from backend.app.routes import assistant, data_quality, history, models, portfolio, projects, scenario, simulations
FRONTEND_BUILD_DIR = FRONTEND_DIR / "dist"

app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.include_router(portfolio.router)
app.include_router(projects.router)
app.include_router(models.router)
app.include_router(history.router)
app.include_router(scenario.router)
app.include_router(assistant.router)
app.include_router(data_quality.router)
app.include_router(simulations.router)


@app.middleware("http")
async def disable_spa_source_cache(request: Request, call_next):
    """Local development must not retain stale ES modules after a reload."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response

@app.get("/api/health")
def health():
    return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}

if FRONTEND_BUILD_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_BUILD_DIR / "assets"), name="assets")

@app.get("/{full_path:path}")
def spa(full_path: str):
    if full_path.startswith("api/"):
        return {"detail": "Not found"}
    built_index = FRONTEND_BUILD_DIR / "index.html"
    if built_index.is_file():
        return FileResponse(built_index)
    return FileResponse(FRONTEND_DIR / "index.html")
