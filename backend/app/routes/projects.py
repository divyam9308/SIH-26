from fastapi import APIRouter, HTTPException
from backend.app.services.benchmark_service import peer_benchmark
from backend.app.services.data_service import list_projects, row_to_dict, sectors, get_project
from backend.app.services.prediction_service import project_forecast, project_prediction
from backend.app.services.monthly_prediction_service import DEFAULT_PRODUCTION_WINDOW, lifecycle_project_forecast

router = APIRouter(prefix="/api/projects", tags=["projects"])

@router.get("")
def projects(search: str | None = None, sector: str | None = None, limit: int = 100):
    df = list_projects(search, sector).head(max(1, min(limit, 200)))
    return {"items": [row_to_dict(r) for _, r in df.iterrows()], "sectors": sectors()}

@router.get("/{code}")
def project(code: str):
    try:
        return row_to_dict(get_project(code))
    except KeyError:
        raise HTTPException(404, "Project not found")

@router.get("/{code}/prediction")
def prediction(code: str):
    try:
        return project_prediction(code)
    except KeyError:
        raise HTTPException(404, "Project not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))

@router.get("/{code}/forecast")
def forecast(code: str):
    try:
        return project_forecast(code)
    except KeyError:
        raise HTTPException(404, "Project not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.get("/{code}/lifecycle-forecast")
def lifecycle_forecast(code: str, window: str = DEFAULT_PRODUCTION_WINDOW):
    try:
        return lifecycle_project_forecast(code, window)
    except KeyError:
        raise HTTPException(404, "No official monthly trajectory is available for this project")
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc))

@router.get("/{code}/peers")
def peers(code: str):
    try:
        return peer_benchmark(code)
    except KeyError:
        raise HTTPException(404, "Project not found")
