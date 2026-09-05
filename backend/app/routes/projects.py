from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from backend.app.services.benchmark_service import peer_benchmark
from backend.app.services.data_service import row_to_dict, get_project
from backend.app.services.prediction_service import project_forecast, project_prediction
from backend.app.services.monthly_prediction_service import DEFAULT_PRODUCTION_WINDOW, lifecycle_project_forecast
from backend.app.services.portfolio_service import SORT_FIELDS, project_page
from backend.app.services.range_portfolio_service import RANGE_WINDOWS
from backend.app.services.range_portfolio_service import historical_peer_benchmark, historical_project, historical_warnings
from backend.app.schemas import ForecastResponse, LifecycleForecastResponse, PeerResponse, ProjectListResponse, ProjectRecord, WarningResponse

router = APIRouter(prefix="/api/projects", tags=["projects"])

@router.get("", response_model=ProjectListResponse)
def projects(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    limit: int | None = Query(None, ge=1, le=200, description="Deprecated alias for page_size."),
    search: str | None = None,
    sector: str | None = None,
    ministry: str | None = None,
    risk_level: str | None = None,
    sort: str = Query("risk_score"),
    direction: Literal["asc", "desc"] = "desc",
    window: str | None = None,
):
    if sort not in SORT_FIELDS:
        raise HTTPException(422, f"Unsupported sort field: {sort}")
    if window and window not in RANGE_WINDOWS:
        raise HTTPException(422, "Unsupported historical window")
    try:
        return project_page(page=page, page_size=limit or page_size, search=search, sector=sector, ministry=ministry, risk_level=risk_level, sort=sort, direction=direction, window=window)
    except ValueError as exc:
        raise HTTPException(409, str(exc))

@router.get("/{code}", response_model=ProjectRecord)
def project(code: str, window: str | None = None):
    if window:
        if window not in RANGE_WINDOWS:
            raise HTTPException(422, "Unsupported historical window")
        try:
            return historical_project(code, window)["record"]
        except KeyError:
            raise HTTPException(404, "Project not found")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
    try:
        return row_to_dict(get_project(code))
    except KeyError:
        raise HTTPException(404, "Project not found")

@router.get("/{code}/prediction", response_model=ForecastResponse, response_model_exclude_unset=True)
def prediction(code: str, window: str | None = None):
    if window:
        if window not in RANGE_WINDOWS:
            raise HTTPException(422, "Unsupported historical window")
        try:
            return historical_project(code, window)["forecast"]
        except KeyError:
            raise HTTPException(404, "Project not found")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
    try:
        return project_prediction(code)
    except KeyError:
        raise HTTPException(404, "Project not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
@router.get("/{code}/forecast", response_model=ForecastResponse, response_model_exclude_unset=True)
def forecast(code: str, window: str | None = None):
    if window:
        if window not in RANGE_WINDOWS:
            raise HTTPException(422, "Unsupported historical window")
        try:
            return historical_project(code, window)["forecast"]
        except KeyError:
            raise HTTPException(404, "Project not found")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
    try:
        return project_forecast(code)
    except KeyError:
        raise HTTPException(404, "Project not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.get("/{code}/lifecycle-forecast", response_model=LifecycleForecastResponse)
def lifecycle_forecast(code: str, window: str = DEFAULT_PRODUCTION_WINDOW):
    try:
        return lifecycle_project_forecast(code, window)
    except KeyError:
        raise HTTPException(404, "No official monthly trajectory is available for this project")
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc))

@router.get("/{code}/peers", response_model=PeerResponse)
def peers(code: str, window: str | None = None):
    if window:
        if window not in RANGE_WINDOWS:
            raise HTTPException(422, "Unsupported historical window")
        try:
            return historical_peer_benchmark(code, window)
        except KeyError:
            raise HTTPException(404, "Project not found")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
    try:
        return peer_benchmark(code)
    except KeyError:
        raise HTTPException(404, "Project not found")


@router.get("/{code}/warnings", response_model=WarningResponse)
def warnings(code: str, window: str | None = None):
    if not window:
        return {"available": False, "reason": "Project-specific warning events are not available for the live project dataset.", "source": None, "items": []}
    if window not in RANGE_WINDOWS:
        raise HTTPException(422, "Unsupported historical window")
    try:
        return historical_warnings(code, window)
    except KeyError:
        raise HTTPException(404, "Project not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
