from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.services.model_service import model_table, global_importances
from backend.app.services.validation_service import rolling_validation_report, validation_payload, validation_report
from backend.app.services.lifecycle_retraining_service import retrain_lifecycle
from backend.app.services.lifecycle_run_service import lifecycle_runs
from backend.app.services.monthly_prediction_service import DEFAULT_PRODUCTION_WINDOW, lifecycle_comparison, forecast_evolution
from backend.app.ml.residual_overrun_experiment import run_residual_overrun_experiment
from backend.app.services.portfolio_service import invalidate_portfolio_cache
from backend.app.services.prediction_service import clear_prediction_caches
from backend.app.services.training_window_performance_service import training_window_performance

router = APIRouter(prefix="/api/models", tags=["models"])


class TrainingRange(BaseModel):
    start_year: int
    end_year: int


@router.get("/metrics")
def metrics():
    return model_table()


@router.get("/importance")
def importance(model_version: str | None = None, model: str | None = None):
    selected = model_version or model
    try:
        return global_importances(selected)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@router.get("/lifecycle-runs")
def lifecycle_run_registry():
    """Return lifecycle model runs that really exist in this checkout/runtime."""
    return lifecycle_runs()


@router.post("/retrain")
def retrain_model(payload: TrainingRange):
    """Retrain the production monthly-lifecycle stack for the selected years."""
    try:
        result = retrain_lifecycle(payload.start_year, payload.end_year)
        clear_prediction_caches()
        invalidate_portfolio_cache()
        return result
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


@router.post("/experiments/residual-overrun")
def residual_overrun_experiment(payload: TrainingRange):
    """Run Experiment 3 without replacing the production lifecycle model."""
    try:
        return run_residual_overrun_experiment(payload.start_year, payload.end_year)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(409, str(exc))


@router.get("/validation")
def validation(model_version: str | None = None, model: str | None = None):
    selected = model_version or model
    try:
        return validation_report(selected)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(409, str(exc))


@router.get("/prediction-validation")
def prediction_validation(limit: int = 100, model_version: str | None = None, model: str | None = None):
    selected = model_version or model
    try:
        return validation_payload(limit, selected)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(409, str(exc))


@router.get("/rolling-validation")
def rolling_validation(model_version: str | None = None, model: str | None = None):
    selected = model_version or model
    try:
        return rolling_validation_report(selected)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@router.get("/monthly-lifecycle-comparison")
def monthly_lifecycle_comparison():
    return lifecycle_comparison()


@router.get("/training-window-performance")
def training_window_metrics():
    try:
        return training_window_performance()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(409, str(exc))


@router.get("/monthly-lifecycle-evolution/{project_id}")
def monthly_lifecycle_evolution(project_id: str, window: str = DEFAULT_PRODUCTION_WINDOW):
    try:
        return forecast_evolution(project_id, window)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
