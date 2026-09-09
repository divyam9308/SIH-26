from fastapi import APIRouter, HTTPException, Query

from backend.app.services.early_warning_service import build_early_warnings

router = APIRouter(prefix="/api/early-warnings", tags=["early-warnings"])


@router.get("")
def early_warnings(window: str = Query("2001_2022"), project_id: str | None = None, as_of: str | None = None, search: str | None = None, severity: str | None = None, sector: str | None = None, ministry: str | None = None, threshold_only: bool = False):
    try:
        return build_early_warnings(window, project_id=project_id, as_of=as_of, search=search, severity=severity, sector=sector, ministry=ministry, threshold_only=threshold_only)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
