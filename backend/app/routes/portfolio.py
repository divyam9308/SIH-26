from fastapi import APIRouter, HTTPException, Query
from backend.app.services.portfolio_service import summary, portfolio_rows, supported_windows
from backend.app.services.range_portfolio_service import RANGE_WINDOWS
from backend.app.schemas import PortfolioRiskResponse, PortfolioSummaryResponse

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

@router.get("/summary", response_model=PortfolioSummaryResponse)
def portfolio_summary(window: str | None = Query(None)):
    if window and window not in RANGE_WINDOWS:
        raise HTTPException(422, "Unsupported historical window")
    return summary(window)

@router.get("/risk", response_model=PortfolioRiskResponse)
def portfolio_risk(limit: int = 20, window: str | None = Query(None)):
    if window and window not in RANGE_WINDOWS:
        raise HTTPException(422, "Unsupported historical window")
    rows = sorted(portfolio_rows(window), key=lambda x: x["priority_score"], reverse=True)
    return {"items": rows[: max(1, min(limit, 100))]}


@router.get("/windows")
def portfolio_windows():
    return {"items": supported_windows()}
