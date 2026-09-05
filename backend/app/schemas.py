"""Explicit public API response contracts consumed by the React application."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class HealthResponse(ContractModel):
    status: str
    app: str
    version: str


class ProjectRecord(ContractModel):
    snapshot_date: str
    sector: str
    ministry: str | None
    implementing_agency: str | None = None
    project_code: str
    project_name: str
    original_cost_cr: float
    revised_cost_cr: float | None
    expenditure_cr: float | None
    original_end_date: str | None
    revised_end_date: str | None
    physical_progress_pct: float | None
    source_url: str
    days_to_original_deadline: int
    schedule_extension_days: float | None
    cost_escalation_pct: float | None
    expenditure_to_original_pct: float | None
    financial_progress_pct: float | None
    schedule_overrun_90d: float | None
    cost_overrun_5pct: float | None
    dq_expenditure_gt_revised: int
    dq_revised_date_before_original: int
    dq_missing_revised_cost: int
    dq_missing_revised_date: int
    dq_missing_progress: int


class CurrentStatus(ContractModel):
    snapshot_month: str
    physical_progress_percentage: float | None
    current_estimated_cost: float | None
    expenditure_cr: float | None
    planned_completion_date: str | None
    progress_delay_percentage_points: float | None


class ShapFactor(ContractModel):
    feature: str
    impact: float
    direction: str


class CapabilityStatus(ContractModel):
    available: bool
    reason: str | None = None
    source: str | None = None


class ExplanationProvenance(ContractModel):
    run_id: str | None
    dataset_fingerprint: str | None
    cache_identity: str
    method: str


class ExplanationSummary(ContractModel):
    """Published decomposition metadata for a frozen local explanation."""
    available: bool
    base_value: float | None = None
    prediction: float | None = None
    net_feature_impact: float | None = None
    displayed_factors_impact: float | None = None
    other_features_impact: float | None = None
    output: str | None = None
    predicted_class: str | None = None
    factor_count: int | None = None
    source: str | None = None
    reconstruction_verified: bool | None = None
    reference_description: str | None = None


class OperationalDriver(ContractModel):
    type: str
    label: str
    category: Literal["COST", "DELAY", "BOTH", "IMPLEMENTATION"]
    evidence: str
    provenance: Literal["direct", "derived"]
    source: str


class QuantileRange(ContractModel):
    p10: float
    p50: float
    p90: float


class ExpectedRange(ContractModel):
    cost_overrun_percentage: QuantileRange
    delay_days: QuantileRange


class CompletionProbability(ContractModel):
    year: int
    probability_percentage: float


class BestModels(ContractModel):
    cost: str
    delay: str


class ForecastResponse(ContractModel):
    project_id: str
    project_name: str
    model_version: str
    dataset_snapshot_date: str
    inference_timestamp: str
    current_status: CurrentStatus
    predicted_cost_overrun_percentage: float
    predicted_cost_overrun_amount_cr: float
    predicted_final_cost_cr: float
    predicted_delay_days: float
    predicted_cost_overrun: float
    predicted_completion_date: str | None
    current_progress: float | None
    predicted_delay_months: float
    risk_score: float
    risk_probability_percentage: float | None
    risk_level: RiskLevel
    model_confidence_percentage: float | None
    confidence_calibration_status: str
    explanation: list[ShapFactor]
    shap_explanation: list[ShapFactor]
    cost_factors: list[ShapFactor]
    delay_factors: list[ShapFactor]
    risk_factors: list[ShapFactor]
    cost_explanation_status: CapabilityStatus
    delay_explanation_status: CapabilityStatus
    risk_explanation_status: CapabilityStatus
    cost_explanation_summary: ExplanationSummary | None = None
    delay_explanation_summary: ExplanationSummary | None = None
    risk_explanation_summary: ExplanationSummary | None = None
    operational_drivers: list[OperationalDriver]
    explanation_provenance: ExplanationProvenance | None = None
    best_models: BestModels
    expected_range: ExpectedRange | None
    completion_probabilities: list[CompletionProbability]
    features_used: list[str]
    model_scope: str


class ObservedProjectValues(ContractModel):
    schedule_extension_days: float | None
    cost_escalation_pct: float | None
    financial_progress_pct: float | None
    physical_progress_pct: float | None


class PortfolioBestModels(ContractModel):
    schedule_classifier: str | None = None
    cost_classifier: str | None = None
    schedule_regressor: str | None = None
    cost_regressor: str | None = None


class ProjectListItem(ContractModel):
    project_code: str
    project_name: str
    sector: str
    ministry: str | None
    implementing_agency: str | None
    snapshot_date: str
    original_cost_cr: float | None
    revised_cost_cr: float | None
    expenditure_cr: float | None
    physical_progress_pct: float | None
    predicted_cost_overrun_percentage: float
    predicted_cost_overrun_amount_cr: float
    predicted_final_cost_cr: float
    predicted_delay_days: float
    predicted_delay_months: float
    predicted_completion_date: str | None
    actual_cost_overrun_percentage: float | None = None
    actual_delay_days: float | None = None
    cost_error_percentage: float | None = None
    delay_error_days: float | None = None
    risk_score: float
    risk_probability_percentage: float | None
    risk_level: RiskLevel
    model_version: str
    model_scope: str
    inference_timestamp: str
    model_confidence_percentage: float | None
    confidence_calibration_status: str


class PortfolioRiskItem(ProjectListItem):
    schedule_risk_probability: float | None
    cost_risk_probability: float | None
    estimated_schedule_extension_days: float | None
    estimated_cost_escalation_pct: float | None
    priority_score: float | None
    priority_level: str
    confidence: str
    exposure_percentile: float | None
    best_models: PortfolioBestModels
    schedule_drivers: list[dict]
    cost_drivers: list[dict]
    observed: ObservedProjectValues


class ProjectListResponse(ContractModel):
    items: list[PortfolioRiskItem]
    total: int
    page: int
    page_size: int
    pages: int
    sectors: list[str]
    ministries: list[str]
    risk_distribution: dict[str, int]
    cost_exposure_by_risk_cr: dict[str, float]
    predicted_cost_exposure_cr: float
    model_version: str | None
    dataset_snapshot: str | None
    inference_timestamp: str


class PortfolioRiskResponse(ContractModel):
    items: list[PortfolioRiskItem]


class ExpenditureProgressPoint(ContractModel):
    project_code: str
    physical_progress_pct: float
    financial_progress_pct: float
    group: Literal["On Track", "Monitor", "At Risk"]


class WarningDriver(ContractModel):
    name: str
    count: int


class PortfolioSummaryResponse(ContractModel):
    projects: int
    original_cost_cr: float
    current_cost_basis_cr: float
    expenditure_cr: float
    predicted_cost_exposure_cr: float
    risk_distribution: dict[str, int]
    cost_exposure_by_risk_cr: dict[str, float]
    sectors: int
    dataset_snapshot: str | None
    dataset_scope: str
    model_version: str | None
    model_scope: str | None
    inference_timestamp: str
    expenditure_progress: list[ExpenditureProgressPoint]
    warning_drivers: list[WarningDriver]
    risk_trend: None
    risk_trend_status: str


class PeerMedians(ContractModel):
    original_cost_cr: float | None
    cost_escalation_pct: float | None
    schedule_extension_days: float | None
    financial_progress_pct: float | None
    physical_progress_pct: float | None


class PeerProject(ContractModel):
    project_code: str
    project_name: str
    original_cost_cr: float | None
    cost_escalation_pct: float | None
    schedule_extension_days: float | None


class PeerResponse(ContractModel):
    sector: str
    peer_count: int
    medians: PeerMedians
    peers: list[PeerProject]


class HistoricalWarning(ContractModel):
    date: str
    type: str
    severity: RiskLevel
    message: str


class WarningResponse(ContractModel):
    available: bool
    reason: str | None = None
    source: str | None = None
    items: list[HistoricalWarning]


class LifecycleProvenance(ContractModel):
    run_id: str | None
    dataset_fingerprint: str | None
    verified: bool


class LifecycleGlobalFactor(ContractModel):
    feature: str
    importance: float


class LifecycleForecastResponse(ContractModel):
    project_id: str
    project_name: str
    model_version: str
    snapshot_date: str
    history_snapshots: int
    predicted_cost_overrun_percentage: float
    predicted_delay_days: float
    risk_level: str
    model_inputs: dict[str, object | None]
    cost_features_used: list[str]
    delay_features_used: list[str]
    risk_features_used: list[str]
    production_cost_baseline: str | None
    production_delay_baseline: str | None
    promoted_from_experiment: str | None
    promoted_delay_from_experiment: str | None
    shap_explanation: list[ShapFactor]
    global_feature_importance: list[LifecycleGlobalFactor]
    explanation_scope: str
    provenance: LifecycleProvenance
    model_scope: str
