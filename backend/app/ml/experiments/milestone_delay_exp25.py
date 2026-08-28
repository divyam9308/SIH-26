"""Experiment 25: project semantics, structured context, and milestone trajectories.

Project names are never used as raw high-cardinality model features. They are
parsed into reusable infrastructure semantics and scope/complexity indicators.
Milestone trajectory values are causal: snapshot t uses only t and earlier rows
from the same canonical project.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import TRAJECTORIES

EXPERIMENT_ID = "exp_25"
EXPERIMENT_NAME = "Project semantics + contextual milestone trajectories"
EXPERIMENT_SCOPE = "cost_delay"

PROJECT_TYPE_RULES = [
    ("metro", r"\bmetro\b"),
    ("railway", r"\b(rail|railway|rail line|doubling|gauge conversion)\b"),
    ("airport", r"\b(airport|aerodrome|runway)\b"),
    ("port", r"\b(port|harbour|harbor|jetty|berth)\b"),
    ("hydro_power", r"\b(hydro|hydel|hydroelectric)\b"),
    ("thermal_power", r"\b(thermal|supercritical|ultra supercritical)\b"),
    ("solar_power", r"\bsolar\b"),
    ("wind_power", r"\bwind\b"),
    ("power_grid", r"\b(transmission|substation|power grid|power evacuation)\b"),
    ("bridge", r"\b(bridge|flyover|viaduct)\b"),
    ("tunnel", r"\btunnel\b"),
    ("road", r"\b(highway|expressway|road|bypass|ring road|lane)\b"),
    ("irrigation_water", r"\b(irrigation|canal|dam|reservoir|water supply|sewer|sewage)\b"),
    ("pipeline", r"\b(pipeline|pipe line|gas line|oil line)\b"),
    ("hospital", r"\b(hospital|medical college|medical institute)\b"),
    ("building", r"\b(building|housing|campus|complex|office)\b"),
]

SEMANTIC_FEATURES = [
    "exp25_project_type",
    "exp25_has_phase",
    "exp25_has_stage",
    "exp25_has_package",
    "exp25_has_corridor",
    "exp25_has_extension",
    "exp25_has_greenfield",
    "exp25_has_modernisation",
    "exp25_phase_number",
    "exp25_stage_number",
    "exp25_package_number",
    "exp25_lane_count",
    "exp25_capacity_mw",
    "exp25_length_km",
    "exp25_unit_count",
    "exp25_scope_marker_count",
    "exp25_sector_project_type",
    "exp25_state_project_type",
]

# These are all prediction-time PAIMANA fields discussed for contextualization.
# Many are already part of production; dict-order de-duplication keeps the exact
# production contract while making state/financial-progress semantics available.
STRUCTURED_CONTEXT_FEATURES = [
    "sector",
    "ministry",
    "implementing_agency",
    "state",
    "approved_cost_cr",
    "revised_cost_cr",
    "cumulative_expenditure_cr",
    "physical_progress",
    "current_schedule_status",
    "exp25_financial_progress",
    "exp25_financial_physical_gap",
]

MILESTONE_FEATURES = [
    "exp25_milestones_achieved",
    "exp25_milestones_total",
    "exp25_milestone_ratio",
    "exp25_milestones_remaining",
    "exp25_milestone_velocity",
    "exp25_milestone_delta",
    "exp25_milestone_stagnant",
    "exp25_months_since_milestone_change",
]

CONTEXT_FEATURES = list(dict.fromkeys(SEMANTIC_FEATURES + STRUCTURED_CONTEXT_FEATURES))
ALL_ADDED_FEATURES = list(dict.fromkeys(CONTEXT_FEATURES + MILESTONE_FEATURES))


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip().lower()


def _number_from(text: str, pattern: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return np.nan
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return np.nan


def _ordinal_from(text: str, word: str) -> float:
    numeric = _number_from(text, rf"\b{word}\s*[-:/]?\s*(\d{{1,3}})\b")
    if pd.notna(numeric):
        return numeric
    roman = re.search(rf"\b{word}\s*[-:/]?\s*(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b", text, flags=re.IGNORECASE)
    if not roman:
        return np.nan
    mapping = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
    return float(mapping[roman.group(1).lower()])


def _project_type(text: str) -> str:
    for label, pattern in PROJECT_TYPE_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return "other"


def add_project_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive stable, prediction-time project semantics without raw-name memorization."""
    result = frame.copy()
    names = result.get("project_name", pd.Series("", index=result.index)).map(_text)
    sectors = result.get("sector", pd.Series("", index=result.index)).map(_text)
    states = result.get("state", pd.Series("", index=result.index)).map(_text)

    result["exp25_project_type"] = names.map(_project_type).astype("string")
    flag_patterns = {
        "exp25_has_phase": r"\bphase\b",
        "exp25_has_stage": r"\bstage\b",
        "exp25_has_package": r"\b(package|pkg)\b",
        "exp25_has_corridor": r"\bcorridor\b",
        "exp25_has_extension": r"\b(extension|extn)\b",
        "exp25_has_greenfield": r"\bgreenfield\b",
        "exp25_has_modernisation": r"\b(modernisation|modernization|upgradation|upgrade)\b",
    }
    for feature, pattern in flag_patterns.items():
        result[feature] = names.str.contains(pattern, case=False, regex=True, na=False).astype(float)

    result["exp25_phase_number"] = names.map(lambda value: _ordinal_from(value, "phase"))
    result["exp25_stage_number"] = names.map(lambda value: _ordinal_from(value, "stage"))
    result["exp25_package_number"] = names.map(
        lambda value: _number_from(value, r"\b(?:package|pkg)\s*[-:/]?\s*(\d{1,3})\b")
    )
    result["exp25_lane_count"] = names.map(
        lambda value: _number_from(value, r"\b(\d{1,2})\s*[- ]?lane\b")
    )
    result["exp25_capacity_mw"] = names.map(
        lambda value: _number_from(value, r"\b(\d+(?:\.\d+)?)\s*mw\b")
    )
    result["exp25_length_km"] = names.map(
        lambda value: _number_from(value, r"\b(\d+(?:\.\d+)?)\s*km\b")
    )
    result["exp25_unit_count"] = names.map(
        lambda value: _number_from(value, r"\b(\d{1,3})\s*(?:units?|nos?\.?\s*units?)\b")
    )
    flag_columns = list(flag_patterns)
    result["exp25_scope_marker_count"] = result[flag_columns].sum(axis=1)

    ptype = result["exp25_project_type"].astype("string").fillna("other")
    result["exp25_sector_project_type"] = (sectors.replace("", "unknown") + "|" + ptype).astype("string")
    result["exp25_state_project_type"] = (states.replace("", "unknown") + "|" + ptype).astype("string")

    financial = pd.to_numeric(result.get("financial_progress"), errors="coerce")
    physical = pd.to_numeric(result.get("physical_progress"), errors="coerce")
    result["exp25_financial_progress"] = financial
    result["exp25_financial_physical_gap"] = financial - physical
    return result


def add_milestone_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Engineer current/past-only milestone features on a monthly history frame."""
    result = frame.copy()
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], errors="coerce")
    result["canonical_project_id"] = result["canonical_project_id"].astype("string")
    status = result.get("milestone_status", pd.Series(None, index=result.index)).astype("string")
    parts = status.str.extract(r"(?P<done>\d+)\s*/\s*(?P<total>\d+)")
    result["exp25_milestones_achieved"] = pd.to_numeric(parts["done"], errors="coerce")
    result["exp25_milestones_total"] = pd.to_numeric(parts["total"], errors="coerce")
    done = result["exp25_milestones_achieved"]
    total = result["exp25_milestones_total"]
    result["exp25_milestone_ratio"] = (done / total).where(total.gt(0)).clip(0, 1)
    result["exp25_milestones_remaining"] = (total - done).where(total.notna() & done.notna()).clip(lower=0)
    for name in MILESTONE_FEATURES[4:]:
        result[name] = np.nan

    ordered = result.sort_values(["canonical_project_id", "snapshot_date"])
    for _, group in ordered.groupby("canonical_project_id", sort=False):
        idx = group.index
        dates = group["snapshot_date"]
        achieved = group["exp25_milestones_achieved"]
        ratios = group["exp25_milestone_ratio"]
        months = dates.diff().dt.days / 30.4375
        delta = achieved.diff()
        velocity = ratios.diff().div(months).where(months.gt(0))
        stagnant = pd.Series(np.where(delta.notna(), (delta <= 0).astype(float), np.nan), index=idx)
        since = pd.Series(np.nan, index=idx, dtype=float)
        previous = np.nan
        last_change = None
        for row_index in idx:
            current = result.at[row_index, "exp25_milestones_achieved"]
            current_date = result.at[row_index, "snapshot_date"]
            if pd.isna(current) or pd.isna(current_date):
                continue
            if pd.isna(previous) or current != previous:
                last_change = current_date
            if last_change is not None:
                since.at[row_index] = max(0.0, (current_date - last_change).days / 30.4375)
            previous = current
        result.loc[idx, "exp25_milestone_velocity"] = velocity.to_numpy()
        result.loc[idx, "exp25_milestone_delta"] = delta.to_numpy()
        result.loc[idx, "exp25_milestone_stagnant"] = stagnant.to_numpy()
        result.loc[idx, "exp25_months_since_milestone_change"] = since.to_numpy()
    return result


def enrich_exp25_features(frame: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add project semantics/context plus causal monthly milestone trajectories."""
    supervised = add_project_context_features(frame)
    supervised["snapshot_date"] = pd.to_datetime(supervised["snapshot_date"], errors="coerce")
    supervised["canonical_project_id"] = supervised["canonical_project_id"].astype("string")
    if history is None:
        if TRAJECTORIES.exists():
            history = pd.read_csv(
                TRAJECTORIES,
                dtype={"canonical_project_id": "string"},
                low_memory=False,
            )
        else:
            history = frame.copy()
    monthly = add_milestone_features(history)
    lookup = monthly[["canonical_project_id", "snapshot_date", *MILESTONE_FEATURES]].drop_duplicates(
        ["canonical_project_id", "snapshot_date"], keep="last"
    )
    supervised = supervised.drop(columns=[c for c in MILESTONE_FEATURES if c in supervised], errors="ignore")
    return supervised.merge(
        lookup,
        on=["canonical_project_id", "snapshot_date"],
        how="left",
        validate="many_to_one",
    )


def decision(cost_improvement: float, delay_improvement: float) -> str:
    """Promotion requires no target regression and a strict gain on at least one target."""
    if cost_improvement >= 0 and delay_improvement >= 0 and (cost_improvement > 0 or delay_improvement > 0):
        return "PROMOTION CANDIDATE"
    return "REGRESSION / DO NOT PROMOTE"
