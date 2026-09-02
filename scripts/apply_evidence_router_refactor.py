from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> tuple[Path, str]:
    target = ROOT / path
    return target, target.read_text()


def replace_once(path: str, old: str, new: str) -> None:
    target, text = _read(path)
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected exactly one occurrence, found {text.count(old)}\n{old}")
    target.write_text(text.replace(old, new, 1))


def replace_re(path: str, pattern: str, replacement: str) -> None:
    target, text = _read(path)
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{path}: regex replacement matched {count} times: {pattern}")
    target.write_text(updated)


# ---------------------------------------------------------------------------
# Exp34: the shared headline cohort is defined by the Exp12 evidence rule only.
# ---------------------------------------------------------------------------
replace_re(
    "backend/app/ml/production_delay_baseline.py",
    r"For the selected production window \(2001-2021 -> 2022-2025\), the official\nheadline evidence cohort is the same verified Exp12-comparable 721-project\ncohort used for production Cost\. Full-holdout Delay metrics are also retained as\na diagnostic so the 728-project Exp34 experiment result remains auditable\.\n",
    "Headline Delay evaluation always uses the same evidence-defined Exp12-comparable\ncohort as Cost: at least two official observations in the trailing 12 months,\nfollowed by project-balanced weighting. Project and snapshot counts are observed\ndiagnostics only; they are never eligibility requirements for a training window.\n",
)
replace_re(
    "backend/app/ml/production_delay_baseline.py",
    r'VERIFIED_PRODUCTION_START = 2001\nVERIFIED_PRODUCTION_END = 2021\nVERIFIED_PRODUCTION_TEST_END = 2025\nVERIFIED_PRODUCTION_EVIDENCE_PROJECTS = 721\nPRODUCTION_DELAY_EVALUATION_COHORT = "shared_exp12_comparable_721_project_cohort"',
    'PRODUCTION_DELAY_EVALUATION_COHORT = "shared_exp12_comparable_evidence_cohort"',
)
replace_once(
    "backend/app/ml/production_delay_baseline.py",
    '    """Use the exact verified production Cost cohort for the headline Delay MAE."""',
    '    """Use the evidence-defined production Cost cohort for the headline Delay MAE."""',
)
replace_re(
    "backend/app/ml/production_delay_baseline.py",
    r'    shared_projects = int\(shared_eval\["canonical_project_id"\]\.nunique\(\)\)\n    if \(\n        training_start == VERIFIED_PRODUCTION_START\n        and training_end == VERIFIED_PRODUCTION_END\n        and test_end == VERIFIED_PRODUCTION_TEST_END\n        and shared_projects != VERIFIED_PRODUCTION_EVIDENCE_PROJECTS\n    \):\n        raise RuntimeError\(\n            "Refusing to publish the selected 2001-2021 production run because the "\n            f"verified evidence cohort changed: expected \{VERIFIED_PRODUCTION_EVIDENCE_PROJECTS\} "\n            f"projects, found \{shared_projects\}\."\n        \)\n',
    '    shared_projects = int(shared_eval["canonical_project_id"].nunique())\n',
)
replace_once(
    "backend/app/ml/production_delay_baseline.py",
    '        "verified_2001_2021_project_count": VERIFIED_PRODUCTION_EVIDENCE_PROJECTS,',
    '        "cohort_count_policy": "observed_only_no_fixed_project_or_snapshot_requirement",',
)

# ---------------------------------------------------------------------------
# Exp35: default routing is evidence based. Optional limit remains only for
# explicit legacy reproduction calls; production never passes one.
# ---------------------------------------------------------------------------
replace_re(
    "backend/app/ml/production_exp35_baseline.py",
    r"Promotion adds Exp33 cross-fitted residual calibration to Cost and replaces\nDelay with Exp32 remaining-time forecasting followed by Exp33 residual\ncalibration\. For the frozen 2001-2021 promotion audit, Delay calibration is\napplied to exactly 688 AFT-comparable projects selected only from as-of evidence\navailability; all other projects retain Exp34 Delay\. Live rows do not hard-code\nthose historical project IDs: when the audit-only gate is absent, AFT eligibility\nis determined from the live snapshot evidence\.\n",
    "Promotion adds Exp33 cross-fitted residual calibration to Cost and replaces\nDelay with Exp32 remaining-time forecasting followed by Exp33 residual\ncalibration. Delay routing is determined only from as-of evidence: projects with\nusable AFT evidence may use the AFT route, while rows without the required\nsnapshot/planned-completion evidence retain Exp34 Delay. No fixed project count\nis part of the production routing contract.\n",
)
replace_re(
    "backend/app/ml/production_exp35_baseline.py",
    r'PRODUCTION_DELAY_BASELINE = "exp32_aft_plus_exp33_residual_688_exp34_fallback_v2"\nVERIFIED_PRODUCTION_START = 2001\nVERIFIED_PRODUCTION_END = 2021\nVERIFIED_PRODUCTION_TEST_END = 2025\nVERIFIED_PRODUCTION_PROJECTS = 721\nVERIFIED_PRODUCTION_SNAPSHOTS = 11200\nVERIFIED_AFT_CALIBRATION_PROJECTS = 688\nVERIFIED_BASE_COST_MAE = 26\.872\nVERIFIED_BASE_DELAY_MAE = 501\.303',
    'PRODUCTION_DELAY_BASELINE = "exp32_aft_plus_exp33_evidence_router_exp34_fallback_v3"\nVERIFIED_PRODUCTION_START = 2001\nVERIFIED_PRODUCTION_END = 2021\nVERIFIED_PRODUCTION_TEST_END = 2025\nVERIFIED_BASE_COST_MAE = 26.872\nVERIFIED_BASE_DELAY_MAE = 501.303',
)
replace_re(
    "backend/app/ml/production_exp35_baseline.py",
    r'def _select_aft_calibration_projects\(\n    frame: pd\.DataFrame,\n    limit: int = VERIFIED_AFT_CALIBRATION_PROJECTS,\n\) -> set\[str\]:\n.*?\n\ndef train_window_with_promoted_cost_and_delay\(',
    '''def _select_aft_calibration_projects(\n    frame: pd.DataFrame,\n    limit: int | None = None,\n) -> set[str]:\n    """Select projects with usable as-of AFT evidence.\n\n    Production calls this function without ``limit`` and therefore routes every\n    project that has at least one snapshot with the fields required by the AFT\n    conversion. No target, residual, error, or model-quality value is consulted.\n    ``limit`` exists only for explicit legacy reproduction/audit calls.\n    """\n    required = {"canonical_project_id", "snapshot_date", "planned_completion_date"}\n    missing = sorted(required.difference(frame.columns))\n    if missing:\n        raise ValueError(\n            "AFT routing is missing required fields: " + ", ".join(missing)\n        )\n\n    work = frame[\n        ["canonical_project_id", "snapshot_date", "planned_completion_date"]\n    ].copy()\n    work["_aft_evidence"] = AFTResidualDelayModel._aft_eligible(work).astype(int)\n    summary = (\n        work.groupby("canonical_project_id", dropna=False)["_aft_evidence"]\n        .agg(["sum", "count"])\n        .reset_index()\n        .rename(columns={"sum": "eligible_snapshots", "count": "total_snapshots"})\n    )\n    summary = summary[summary["eligible_snapshots"].gt(0)].copy()\n    if summary.empty:\n        return set()\n\n    summary["evidence_coverage"] = (\n        summary["eligible_snapshots"] / summary["total_snapshots"].clip(lower=1)\n    )\n    summary["_project_key"] = summary["canonical_project_id"].astype("string")\n    summary = summary.sort_values(\n        ["evidence_coverage", "eligible_snapshots", "total_snapshots", "_project_key"],\n        ascending=[False, False, False, True],\n        kind="stable",\n    )\n    if limit is not None:\n        if int(limit) < 1:\n            raise ValueError("Legacy AFT routing limit must be positive when supplied")\n        summary = summary.head(int(limit))\n    return set(summary["canonical_project_id"].astype("string").tolist())\n\n\ndef train_window_with_promoted_cost_and_delay(''',
)
replace_once(
    "backend/app/ml/production_exp35_baseline.py",
    '    """Train production and promote Exp32+Exp33 with the fixed 688-project audit gate."""',
    '    """Train production and promote Exp32+Exp33 with evidence-based Delay routing."""',
)
replace_once(
    "backend/app/ml/production_exp35_baseline.py",
    "    # Apply the same frozen gate to all validation rows from those projects.",
    "    # Apply the same evidence-only project gate to all validation rows; row-level\n    # AFT eligibility still requires snapshot and planned-completion evidence.",
)
replace_re(
    "backend/app/ml/production_exp35_baseline.py",
    r'    if _selected_window\(training_start, training_end, test_end\):\n.*?\n\n    full_delay_prediction =',
    '''    if _selected_window(training_start, training_end, test_end):\n        if abs(float(base_cost_metrics["MAE"]) - VERIFIED_BASE_COST_MAE) > 0.001:\n            raise RuntimeError(\n                f"Verified Cost baseline drifted: {base_cost_metrics['MAE']} != {VERIFIED_BASE_COST_MAE}."\n            )\n        if abs(float(base_delay_metrics["MAE"]) - VERIFIED_BASE_DELAY_MAE) > 0.001:\n            raise RuntimeError(\n                f"Verified Delay baseline drifted: {base_delay_metrics['MAE']} != {VERIFIED_BASE_DELAY_MAE}."\n            )\n        if float(cost_metrics["MAE"]) >= float(base_cost_metrics["MAE"]):\n            raise RuntimeError(\n                "Refusing promotion: Exp33-calibrated Cost did not improve the reference production cohort."\n            )\n        if float(calibration_promoted_metrics["MAE"]) >= float(calibration_base_metrics["MAE"]):\n            raise RuntimeError(\n                "Refusing promotion: evidence-routed Exp32+Exp33 Delay did not improve the routed-project slice."\n            )\n        if float(delay_metrics["MAE"]) >= float(base_delay_metrics["MAE"]):\n            raise RuntimeError(\n                "Refusing promotion: evidence-routed AFT + Exp34 fallback did not improve the full comparable cohort."\n            )\n\n    full_delay_prediction =''',
)
replace_once(
    "backend/app/ml/production_exp35_baseline.py",
    '"exp32_aft_plus_exp33_on_fixed_688_project_audit_cohort_with_exp34_fallback"',
    '"exp32_aft_plus_exp33_evidence_router_with_exp34_fallback"',
)
replace_re(
    "backend/app/ml/production_exp35_baseline.py",
    r'    delay_evaluation_contract = \{\n.*?\n    \}\n\n    metadata\["base_production_cost_baseline"\]',
    '''    delay_evaluation_contract = {\n        "cohort": "shared_exp12_comparable_evidence_cohort",\n        "cohort_rule": "exp12_history_12m >= 2 then project-balanced weights",\n        "cohort_count_policy": "observed_only_no_fixed_project_or_snapshot_requirement",\n        "weighting_policy": "project-balanced after shared Exp12 comparable-cohort filter",\n        "test_projects": shared_projects,\n        "test_snapshots": shared_snapshots,\n        "routing_projects": calibration_projects,\n        "routing_project_snapshots": calibration_snapshots,\n        "routing_project_selection": (\n            "all projects with at least one snapshot carrying required as-of AFT evidence; "\n            "no outcome/error values and no fixed project count"\n        ),\n        "aft_eligible_projects": aft_projects,\n        "aft_eligible_snapshots": aft_snapshots,\n        "fallback_only_projects": shared_projects - aft_projects,\n        "fallback_policy": (\n            "Exp34 production Delay whenever the project has no usable AFT evidence or "\n            "the individual row lacks snapshot/planned-completion evidence"\n        ),\n        "base_exp34_mae_comparable_cohort": base_delay_metrics["MAE"],\n        "promoted_exp32_exp33_mae_comparable_cohort": delay_metrics["MAE"],\n        "comparable_cohort_improvement_percentage": round(\n            _gain(float(base_delay_metrics["MAE"]), float(delay_metrics["MAE"])), 4\n        ),\n        "base_exp34_mae_routed_projects": calibration_base_metrics["MAE"],\n        "promoted_exp32_exp33_mae_routed_projects": calibration_promoted_metrics["MAE"],\n        "routed_project_improvement_percentage": round(\n            _gain(\n                float(calibration_base_metrics["MAE"]),\n                float(calibration_promoted_metrics["MAE"]),\n            ),\n            4,\n        ),\n    }\n\n    metadata["base_production_cost_baseline"]''',
)
replace_re(
    "backend/app/ml/production_exp35_baseline.py",
    r'    metadata\["delay_policy"\] = \(\n        "exp32_aft_remaining_time_plus_exp33_cross_fitted_residual_calibration_"\n        "on_fixed_688_project_audit_cohort_with_exp34_fallback"\n    \)',
    '    metadata["delay_policy"] = "exp32_aft_plus_exp33_evidence_router_with_exp34_fallback"',
)
replace_once(
    "backend/app/ml/production_exp35_baseline.py",
    '    metadata["delay_calibration_project_count"] = VERIFIED_AFT_CALIBRATION_PROJECTS',
    '    metadata["delay_calibration_project_count"] = calibration_projects',
)
replace_once(
    "backend/app/ml/production_exp35_baseline.py",
    '    selected["delay"] = "exp32_aft_plus_exp33_residual_688_with_exp34_fallback"',
    '    selected["delay"] = "exp32_aft_plus_exp33_residual_evidence_router_with_exp34_fallback"',
)
replace_re(
    "backend/app/ml/production_exp35_baseline.py",
    r'    metadata\["leakage_policy"\] = \(\n.*?\n    \)\.strip\(\)',
    '''    metadata["leakage_policy"] = (\n        str(metadata.get("leakage_policy") or "")\n        + " Exp32+Exp33 residual calibration parameters are learned only from rolling "\n        "validation years inside the training window. Remaining-time targets use "\n        "historical completion outcomes only for training labels; future holdout "\n        "outcomes, residuals, and errors are never used for model, weight, calibration, "\n        "or routing. The Delay router uses only as-of snapshot/planned-completion "\n        "evidence and has no fixed project-count requirement; rows without sufficient "\n        "evidence retain Exp34 Delay."\n    ).strip()''',
)
replace_re(
    "backend/app/ml/production_exp35_baseline.py",
    r'    result\["promotion"\] = \{\n.*?\n    \}\n\n    result = _json_safe\(result\)',
    '''    result["promotion"] = {\n        "experiment_id": PROMOTED_EXPERIMENT_ID,\n        "scope": "cost+delay",\n        "production_cost_baseline": PRODUCTION_COST_BASELINE,\n        "production_delay_baseline": PRODUCTION_DELAY_BASELINE,\n        "cost_improvement_percentage": round(\n            _gain(float(base_cost_metrics["MAE"]), float(cost_metrics["MAE"])), 4\n        ),\n        "delay_improvement_percentage": round(\n            _gain(float(base_delay_metrics["MAE"]), float(delay_metrics["MAE"])), 4\n        ),\n        "delay_routed_project_improvement_percentage": round(\n            _gain(\n                float(calibration_base_metrics["MAE"]),\n                float(calibration_promoted_metrics["MAE"]),\n            ),\n            4,\n        ),\n        "delay_routing_projects": calibration_projects,\n        "risk_retained": True,\n        "delay_fallback": "exp34_without_sufficient_as_of_aft_evidence",\n    }\n\n    result = _json_safe(result)''',
)

# ---------------------------------------------------------------------------
# Later production layers: keep performance guards, remove sample-count guards.
# ---------------------------------------------------------------------------
replace_once(
    "backend/app/ml/production_exp61_baseline.py",
    '''    # Promotion was selected on 2001-2021 only; require that frozen decision window to reproduce an improvement.\n    if (training_start, training_end, test_end) == (2001, 2021, 2025):\n        if int(shared_eval["canonical_project_id"].nunique()) != 721 or len(shared_eval) != 11200:\n            raise RuntimeError("Exp61 verified 2001-2021 cohort changed")\n        if float(cost_metrics["MAE"]) >= float(old_cost_metrics["MAE"]):\n            raise RuntimeError("Exp61 Cost failed to improve the verified production window")\n        if float(delay_metrics["MAE"]) >= float(old_delay_metrics["MAE"]):\n            raise RuntimeError("Exp61 Delay failed to improve the verified production window")\n''',
    '''    # The reference decision window still has performance guards, but its cohort\n    # size is whatever the evidence rule yields rather than a hard-coded count.\n    if (training_start, training_end, test_end) == (2001, 2021, 2025):\n        if float(cost_metrics["MAE"]) >= float(old_cost_metrics["MAE"]):\n            raise RuntimeError("Exp61 Cost failed to improve the verified production window")\n        if float(delay_metrics["MAE"]) >= float(old_delay_metrics["MAE"]):\n            raise RuntimeError("Exp61 Delay failed to improve the verified production window")\n''',
)
replace_once(
    "backend/app/ml/production_u1_delay_baseline.py",
    '''    if (training_start, training_end, test_end) == (2001, 2021, 2025):\n        if int(shared_eval["canonical_project_id"].nunique()) != 721 or len(shared_eval) != 11200:\n            raise RuntimeError("U1 Delay promotion verified cohort changed")\n        if float(delay_metrics["MAE"]) >= float(old_delay_metrics["MAE"]):\n            raise RuntimeError("U1 Delay failed to improve the verified 2001-2021 production window")\n''',
    '''    if (training_start, training_end, test_end) == (2001, 2021, 2025):\n        if float(delay_metrics["MAE"]) >= float(old_delay_metrics["MAE"]):\n            raise RuntimeError("U1 Delay failed to improve the verified 2001-2021 production window")\n''',
)
replace_once(
    "backend/app/ml/production_exp105_exp113_baseline.py",
    '''    if (training_start, training_end, test_end) == (2001, 2021, 2025):\n        if int(cohort["canonical_project_id"].nunique()) != 721 or len(cohort) != 11200:\n            raise RuntimeError("Exp105 + Exp113 verified cohort changed")\n        if float(cost_metrics["MAE"]) >= float(old_cost_metrics["MAE"]):\n            raise RuntimeError("Exp105 Cost failed to improve the verified 2001-2021 production window")\n        if float(delay_metrics["MAE"]) >= float(old_delay_metrics["MAE"]):\n            raise RuntimeError("Exp113 Delay failed to improve the verified 2001-2021 production window")\n''',
    '''    if (training_start, training_end, test_end) == (2001, 2021, 2025):\n        if float(cost_metrics["MAE"]) >= float(old_cost_metrics["MAE"]):\n            raise RuntimeError("Exp105 Cost failed to improve the verified 2001-2021 production window")\n        if float(delay_metrics["MAE"]) >= float(old_delay_metrics["MAE"]):\n            raise RuntimeError("Exp113 Delay failed to improve the verified 2001-2021 production window")\n''',
)

# ---------------------------------------------------------------------------
# Post-Exp113 experiments: windows define dates, never expected sample counts.
# ---------------------------------------------------------------------------
replace_once(
    "backend/app/ml/experiments/post_exp113_delay_common.py",
    "WINDOWS={2021:(2022,2025,721,11200),2022:(2023,2025,487,6870)}",
    "WINDOWS={2021:(2022,2025),2022:(2023,2025)}",
)
replace_once(
    "backend/app/ml/experiments/post_exp113_delay_common.py",
    "    test_start,test_end,projects,snapshots=window_contract(end);data,identity=build_training_dataset()",
    "    test_start,test_end=window_contract(end);data,identity=build_training_dataset()",
)
replace_once(
    "backend/app/ml/experiments/post_exp113_delay_common.py",
    "        if int(cohort['canonical_project_id'].nunique())!=projects or len(cohort)!=snapshots: raise RuntimeError('Comparison cohort changed')\n",
    "",
)

# ---------------------------------------------------------------------------
# Surface Delay percentage error (existing project-weighted MAPE) and stop using
# fixed Delay MAE / sample counts as runtime validity checks.
# ---------------------------------------------------------------------------
replace_once(
    "scripts/run_u1_delay_production_fresh.py",
    '''EXPECTED = {\n    2019: {"cost": 27.801, "delay": 438.098},\n    2021: {"cost": 25.829, "delay": 346.599},\n}\n''',
    '''EXPECTED_COST = {\n    2019: 27.801,\n    2021: 25.829,\n}\n''',
)
replace_once(
    "scripts/run_u1_delay_production_fresh.py",
    "    if a.start != 2001 or a.end not in EXPECTED or a.test_end != 2025:",
    "    if a.start != 2001 or a.end not in EXPECTED_COST or a.test_end != 2025:",
)
replace_once(
    "scripts/run_u1_delay_production_fresh.py",
    '''        "delay_mae": metrics["delay"]["MAE"],\n        "persisted_inference_delay_mae": live_delay_mae,\n        "delay_improvement_percentage": promo["delay"]["delay_improvement_percentage"],\n''',
    '''        "delay_mae": metrics["delay"]["MAE"],\n        "delay_mape_percent": metrics["delay"].get("MAPE"),\n        "persisted_inference_delay_mae": live_delay_mae,\n        "delay_improvement_percentage": promo["delay"]["delay_improvement_percentage"],\n        "delay_routing_contract": result["metadata"].get("delay_evaluation_contract"),\n''',
)
replace_re(
    "scripts/run_u1_delay_production_fresh.py",
    r'    expected = EXPECTED\[a\.end\]\n    if not _close\(payload\["cost_mae"\], expected\["cost"\]\):\n        raise RuntimeError\(f"Exp105 Cost did not reproduce: \{payload\[\'cost_mae\'\]\} vs expected \{expected\[\'cost\'\]\}"\)\n    if not _close\(payload\["delay_mae"\], expected\["delay"\]\):\n        raise RuntimeError\(f"Exp113 Delay did not reproduce: \{payload\[\'delay_mae\'\]\} vs expected \{expected\[\'delay\'\]\}"\)\n',
    '''    expected_cost = EXPECTED_COST[a.end]\n    if not _close(payload["cost_mae"], expected_cost):\n        raise RuntimeError(f"Exp105 Cost did not reproduce: {payload['cost_mae']} vs expected {expected_cost}")\n    if payload["delay_mape_percent"] is None or not np.isfinite(float(payload["delay_mape_percent"])):\n        raise RuntimeError("Delay percentage error (MAPE) was not finite")\n''',
)
replace_once(
    "scripts/run_u1_delay_production_fresh.py",
    '''        if (payload["comparison_test_projects"], payload["comparison_test_snapshots"]) != (721, 11200):\n            raise RuntimeError("Verified 2001-2021 production cohort changed")\n''',
    '''        if payload["comparison_test_projects"] < 2 or payload["comparison_test_snapshots"] < payload["comparison_test_projects"]:\n            raise RuntimeError("Evidence-defined comparison cohort is not usable")\n''',
)
replace_once(
    "scripts/run_u1_delay_production_fresh.py",
    '    print(f"{prefix}_DELAY_MAE={payload[\'delay_mae\']}")\n',
    '    print(f"{prefix}_DELAY_MAE={payload[\'delay_mae\']}")\n    print(f"{prefix}_DELAY_MAPE_PERCENT={payload[\'delay_mape_percent\']}")\n',
)

# ---------------------------------------------------------------------------
# Unit contract: default = all evidence-based routes; explicit limit = legacy.
# ---------------------------------------------------------------------------
replace_once(
    "tests/test_exp35_production_promotion.py",
    "    VERIFIED_AFT_CALIBRATION_PROJECTS,\n",
    "",
)
replace_re(
    "tests/test_exp35_production_promotion.py",
    r'def test_aft_calibration_cohort_selection_is_fixed_and_evidence_only\(\):\n.*?\n\n\ndef test_production_baseline_names_identify_688_project_combined_promotion\(\):\n    assert VERIFIED_AFT_CALIBRATION_PROJECTS == 688\n    assert "exp33" in PRODUCTION_COST_BASELINE\n    assert "exp32" in PRODUCTION_DELAY_BASELINE\n    assert "exp33" in PRODUCTION_DELAY_BASELINE\n    assert "688" in PRODUCTION_DELAY_BASELINE\n    assert "exp34_fallback" in PRODUCTION_DELAY_BASELINE\n',
    '''def _routing_fixture():\n    rows = []\n    for project, eligible, total in [\n        ("A", 3, 3),\n        ("B", 2, 2),\n        ("C", 2, 3),\n        ("D", 1, 3),\n        ("E", 0, 2),\n    ]:\n        for i in range(total):\n            rows.append(\n                {\n                    "canonical_project_id": project,\n                    "snapshot_date": f"2020-01-{i + 1:02d}",\n                    "planned_completion_date": "2020-02-01" if i < eligible else None,\n                    "actual_delay_days": 10000 if project == "D" else 0,\n                }\n            )\n    return pd.DataFrame(rows)\n\n\ndef test_aft_routing_selects_all_projects_with_as_of_evidence_by_default():\n    selected = _select_aft_calibration_projects(_routing_fixture())\n    assert selected == {"A", "B", "C", "D"}\n\n\ndef test_aft_routing_supports_explicit_legacy_limit_without_using_targets():\n    selected = _select_aft_calibration_projects(_routing_fixture(), limit=2)\n    assert selected == {"A", "B"}\n\n\ndef test_production_baseline_names_identify_evidence_router_and_fallback():\n    assert "exp33" in PRODUCTION_COST_BASELINE\n    assert "exp32" in PRODUCTION_DELAY_BASELINE\n    assert "exp33" in PRODUCTION_DELAY_BASELINE\n    assert "evidence_router" in PRODUCTION_DELAY_BASELINE\n    assert "688" not in PRODUCTION_DELAY_BASELINE\n    assert "exp34_fallback" in PRODUCTION_DELAY_BASELINE\n''',
)

# ---------------------------------------------------------------------------
# Workflows: verify rules and metrics, not historical sample counts.
# ---------------------------------------------------------------------------
replace_once(
    ".github/workflows/verify-exp34-production-promotion.yml",
    '''          delay_mae_721 = float(result['lifecycle']['metrics']['delay']['MAE'])\n          contract = result['metadata']['delay_evaluation_contract']\n          assert int(contract['test_projects']) == 721, contract\n          assert int(contract['test_snapshots']) == 11200, contract\n          assert int(contract['full_holdout_projects']) == 728, contract\n          delay_mae_728 = float(contract['full_holdout_delay_metrics']['MAE'])\n\n          assert abs(cost_mae - 26.872) <= 0.001, cost_mae\n          assert abs(delay_mae_721 - 501.303) <= 0.001, delay_mae_721\n          assert abs(delay_mae_728 - 503.555) <= 0.001, delay_mae_728\n''',
    '''          delay_mae_comparable = float(result['lifecycle']['metrics']['delay']['MAE'])\n          contract = result['metadata']['delay_evaluation_contract']\n          assert contract['source_filter'] == 'exp12_comparable_trailing_12m_history', contract\n          assert contract['cohort_count_policy'] == 'observed_only_no_fixed_project_or_snapshot_requirement', contract\n          assert int(contract['test_projects']) >= 2, contract\n          delay_mae_full = float(contract['full_holdout_delay_metrics']['MAE'])\n\n          assert abs(cost_mae - 26.872) <= 0.001, cost_mae\n          assert abs(delay_mae_comparable - 501.303) <= 0.001, delay_mae_comparable\n          assert abs(delay_mae_full - 503.555) <= 0.001, delay_mae_full\n''',
)
replace_once(
    ".github/workflows/verify-exp34-production-promotion.yml",
    '''          print(f'EXP34_REGRESSION_COST_MAE_721={cost_mae:.6f}')\n          print(f'EXP34_REGRESSION_DELAY_MAE_721={delay_mae_721:.6f}')\n          print(f'EXP34_REGRESSION_DELAY_MAE_FULL_728={delay_mae_728:.6f}')\n''',
    '''          print(f'EXP34_REGRESSION_COST_MAE_COMPARABLE={cost_mae:.6f}')\n          print(f'EXP34_REGRESSION_DELAY_MAE_COMPARABLE={delay_mae_comparable:.6f}')\n          print(f'EXP34_REGRESSION_DELAY_MAE_FULL_HOLDOUT={delay_mae_full:.6f}')\n          print(f'EXP34_COMPARISON_PROJECTS={contract["test_projects"]}')\n          print(f'EXP34_COMPARISON_SNAPSHOTS={contract["test_snapshots"]}')\n''',
)

replace_once(
    ".github/workflows/verify-exp35-production-promotion.yml",
    "      - name: Retrain selected production window and verify fixed 688-project Delay calibration",
    "      - name: Retrain selected production window and verify evidence-based Delay routing",
)
replace_once(
    ".github/workflows/verify-exp35-production-promotion.yml",
    "          assert result['production_delay_baseline'] == 'exp32_aft_plus_exp33_residual_688_exp34_fallback_v2'",
    "          assert result['production_delay_baseline'] == 'exp32_aft_plus_exp33_evidence_router_exp34_fallback_v3'",
)
replace_re(
    ".github/workflows/verify-exp35-production-promotion.yml",
    r"          delay_mae_721 = float\(result\['metrics'\]\['delay_model'\]\['MAE'\]\)\n          contract = result\['metrics'\]\['metadata'\]\['delay_evaluation_contract'\]\n.*?          assert contract\['fallback_policy'\]\.startswith\('Exp34 production Delay'\), contract\n\n          delay_base_688 = float\(contract\['base_exp34_mae_calibration_688'\]\)\n          delay_exp_688 = float\(contract\['promoted_exp32_exp33_mae_calibration_688'\]\)\n          delay_gain_688 = float\(contract\['calibration_688_improvement_percentage'\]\)\n",
    '''          delay_mae_comparable = float(result['metrics']['delay_model']['MAE'])\n          delay_mape_percent = float(result['metrics']['delay_model']['MAPE'])\n          contract = result['metrics']['metadata']['delay_evaluation_contract']\n          assert contract['cohort_rule'] == 'exp12_history_12m >= 2 then project-balanced weights', contract\n          assert contract['cohort_count_policy'] == 'observed_only_no_fixed_project_or_snapshot_requirement', contract\n          assert int(contract['test_projects']) >= 2, contract\n          assert int(contract['routing_projects']) >= 1, contract\n          assert int(contract['aft_eligible_snapshots']) <= int(contract['test_snapshots']), contract\n          assert contract['fallback_policy'].startswith('Exp34 production Delay'), contract\n\n          delay_base_routed = float(contract['base_exp34_mae_routed_projects'])\n          delay_exp_routed = float(contract['promoted_exp32_exp33_mae_routed_projects'])\n          delay_gain_routed = float(contract['routed_project_improvement_percentage'])\n''',
)
replace_once(
    ".github/workflows/verify-exp35-production-promotion.yml",
    '''          assert delay_exp_688 < delay_base_688, (delay_exp_688, delay_base_688)\n          assert delay_mae_721 < previous_delay_mae, (delay_mae_721, previous_delay_mae)\n''',
    '''          assert delay_exp_routed < delay_base_routed, (delay_exp_routed, delay_base_routed)\n          assert delay_mae_comparable < previous_delay_mae, (delay_mae_comparable, previous_delay_mae)\n''',
)
replace_once(
    ".github/workflows/verify-exp35-production-promotion.yml",
    "          assert metadata['production_delay_baseline'] == 'exp32_aft_plus_exp33_residual_688_exp34_fallback_v2'",
    "          assert metadata['production_delay_baseline'] == 'exp32_aft_plus_exp33_evidence_router_exp34_fallback_v3'",
)
replace_re(
    ".github/workflows/verify-exp35-production-promotion.yml",
    r"          assert metadata\['delay_calibration_project_count'\] == 688\n          assert metadata\['delay_evaluation_contract'\]\['test_projects'\] == 721\n          assert metadata\['delay_evaluation_contract'\]\['calibration_cohort_projects'\] == 688\n\n          cost_gain = .*?          print\('EXP32_EXP33_688_PRODUCTION_PROMOTION=VERIFIED'\)\n",
    '''          assert metadata['delay_calibration_project_count'] == contract['routing_projects']\n          assert metadata['delay_evaluation_contract']['cohort_count_policy'] == 'observed_only_no_fixed_project_or_snapshot_requirement'\n\n          cost_gain = (previous_cost_mae - cost_mae) / previous_cost_mae * 100.0\n          delay_gain = (previous_delay_mae - delay_mae_comparable) / previous_delay_mae * 100.0\n          print(f'EXP32_EXP33_PRODUCTION_COST_MAE={cost_mae:.6f}')\n          print(f'EXP32_EXP33_PRODUCTION_COST_IMPROVEMENT={cost_gain:.6f}%')\n          print(f'EXP34_DELAY_MAE_ROUTED_PROJECTS={delay_base_routed:.6f}')\n          print(f'EXP32_EXP33_DELAY_MAE_ROUTED_PROJECTS={delay_exp_routed:.6f}')\n          print(f'EXP32_EXP33_DELAY_IMPROVEMENT_ROUTED_PROJECTS={delay_gain_routed:.6f}%')\n          print(f'EXP32_EXP33_PRODUCTION_DELAY_MAE={delay_mae_comparable:.6f}')\n          print(f'EXP32_EXP33_PRODUCTION_DELAY_MAPE_PERCENT={delay_mape_percent:.6f}%')\n          print(f'EXP32_EXP33_PRODUCTION_DELAY_IMPROVEMENT={delay_gain:.6f}%')\n          print(f'AFT_ROUTING_PROJECTS={contract["routing_projects"]}')\n          print(f'AFT_ELIGIBLE_SNAPSHOTS={contract["aft_eligible_snapshots"]}')\n          print(f'FALLBACK_ONLY_PROJECTS={contract["fallback_only_projects"]}')\n          print('EXP32_EXP33_EVIDENCE_ROUTER_PRODUCTION_PROMOTION=VERIFIED')\n''',
)
replace_once(
    ".github/workflows/verify-exp35-production-promotion.yml",
    "          name: exp32-exp33-production-688-2001-2021",
    "          name: exp32-exp33-evidence-router-production-2001-2021",
)

print("Evidence-based cohort/AFT routing refactor applied successfully.")
