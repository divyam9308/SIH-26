from pathlib import Path


def test_risk_metrics_experiment_is_scoped_to_verified_window():
    text = Path("scripts/run_risk_metrics_2001_2021.py").read_text()
    assert "2001" in text
    assert "2021" in text
    assert "2025" in text
    assert "verify_frozen_reference=True" in text
