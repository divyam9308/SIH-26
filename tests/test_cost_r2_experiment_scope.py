from pathlib import Path


def test_cost_r2_experiment_is_scoped_to_verified_window():
    text = Path("scripts/run_cost_r2_2001_2021.py").read_text()
    assert "2001" in text and "2021" in text and "2025" in text
    assert "verify_frozen_reference=True" in text
    assert "/api/models/retrain" not in text
