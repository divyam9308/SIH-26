from backend.app.services.training_window_performance_service import training_window_performance


def test_training_window_performance_uses_latest_available_production_evaluations():
    payload = training_window_performance()

    assert payload["evaluation_period"] == "Each saved production artifact's latest temporal holdout"
    assert [item["end_year"] for item in payload["windows"]] == [2017, 2021, 2022]
    assert payload["windows"][0]["cost_mae"] == 32.886
    assert payload["windows"][1]["source"] == "canonical_monthly_lifecycle"
    assert payload["windows"][2]["delay_mae_days"] == 294.287
