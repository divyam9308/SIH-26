from backend.app.services import training_window_performance_service


def test_training_window_performance_uses_shared_2023_2025_comparison():
    payload = training_window_performance_service.training_window_performance()

    assert [item["end_year"] for item in payload["windows"]] == [2017, 2021, 2022]
    assert [item["cost_mae"] for item in payload["windows"]] == [42.160, 24.266, 23.750]
    assert [item["delay_mae_days"] for item in payload["windows"]] == [494.779, 343.592, 294.287]
    assert [item["cost_r2"] for item in payload["windows"]] == [0.5301, 0.7716, 0.8256]
    assert all(item["evaluation_period"] == "2023–2025" for item in payload["windows"])
    assert all(item["source"] == "provided_shared_holdout_comparison" for item in payload["windows"])
