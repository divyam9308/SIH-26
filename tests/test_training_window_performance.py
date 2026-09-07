from backend.app.services import training_window_performance_service


def test_training_window_performance_uses_shared_2023_2025_comparison():
    payload = training_window_performance_service.training_window_performance()

    assert [item["end_year"] for item in payload["windows"]] == [2017, 2021, 2022]
    assert [item["cost_mae"] for item in payload["windows"]] == [45.472, 25.829, 24.257]
    assert [item["delay_mae_days"] for item in payload["windows"]] == [472.326, 345.511, 294.412]
    assert all(item["evaluation_period"] == "2023–2025" for item in payload["windows"])
    assert all(item["source"] == "provided_shared_holdout_comparison" for item in payload["windows"])
