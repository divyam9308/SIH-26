from backend.app.services.training_window_performance_service import training_window_performance


def test_training_window_performance_uses_one_common_future_cohort():
    payload = training_window_performance()

    assert payload["evaluation_period"] == "2023-2024 completed projects"
    assert payload["sample_count"] == 39
    assert [item["end_year"] for item in payload["windows"]] == [2017, 2021, 2022]
    assert all(item["sample_count"] == payload["sample_count"] for item in payload["windows"])
    assert payload["windows"][0]["cost_mae"] == 27.758
    assert payload["windows"][2]["delay_mae_days"] == 772.781
