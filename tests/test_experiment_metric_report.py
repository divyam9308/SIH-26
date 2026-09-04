from scripts.experiment_metric_report import build_report


def payload(cost_r2=.389, cost_mae=25.923, cost_rmse=40.0, delay_r2=.826, delay_mae=342.35, delay_rmse=500.0, f1=.407, precision=.398, recall=.420):
    return {
        "metadata": {"training_period": [2001, 2021], "testing_period": [2022, 2025], "dataset_fingerprint": "same"},
        "lifecycle": {"metrics": {
            "cost": {"R2": cost_r2, "MAE": cost_mae, "RMSE": cost_rmse},
            "delay": {"R2": delay_r2, "MAE": delay_mae, "RMSE": delay_rmse},
            "risk": {"macro_f1": f1, "macro_precision": precision, "macro_recall": recall},
        }},
    }


def test_cost_report_accepts_real_r2_improvement_without_material_mae_regression():
    report = build_report(baseline=payload(), candidate=payload(cost_r2=.42, cost_mae=26.0), target="cost_r2")
    assert report["decision"] == "ACCEPT"
    assert report["delta_candidate_minus_baseline"]["cost_r2"] > 0


def test_cost_report_rejects_no_change():
    assert build_report(baseline=payload(), candidate=payload(), target="cost_r2")["decision"] == "REJECT"
