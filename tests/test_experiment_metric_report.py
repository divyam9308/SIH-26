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


def test_risk_report_requires_f1_precision_and_recall_all_to_improve():
    report = build_report(baseline=payload(), candidate=payload(f1=.43, precision=.41, recall=.44), target="risk_metrics")
    assert report["decision"] == "ACCEPT"


def test_risk_report_rejects_precision_recall_tradeoff_when_all_three_do_not_improve():
    report = build_report(baseline=payload(), candidate=payload(f1=.43, precision=.39, recall=.45), target="risk_metrics")
    assert report["decision"] == "REJECT"
