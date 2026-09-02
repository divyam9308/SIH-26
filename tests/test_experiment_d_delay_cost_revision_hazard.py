import pandas as pd

from backend.app.ml.experiments.experiment_d_delay_cost_revision_hazard import revision_labels


def test_revision_labels_respect_horizons_and_use_future_revision_events_only():
    frame = pd.DataFrame(
        {
            "canonical_project_id": ["P1"] * 4,
            "snapshot_date": pd.to_datetime(["2020-01-01", "2020-03-01", "2020-08-01", "2021-02-01"]),
            "revised_cost_cr": [100.0, 100.0, 120.0, 120.0],
        }
    )
    labelled = revision_labels(frame)
    # From Jan to the Aug revision is >3m but <12m.
    assert labelled.loc[0, "_label_3m"] == 0
    assert labelled.loc[0, "_label_12m"] == 1
    # From March to August is within 6m.
    assert labelled.loc[1, "_label_6m"] == 1
    # Once the revision has happened, there is no later event.
    assert labelled.loc[2, "_label_12m"] == 0


def test_delay_target_is_not_used_to_construct_revision_labels():
    frame = pd.DataFrame(
        {
            "canonical_project_id": ["P1"] * 3,
            "snapshot_date": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01"]),
            "revised_cost_cr": [100.0, 130.0, 130.0],
            "actual_delay_days": [10, 20, 30],
        }
    )
    a = revision_labels(frame)
    frame["actual_delay_days"] = [9999, -9999, 123456]
    b = revision_labels(frame)
    assert a[["_label_3m", "_label_6m", "_label_12m"]].equals(b[["_label_3m", "_label_6m", "_label_12m"]])
