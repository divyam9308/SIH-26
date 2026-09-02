import numpy as np
import pandas as pd

from backend.app.ml.experiments.experiment_f_cost_remaining_to_complete import (
    reconstruct_cost_overrun,
    remaining_cost_ratio_target,
)


def test_remaining_cost_target_is_money_still_required_normalized_by_approved_cost():
    frame = pd.DataFrame(
        {
            "approved_cost_cr": [100.0, 200.0],
            "cumulative_expenditure_cr": [40.0, 180.0],
            "reported_completion_expenditure_cr": [130.0, 170.0],
        }
    )
    target = remaining_cost_ratio_target(frame)
    assert np.isclose(target.iloc[0], 0.9)
    # Already-spent money is never undone by the remaining-spend target.
    assert np.isclose(target.iloc[1], 0.0)


def test_reconstruction_cannot_predict_final_cost_below_current_spend():
    score = pd.DataFrame(
        {
            "approved_cost_cr": [100.0, 100.0],
            "cumulative_expenditure_cr": [120.0, 40.0],
        }
    )
    prediction = reconstruct_cost_overrun(score, np.array([-1.0, 0.5]))
    # First row has already spent 20% above approval, so reconstructed overrun cannot fall below 20%.
    assert prediction[0] >= 20.0 - 1e-9
    assert np.isclose(prediction[1], -10.0)
