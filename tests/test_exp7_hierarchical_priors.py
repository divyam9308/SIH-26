import numpy as np
import pandas as pd

from backend.app.ml.experiments.hierarchical_residual_priors_exp7 import (
    AGENCY_K,
    SECTOR_K,
    _corrections,
    _group_prior,
    _project_level_residuals,
)


def test_exp7_uses_one_residual_contribution_per_project():
    frame = pd.DataFrame(
        {
            "canonical_project_id": ["A", "A", "A", "B"],
            "implementing_agency": ["Agency 1", "Agency 1", "Agency 1", "Agency 1"],
            "sector": ["Road", "Road", "Road", "Road"],
            "actual_cost_overrun_percentage": [10.0, 20.0, 30.0, 40.0],
        }
    )
    predictions = np.array([0.0, 0.0, 0.0, 0.0])
    projects = _project_level_residuals(frame, predictions)
    assert len(projects) == 2
    assert projects.set_index("canonical_project_id").loc["A", "residual"] == 20.0
    assert projects.set_index("canonical_project_id").loc["B", "residual"] == 40.0


def test_exp7_unseen_groups_receive_zero_correction():
    project_residuals = pd.DataFrame(
        {
            "canonical_project_id": ["A", "B"],
            "implementing_agency": ["Agency 1", "Agency 1"],
            "sector": ["Road", "Road"],
            "residual": [10.0, 20.0],
        }
    )
    agency = _group_prior(project_residuals, "implementing_agency", AGENCY_K)
    sector = _group_prior(project_residuals, "sector", SECTOR_K)
    unseen = pd.DataFrame({"implementing_agency": ["New Agency"], "sector": ["New Sector"]})
    correction = _corrections(unseen, agency, sector)
    assert correction.tolist() == [0.0]
