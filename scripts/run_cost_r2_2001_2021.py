from pathlib import Path
import json

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.production_exp105_exp113_baseline import train_window_with_promoted_cost_and_delay
from scripts.experiment_metric_report import build_report, load_frozen_baseline, write_report


def main() -> None:
    root = Path("test-output/cost-r2-2001-2021")
    root.mkdir(parents=True, exist_ok=True)

    # Load the immutable production baseline before training the candidate.
    baseline = load_frozen_baseline()
    data, identity = build_training_dataset()
    candidate = train_window_with_promoted_cost_and_delay(
        2001,
        2021,
        2025,
        data=data,
        identity=identity,
        artifact_root=root,
        verify_frozen_reference=True,
    )
    (root / "candidate_result.json").write_text(json.dumps(candidate, indent=2, allow_nan=False))
    report = build_report(baseline=baseline, candidate=candidate, target="cost_r2")
    write_report(report, root / "experiment_result.json")


if __name__ == "__main__":
    main()
