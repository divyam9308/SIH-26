from pathlib import Path
import json

from backend.app.ml.production_exp105_exp113_baseline import train_window_with_promoted_cost_and_delay


def main() -> None:
    root = Path("test-output/cost-r2-2001-2021")
    root.mkdir(parents=True, exist_ok=True)
    result = train_window_with_promoted_cost_and_delay(
        2001,
        2021,
        2025,
        artifact_root=root,
        verify_frozen_reference=True,
    )
    (root / "experiment_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
