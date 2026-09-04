"""Run the existing Exp105/Exp113 frozen verifier through the fast execution wrapper."""
from __future__ import annotations

import time

from backend.app.ml.production_exp105_exp113_fast import train_window_with_promoted_cost_and_delay
from scripts import run_u1_delay_production_fresh as verifier


def main() -> None:
    verifier.train_window_with_promoted_cost_and_delay = train_window_with_promoted_cost_and_delay
    started = time.perf_counter()
    verifier.main()
    print(f"FAST_CANONICAL_TOTAL_SECONDS={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
