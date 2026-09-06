"""Run one post-Exp113 Delay experiment on one verified window."""
import argparse, importlib

from backend.app.ml.experiments import post_exp113_delay_common as common
from backend.app.ml.production_exp105_exp113_fast import (
    train_window_with_promoted_cost_and_delay as fast_train_current_production,
)


def main():
    # Execution-only substitution: use the performance-preserving wrapper for
    # every production fit, including the six strict-forward OOF folds. The
    # wrapper delegates to the exact canonical Exp105+Exp113 model logic and
    # changes only parallelism/caching, so the scientific comparison contract
    # (features, folds, cohorts, targets and promotion rules) is unchanged.
    common.train_current_production = fast_train_current_production

    p = argparse.ArgumentParser()
    p.add_argument('--exp', type=int, choices=range(120, 130), required=True)
    p.add_argument('--end', type=int, choices=[2021, 2022], required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    m = importlib.import_module(f'backend.app.ml.experiments.exp{a.exp}_delay')
    m.fit_experiment(a.end, a.output)


if __name__ == '__main__':
    main()
