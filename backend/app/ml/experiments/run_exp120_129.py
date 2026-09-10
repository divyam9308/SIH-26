"""Run one post-Exp113 Delay experiment on one verified window."""
import argparse, importlib
from pathlib import Path

from backend.app.ml.experiments import post_exp113_delay_common as common
from backend.app.ml.production_exp105_exp113_fast import (
    train_window_with_promoted_cost_and_delay as fast_train_current_production,
)


def main():
    # Execution-only substitution: use the performance-preserving wrapper for
    # every production fit, including strict-forward OOF folds. The wrapper
    # delegates to the exact canonical Exp105+Exp113 model logic and changes
    # only parallelism/caching, so the scientific comparison contract is unchanged.
    common.train_current_production = fast_train_current_production

    p = argparse.ArgumentParser()
    p.add_argument('--exp', type=int, choices=range(120, 130), required=True)
    p.add_argument('--end', type=int, choices=[2021, 2022])
    p.add_argument('--output', required=True)
    p.add_argument('--oof-year', type=int)
    p.add_argument('--oof-dir', type=Path)
    a = p.parse_args()
    m = importlib.import_module(f'backend.app.ml.experiments.exp{a.exp}_delay')

    if a.oof_year is not None and a.oof_dir is not None:
        p.error('--oof-year and --oof-dir are mutually exclusive')

    # Exp126 supports execution-only CI sharding of its exact six strict-forward
    # production OOF folds. Other Exp120-129 runners retain their prior behavior.
    if a.oof_year is not None:
        if a.exp != 126 or not hasattr(m, 'build_oof_fold'):
            p.error('--oof-year is supported only by Exp126 on this branch')
        m.build_oof_fold(a.oof_year, a.output)
        return

    if a.end is None:
        p.error('--end is required for the final experiment comparison')

    if a.oof_dir is not None:
        if a.exp != 126 or not hasattr(m, 'load_oof_dir'):
            p.error('--oof-dir is supported only by Exp126 on this branch')
        oof = m.load_oof_dir(a.oof_dir)
        m.fit_experiment(a.end, a.output, precomputed_oof=oof)
        return

    m.fit_experiment(a.end, a.output)


if __name__ == '__main__':
    main()
