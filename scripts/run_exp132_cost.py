from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.ml.experiments.exp132_scale_regime_cost import (
    OOF_YEARS,
    build_oof_fold,
    fit_experiment,
    load_oof_dir,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=int, choices=[2021], default=2021)
    parser.add_argument("--output")
    parser.add_argument("--oof-year", type=int, choices=OOF_YEARS)
    parser.add_argument("--oof-dir", type=Path)
    args = parser.parse_args()

    if args.oof_year is not None and args.oof_dir is not None:
        parser.error("--oof-year and --oof-dir are mutually exclusive")

    if args.oof_year is not None:
        output = args.output or f"audit_oof/cost-oof-{args.oof_year}.pkl"
        build_oof_fold(args.oof_year, output)
    else:
        if not args.output:
            parser.error("--output is required for the final Exp132 comparison")
        oof = load_oof_dir(args.oof_dir) if args.oof_dir is not None else None
        fit_experiment(args.end, args.output, precomputed_oof=oof)
