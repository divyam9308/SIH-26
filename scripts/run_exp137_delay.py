import argparse

from backend.app.ml.experiments.exp137_delay_asymmetric_cap import fit_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=int, choices=[2022], default=2022)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fit_experiment(args.end, args.output)
