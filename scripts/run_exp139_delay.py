import argparse

from backend.app.ml.experiments.exp139_sector_stratified_delay import fit_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=int, choices=[2021, 2022], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fit_experiment(args.end, args.output)
