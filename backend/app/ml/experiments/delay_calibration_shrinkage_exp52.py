"""Experiment 52 / D4: fold-stable shrunk AFT Delay residual calibration."""
import sys
from backend.app.ml.experiments.nextgen_common import fit_delay_calibration,run_cli
EXPERIMENT_ID="exp_52";EXPERIMENT_SEQUENCE=52;MARKER="EXP52";EXPERIMENT_NAME="Fold-stable shrunk AFT Delay calibration";EXPERIMENT_SCOPE="delay";CHANGED_DIMENSION="fold_stable_shrunk_aft_calibration"
def fit_experiment(**kwargs): return fit_delay_calibration(exp_id=EXPERIMENT_ID,name=EXPERIMENT_NAME,strength=40.0,**kwargs)
if __name__=="__main__":run_cli(sys.modules[__name__])
