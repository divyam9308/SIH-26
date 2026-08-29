"""Experiment 58 / C7+D8: normalized taxonomy + strict temporal hierarchical priors."""
import sys
from backend.app.ml.experiments.nextgen_common import fit_priors,run_cli
EXPERIMENT_ID="exp_58";EXPERIMENT_SEQUENCE=58;MARKER="EXP58";EXPERIMENT_NAME="Normalized taxonomy and hierarchical Cost/Delay priors";EXPERIMENT_SCOPE="cost+delay";CHANGED_DIMENSION="normalized_taxonomy_temporal_hierarchical_priors"
def fit_experiment(**kwargs):return fit_priors(exp_id=EXPERIMENT_ID,name=EXPERIMENT_NAME,**kwargs)
if __name__=="__main__":run_cli(sys.modules[__name__])
