import argparse
from backend.app.ml.experiments.exp132_scale_regime_cost import fit_experiment
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2021],default=2021);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
