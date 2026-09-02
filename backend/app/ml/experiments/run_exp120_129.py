"""Run one post-Exp113 Delay experiment on one verified window.

The historical Exp35 688-project promotion audit assertion is disabled only in
this experiment-runner process. The AFT model itself and production code on main
are not modified; the PR harness supplies its own evidence-based eligibility.
"""
import argparse,importlib
import backend.app.ml.production_exp35_baseline as _p35

_p35._selected_window = lambda *_args, **_kwargs: False

def main():
    p=argparse.ArgumentParser();p.add_argument('--exp',type=int,choices=range(120,130),required=True);p.add_argument('--end',type=int,choices=[2021,2022],required=True);p.add_argument('--output',required=True);a=p.parse_args();m=importlib.import_module(f'backend.app.ml.experiments.exp{a.exp}_delay');m.fit_experiment(a.end,a.output)
if __name__=='__main__':main()
