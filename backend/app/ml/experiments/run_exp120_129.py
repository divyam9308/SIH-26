"""Run one post-Exp113 Delay experiment on one verified window."""
import argparse,importlib

def main():
    p=argparse.ArgumentParser();p.add_argument('--exp',type=int,choices=range(120,130),required=True);p.add_argument('--end',type=int,choices=[2021,2022],required=True);p.add_argument('--output',required=True);a=p.parse_args();m=importlib.import_module(f'backend.app.ml.experiments.exp{a.exp}_delay');m.fit_experiment(a.end,a.output)
if __name__=='__main__':main()
