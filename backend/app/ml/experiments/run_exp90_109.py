"""Generic runner for exactly one Exp90-109 adapter present on a challenger branch."""
from __future__ import annotations
import argparse,importlib,importlib.util

def _adapter():
    found=[]
    for seq in range(90,110):
        name=f'backend.app.ml.experiments.adapter_exp{seq}'
        if importlib.util.find_spec(name) is not None: found.append(importlib.import_module(name))
    if len(found)!=1: raise RuntimeError(f'Expected exactly one Exp90-109 adapter, found {[getattr(x,"EXPERIMENT_SEQUENCE",None) for x in found]}')
    return found[0]

def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();m=_adapter();print(f'RUNNING_EXP={m.EXPERIMENT_SEQUENCE}');m.fit_experiment(a.end,a.output)
if __name__=='__main__': main()
