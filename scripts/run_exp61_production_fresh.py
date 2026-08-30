"""Freshly train promoted Exp61 production and print Cost/Delay MAE for one window."""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import pandas as pd
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.production_exp61_baseline import train_window_with_promoted_cost_and_delay


def main():
    p=argparse.ArgumentParser(); p.add_argument('--start',type=int,required=True); p.add_argument('--end',type=int,required=True); p.add_argument('--test-end',type=int,default=2025); p.add_argument('--output',required=True); a=p.parse_args()
    data,identity=build_training_dataset(); data=data.copy(); data['completion_year']=pd.to_numeric(data['completion_year'],errors='coerce')
    with tempfile.TemporaryDirectory(prefix=f'exp61-production-{a.end}-') as td:
        result=train_window_with_promoted_cost_and_delay(a.start,a.end,a.test_end,data=data,identity=identity,artifact_root=Path(td))
    metrics=result['lifecycle']['metrics']; promo=result['promotion']
    payload={'window':f'{a.start}_{a.end}','test_end':a.test_end,'cost_mae':metrics['cost']['MAE'],'delay_mae':metrics['delay']['MAE'],'promotion':promo,'production_cost_baseline':result['metadata']['production_cost_baseline'],'production_delay_baseline':result['metadata']['production_delay_baseline']}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    prefix=f'EXP61_PRODUCTION_{a.start}_{a.end}'
    print(f'{prefix}_COST_MAE={payload["cost_mae"]}')
    print(f'{prefix}_DELAY_MAE={payload["delay_mae"]}')
    print(f'{prefix}_PREVIOUS_COST_MAE={promo["previous_cost_mae"]}')
    print(f'{prefix}_PREVIOUS_DELAY_MAE={promo["previous_delay_mae"]}')
    print(f'{prefix}_COST_IMPROVEMENT_PERCENT={promo["cost_improvement_percentage"]}')
    print(f'{prefix}_DELAY_IMPROVEMENT_PERCENT={promo["delay_improvement_percentage"]}')

if __name__=='__main__': main()
