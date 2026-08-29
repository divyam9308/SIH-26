"""Experiment 59 / C6+D9: low-dimensional trajectory path scores."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from backend.app.ml.experiments.nextgen_common import fit_features,run_cli
EXPERIMENT_ID="exp_59";EXPERIMENT_SEQUENCE=59;MARKER="EXP59";EXPERIMENT_NAME="Fold-safe PCA trajectory path scores";EXPERIMENT_SCOPE="cost+delay";CHANGED_DIMENSION="training_only_pca_path_shape_features"
FEATURES=["exp59_path_pc1","exp59_path_pc2","exp59_path_pc3"]
CANDIDATES=["exp12_cost_velocity_3m","exp12_cost_velocity_6m","exp12_cost_velocity_12m","exp12_expenditure_velocity_3m","exp12_expenditure_velocity_6m","exp12_expenditure_velocity_12m","exp12_slippage_velocity_3m","exp12_slippage_velocity_6m","exp12_slippage_velocity_12m","exp34_cost_revision_count","exp34_schedule_revision_count","exp34_cost_worsening_share","exp34_delay_worsening_share"]
def engineer_pca(frame:pd.DataFrame)->pd.DataFrame:
 out=frame.copy();cols=[c for c in CANDIDATES if c in out.columns]
 if len(cols)<3:raise ValueError("Exp59 requires at least three available trajectory columns")
 trainmask=pd.to_numeric(out["completion_year"],errors="coerce").between(2001,2021,inclusive="both");trainx=out.loc[trainmask,cols].apply(pd.to_numeric,errors="coerce");med=trainx.median().fillna(0.0);allx=out[cols].apply(pd.to_numeric,errors="coerce").fillna(med);scaler=StandardScaler();ztrain=scaler.fit_transform(trainx.fillna(med));pca=PCA(n_components=3,random_state=59059);pca.fit(ztrain);scores=pca.transform(scaler.transform(allx))
 for i,f in enumerate(FEATURES):out[f]=scores[:,i]
 out.attrs["exp59_explained_variance"]=pca.explained_variance_ratio_.tolist();return out
def fit_experiment(**kwargs):return fit_features(exp_id=EXPERIMENT_ID,name=EXPERIMENT_NAME,dimension=CHANGED_DIMENSION,scope="cost+delay",engineer=engineer_pca,cost_new=FEATURES,delay_new=FEATURES,details={"pca_fit_window":"completion_year 2001-2021 only","candidate_columns":CANDIDATES},**kwargs)
if __name__=="__main__":run_cli(sys.modules[__name__])
