"""
XGBoost training script for Jane Street Market Prediction.
Converted from market-prediction-xgboost-with-gpu-fit-in-1min.ipynb.

Artifacts saved to:  ./tmp/xgboost/
  - xgb_fold{n}.json      model checkpoints
  - feature_mean.npy      per-feature means for inference NaN-fill
  - feature_cols.csv      ordered feature column names
"""

import warnings
warnings.filterwarnings('ignore')

import os
import gc
import time
import subprocess

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
import xgboost as xgb

# ============================================================
# Config
# ============================================================
TMP_DIR   = './tmp/xgboost'
DATA_PATH = './data/train.csv'
NUM_FOLDS = 5

os.makedirs(TMP_DIR, exist_ok=True)

XGB_BASE_PARAMS = dict(
    n_estimators     = 400,
    max_depth        = 7,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 50,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    eval_metric      = 'logloss',
)


# ============================================================
# GPU detection
# ============================================================
def get_device_params() -> dict:
    """Return XGBoost device config based on available hardware."""
    try:
        result = subprocess.run(
            ['nvidia-smi'], capture_output=True, text=True, timeout=10
        )
        has_gpu = result.returncode == 0
    except Exception:
        has_gpu = False

    if not has_gpu:
        print('No GPU detected — training on CPU.')
        return {'tree_method': 'hist'}

    major = int(xgb.__version__.split('.')[0])
    if major >= 2:
        print(f'XGBoost {xgb.__version__} + GPU  (device=cuda, tree_method=hist)')
        return {'device': 'cuda', 'tree_method': 'hist'}
    else:
        print(f'XGBoost {xgb.__version__} + GPU  (tree_method=gpu_hist)')
        return {'tree_method': 'gpu_hist'}


# ============================================================
# Utility Score (Jane Street competition metric)
# ============================================================
def utility_score(date: np.ndarray, weight: np.ndarray,
                  resp: np.ndarray, action: np.ndarray) -> float:
    date = date.astype(int)
    count_i = len(np.unique(date))
    Pi = np.bincount(date, weight * resp * action)
    t  = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    return float(np.clip(t, 0, 6) * np.sum(Pi))


# ============================================================
# Load & preprocess
# ============================================================
print('Loading data...')
train = pd.read_csv(DATA_PATH)
print(f'Raw shape: {train.shape}')

# Keep only positive-weight rows
train = train[train['weight'] != 0].reset_index(drop=True)

# Binary target
train['action'] = (train['resp'] > 0).astype('int')

# Feature columns
feature_cols = [c for c in train.columns if c.startswith('feature')]

# Fill NaN with per-feature mean (dataset-level, no lookahead for OOF)
print('Filling missing values with feature means...')
f_mean = train[feature_cols].mean()
train[feature_cols] = train[feature_cols].fillna(f_mean)

# Persist fill values for inference
np.save(os.path.join(TMP_DIR, 'feature_mean.npy'), f_mean.values)
pd.Series(feature_cols).to_csv(
    os.path.join(TMP_DIR, 'feature_cols.csv'), index=False, header=False
)

X      = train[feature_cols].values
y      = train['action'].values
dates  = train['date'].values
weights = train['weight'].values
resps   = train['resp'].values

print(f'Dataset: {X.shape[0]:,} rows x {X.shape[1]} features')
print(f'Positive rate: {y.mean():.4f}\n')

# ============================================================
# Cross-validated training
# ============================================================
device_params = get_device_params()
params        = {**XGB_BASE_PARAMS, **device_params}

gkf      = GroupKFold(n_splits=NUM_FOLDS)
oof      = np.zeros(len(train))
fold_auc = []
start    = time.time()

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, dates)):
    print(f'\n{"="*60}')
    print(f' FOLD {fold}  (train={len(tr_idx):,}, valid={len(va_idx):,})')
    print(f'{"="*60}')

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    clf = xgb.XGBClassifier(**params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=100,
    )

    va_pred      = clf.predict_proba(X_va)[:, 1]
    oof[va_idx]  = va_pred
    va_binary    = (va_pred >= 0.5).astype(int)

    auc = roc_auc_score(y_va, va_pred)
    ll  = log_loss(y_va, va_pred)
    u   = utility_score(dates[va_idx], weights[va_idx], resps[va_idx], va_binary)

    print(f'  >> AUC={auc:.5f}  LogLoss={ll:.5f}  Utility={u:.2f}')
    fold_auc.append(auc)

    model_path = os.path.join(TMP_DIR, f'xgb_fold{fold}.json')
    clf.save_model(model_path)
    print(f'  Model saved: {model_path}')

    del clf, X_tr, y_tr, X_va, y_va
    gc.collect()

# ============================================================
# Summary
# ============================================================
print(f'\n{"="*60}')
print(f' SUMMARY')
print(f'{"="*60}')
print(f'Fold AUCs  : {[f"{x:.5f}" for x in fold_auc]}')
print(f'Mean AUC   : {np.mean(fold_auc):.5f} ± {np.std(fold_auc):.5f}')
print(f'OOF AUC    : {roc_auc_score(y, oof):.5f}')
print(f'Total time : {(time.time() - start) / 60:.2f} min')
