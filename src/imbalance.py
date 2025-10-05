# src/imbalance.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
try:
    from imblearn.combine import SMOTEENN
    HAS_SMOTEENN = True
except Exception:
    HAS_SMOTEENN = False

ROOT = Path(__file__).resolve().parents[1]

def load_splits():
    train = pd.read_csv(ROOT / "data/processed/train.csv", parse_dates=["event_time"])
    valid = pd.read_csv(ROOT / "data/processed/valid.csv", parse_dates=["event_time"])
    test  = pd.read_csv(ROOT / "data/processed/test.csv",  parse_dates=["event_time"])
    # 只取数值特征，避免把 datetime 放进模型（否则会 DType 错误）
    feature_cols = ["Amount"] + [c for c in train.columns if c.startswith("V")]
    X_tr, y_tr = train[feature_cols], train["Class"].values
    X_va, y_va = valid[feature_cols], valid["Class"].values
    X_te, y_te = test[feature_cols],  test["Class"].values
    return (X_tr,y_tr), (X_va,y_va), (X_te,y_te), feature_cols

def make_lr(class_weight=None):
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=3000, class_weight=class_weight))
    ])

def ap_cv(pipe, X, y, folds=5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    # 使用 cross_val_predict 严格在 CV 内产生 out-of-fold 概率
    proba_oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:,1]
    return average_precision_score(y, proba_oof), proba_oof #average precision, predictions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    (X_tr,y_tr), (X_va,y_va), (X_te,y_te), feats = load_splits()

    strategies = {}

    # 1) 无采样（仅 class_weight=None）
    strategies["no_sampling"] = make_lr(class_weight=None)

    # 2) class_weight=balanced
    strategies["class_weight"] = make_lr(class_weight="balanced")

    # 3) RandomUnderSampler（CV 内）
    strategies["undersample"] = ImbPipeline(steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("rus", RandomUnderSampler(random_state=42)),
        ("lr", LogisticRegression(max_iter=3000))
    ])

    # 4) SMOTE（CV 内）
    strategies["smote"] = ImbPipeline(steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("smote", SMOTE(sampling_strategy=0.1, random_state=42)),
        ("lr", LogisticRegression(max_iter=3000))
    ])

    # 5) SMOTEENN（可选）
    if HAS_SMOTEENN:
        strategies["smoteenn"] = ImbPipeline(steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("smoteenn", SMOTEENN(random_state=42)),
            ("lr", LogisticRegression(max_iter=3000))
        ])

    rows = []
    for name, pipe in strategies.items():
        ap_cv_mean, oof = ap_cv(pipe, X_tr, y_tr, folds=args.folds)
        rows.append({"strategy": name, "cv_ap": ap_cv_mean})

    df = pd.DataFrame(rows).sort_values("cv_ap", ascending=False)
    out = ROOT / "reports" / "imbalance_cv.csv"
    df.to_csv(out, index=False)
    print(df)
    print(f"[OK] saved {out}")

if __name__ == "__main__":
    main()
