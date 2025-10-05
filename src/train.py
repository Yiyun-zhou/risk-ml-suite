from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# -----------------------------
# 基础路径
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Class"   

# -----------------------------
# 工具：KS 统计量
# -----------------------------
def ks_stat(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """KS = max |TPR - FPR|."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(np.abs(tpr - fpr)))

# -----------------------------
# 工具：特征列选择（适配 Kaggle creditcard）
# -----------------------------
def infer_feature_cols(df: pd.DataFrame, time_incl: bool = False, amount_incl: bool = True) -> list[str]:
    """
    Kaggle creditcard.csv: V1..V28 + Amount (+ 可选 Time)
    自动挑选所有以 'V' 开头的列，以及可选的 Amount / Time。
    """
    cols = []

    # V1...Vn
    v_cols = [c for c in df.columns if c.startswith("V")]
    cols.extend(sorted(v_cols, key=lambda x: (len(x), x)))  # 排序保证一致性

    # 加 Amount
    if amount_incl and "Amount" in df.columns:
        cols.insert(0, "Amount")

    # 加 Time
    if time_incl and "Time" in df.columns:   # 注意这里是 Time，不是 event_time
        cols.insert(0, "Time")

    # 去掉目标列（保险）
    cols = [c for c in cols if c != TARGET]

    # 确保全是数值列
    num_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    return num_cols


# -----------------------------
# 模型构建
# -----------------------------
def build_lr_pipeline() -> Pipeline:
    """
    逻辑回归：标准化 + class_weight='balanced'。
    没有类别特征，直接数值标准化。
    """
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)),
    ])
    return pipe

def try_build_xgb(scale_pos_weight:float):
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        )
        return model
    except Exception:
        return None

def try_build_lgbm():
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=400,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        return model
    except Exception:
        return None

# -----------------------------
# 数据读取
# -----------------------------
def load_processed():
    """
    读取 data/processed 下的 train/valid/test.csv。
    Kaggle 拆分文件通常没有 event_time，所以不做 parse_dates。
    """
    train = pd.read_csv(DATA_DIR / "train.csv")
    valid = pd.read_csv(DATA_DIR / "valid.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")

    # 选择特征列（以 train 为准）
    feature_cols = infer_feature_cols(train)

    X_tr, y_tr = train[feature_cols], train[TARGET].values
    X_va, y_va = valid[feature_cols], valid[TARGET].values
    X_te, y_te = test[feature_cols],  test[TARGET].values
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), feature_cols

# -----------------------------
# 主流程
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lr", "xgb", "lgbm"],
                    help="选择要训练的模型：lr xgb lgbm（可多选）")
    args = ap.parse_args()

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), feature_cols = load_processed()

    # 类别不均衡参数：给 XGB 用
    pos = float(np.sum(y_tr))
    neg = float(len(y_tr) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    models = []
    if "lr" in args.models:
        models.append(("lr", build_lr_pipeline()))
    if "xgb" in args.models:
        x = try_build_xgb(scale_pos_weight)
        if x is not None:
            models.append(("xgb", x))
    if "lgbm" in args.models:
        l = try_build_lgbm()
        if l is not None:
            models.append(("lgbm", l))

    rows = []
    for name, model in models:
        # 训练
        model.fit(X_tr, y_tr)
        # 保存
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

        # 验证集
        proba_va = model.predict_proba(X_va)[:, 1]
        va_roc  = roc_auc_score(y_va, proba_va)
        va_ap   = average_precision_score(y_va, proba_va)
        va_ks   = ks_stat(y_va, proba_va)

        # 测试集
        proba_te = model.predict_proba(X_te)[:, 1]
        te_roc  = roc_auc_score(y_te, proba_te)
        te_ap   = average_precision_score(y_te, proba_te)
        te_ks   = ks_stat(y_te, proba_te)

        rows.append({
            "model": name,
            "features": ",".join(feature_cols),
            "valid_roc_auc": float(va_roc),
            "valid_auc_pr": float(va_ap),
            "valid_ks": float(va_ks),
            "test_roc_auc": float(te_roc),
            "test_auc_pr": float(te_ap),
            "test_ks": float(te_ks),
        })

    # 汇总到报告
    out_df = pd.DataFrame(rows)
    out_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    print("Saved:", REPORTS_DIR / "model_metrics.csv")

if __name__ == "__main__":
    main()

