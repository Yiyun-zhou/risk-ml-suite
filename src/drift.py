import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib

def compute_drift_metrics(model, test_df):
    # 按小时分组
    test_df["hour"] = test_df["event_time"].dt.to_period("H")
    feats = ["Amount"] + [c for c in test_df.columns if c.startswith("V")]

    rows = []
    for m, d in test_df.groupby("hour"):
        X, y = d[feats], d["Class"]
        proba = model.predict_proba(X)[:,1]
        ap = average_precision_score(y, proba)
        roc = roc_auc_score(y, proba)
        rows.append({"hour": str(m), "auc_pr": ap, "roc_auc": roc, "count": len(d)})

    metrics = pd.DataFrame(rows)
    metrics.to_csv("drift_metrics.csv", index=False)

    return metrics


def plot_drift_trend(metrics):
    # 趋势图
    plt.figure(figsize=(8,5))
    plt.plot(metrics["month"], metrics["auc_pr"], marker="o", label="AUC-PR")
    plt.plot(metrics["month"], metrics["roc_auc"], marker="s", label="ROC-AUC")
    plt.xticks(rotation=45)
    plt.ylabel("Score"); plt.title("Rolling metrics by month (test)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("figures/rolling_metrics.png", dpi=160, bbox_inches="tight")
    plt.show()


# src/drift.py 里替换
import numpy as np
import pandas as pd

def compute_psi(expected, actual, bins=10, strategy="quantile",  _EPS = 1e-6):
    e = pd.Series(expected).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    a = pd.Series(actual).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(e) == 0 or len(a) == 0:
        return np.nan

    if strategy == "quantile":
        cuts = np.quantile(e, np.linspace(0, 1, bins + 1))
    elif strategy == "uniform":
        lo, hi = float(e.min()), float(e.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi: return 0.0
        cuts = np.linspace(lo, hi, bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    cuts = np.unique(cuts)
    if len(cuts) <= 1: return 0.0

    e_counts, _ = np.histogram(e, bins=cuts)
    a_counts, _ = np.histogram(a, bins=cuts)
    e_ratio = np.clip(e_counts / max(e_counts.sum(), 1), _EPS, 1.0)
    a_ratio = np.clip(a_counts / max(a_counts.sum(), 1), _EPS, 1.0)

    return float(np.sum((e_ratio - a_ratio) * np.log(e_ratio / a_ratio)))


def psi_report(expected_df, actual_df, features, bins=10, strategy="quantile", _EPS = 1e-6):
    rows = []
    for f in features:
        psi = compute_psi(expected_df[f].values, actual_df[f].values, bins=bins, strategy=strategy)
        if np.isnan(psi):
            status = "insufficient_data"
        elif psi < 0.1:
            status = "stable"
        elif psi < 0.25:
            status = "slight_drift"
        else:
            status = "major_drift"
        rows.append({"feature": f, "psi": psi, "status": status})
    
    psi_repo = pd.DataFrame(rows)
    psi_repo.to_csv('PSI_feature_metrics.csv', index=False)

    return psi_repo

