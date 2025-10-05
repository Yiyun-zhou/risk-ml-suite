from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, joblib
from sklearn.metrics import precision_recall_curve, roc_curve
from .utils import load_config, optimize_threshold, precision_recall_at_threshold, save_json

ROOT = Path(__file__).resolve().parents[1]

def main():
    cfg = load_config(ROOT / "config.yaml")
    metrics = pd.read_csv(ROOT / "reports" / "model_metrics.csv")
    best = metrics.sort_values("valid_auc_pr", ascending=False).iloc[0]["model"]
    model = joblib.load(ROOT / "models" / f"{best}.joblib")

    valid = pd.read_csv(ROOT / "data/processed/valid.csv", parse_dates=["event_time"])
    test  = pd.read_csv(ROOT / "data/processed/test.csv", parse_dates=["event_time"])
    yv = valid[cfg["defaults"]["target"]].values
    Xt = test.drop(columns=[cfg["defaults"]["target"]])
    yt = test[cfg["defaults"]["target"]].values

    pv = model.predict_proba(valid.drop(columns=[cfg["defaults"]["target"]]))[:,1]
    pt = model.predict_proba(Xt)[:,1]

    thr, cost = optimize_threshold(yv, pv, cfg["costs"])
    p,r,f1,cm = precision_recall_at_threshold(yt, pt, thr)

    out = {
        "best_model": best,
        "threshold": float(thr),
        "expected_cost_valid": float(cost),
        "test_precision": float(p),
        "test_recall": float(r),
        "test_f1": float(f1),
        "test_confusion_matrix": [int(x) for x in cm.ravel()]
    }
    save_json(out, ROOT / "reports" / "threshold_metrics.json")

    pr_p, pr_r, _ = precision_recall_curve(yt, pt)
    plt.figure()
    plt.step(pr_r, pr_p, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall (test)")
    (ROOT / "reports/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(ROOT / "reports/figures/pr_curve.png", dpi=160, bbox_inches="tight"); plt.close()

    fpr, tpr, _ = roc_curve(yt, pt)
    plt.figure(); plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (test)")
    plt.savefig(ROOT / "reports/figures/roc_curve.png", dpi=160, bbox_inches="tight"); plt.close()

    # daily aggregates for dashboard
    test_df = test.copy()
    test_df["proba"] = pt
    test_df["date"] = test_df["event_time"].dt.date
    daily = test_df.groupby("date").agg(
        volume=("tx_id","count"),
        frauds=("Class","sum"),
        fraud_rate=("Class","mean"),
        pr_50=("proba", "median"),
    ).reset_index()
    daily.to_csv(ROOT / "reports/daily_agg.csv", index=False)

    # PSI baseline vs latest week
    first_week = test_df[test_df["event_time"] < test_df["event_time"].min() + pd.Timedelta(days=7)]["proba"].values
    last_week  = test_df[test_df["event_time"] > test_df["event_time"].max() - pd.Timedelta(days=7)]["proba"].values
    np.save(ROOT / "reports/psi_baseline.npy", first_week)
    np.save(ROOT / "reports/psi_actual.npy",   last_week)

if __name__ == "__main__":
    main()
