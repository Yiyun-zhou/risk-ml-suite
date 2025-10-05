# ===== calibration_utils.py (or paste into a notebook cell) =====
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score
import shap

# --------- 概率校准 ---------

def _evaluate_probs(model, Xv, yv, Xt, yt, p_te) -> pd.DataFrame:
    # 1. Platt scaling (sigmoid)
    cal_platt = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
    cal_platt.fit(Xv, yv)
    p_platt = cal_platt.predict_proba(Xt)[:,1]

    # 2. Isotonic regression
    cal_iso = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    cal_iso.fit(Xv, yv)
    p_iso = cal_iso.predict_proba(Xt)[:,1]

    # 3. Raw
    p_raw = p_te

    p_list = [p_raw, p_platt, p_iso]
    model_name = ['Raw', 'Platt', 'Isotonic']
    rows = []

    for i in range(len(p_list)):
        rows.append({
            "model": model_name[i],
            "brier": brier_score_loss(yt, p_list[i]),
            "roc_auc": roc_auc_score(yt, p_list[i]),
            "auc_pr": average_precision_score(yt, p_list[i]),
        })
    return p_platt, p_iso, pd.DataFrame(rows).sort_values(["brier","auc_pr"], ascending=[True, False]).reset_index(drop=True)

def _plot_calibration_curves(p_raw, p_platt, p_iso, y_true):

    plt.figure(figsize=(6,6))
    for lab, p in [("raw", p_raw), ("platt", p_platt), ("isotonic", p_iso)]:
        prob_true, prob_pred = calibration_curve(y_true, p, n_bins=20)
        plt.plot(prob_pred, prob_true, marker="o", label=lab)
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("Predicted probability")
    plt.ylabel("True frequency")
    plt.title("Calibration curves (test)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("figures/calibration_curves.png", dpi=160, bbox_inches="tight")
    plt.show()


# --------- Shap ---------

def shap_val(model, X_test):
# 用 tree explainer (适合 XGB/LGBM)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return explainer, shap_values

def plot_shap_summary_bar(shap_values, Xt):
    # 全局特征重要性
    shap.summary_plot(shap_values, Xt, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("figures/shap_summary_bar.png", dpi=160)
    plt.show()

# SHAP 分布图
def plot_shap_distribution_dot(shap_values, Xt):
    shap.summary_plot(shap_values, Xt, show=False)
    plt.tight_layout()
    plt.savefig("figures/shap_summary_dot.png", dpi=160)
    plt.show()

# 局部解释（举例第 100 条交易）
def explain_transaction(explainer, shap_values, Xt, i):
    shap.force_plot(explainer.expected_value, shap_values[i,:], Xt.iloc[i,:], matplotlib=True)
    plt.savefig("figures/shap_force_example.png", dpi=160)
    plt.show()


