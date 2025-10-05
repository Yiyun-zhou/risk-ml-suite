from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn import metrics

def ks_stat(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))

def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(metrics.average_precision_score(y_true, y_score))

def precision_recall_at_threshold(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    p = metrics.precision_score(y_true, y_pred, zero_division=0)
    r = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    cm = metrics.confusion_matrix(y_true, y_pred)
    return p, r, f1, cm

def expected_cost(y_true, y_score, thr, costs):
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    c = (
        tp * costs.get("true_positive", 0.0)
        + tn * costs.get("true_negative", 0.0)
        + fp * costs.get("false_positive", 0.0)
        + fn * costs.get("false_negative", 0.0)
    )
    return float(c)

def optimize_threshold(y_true, y_score, costs, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    best_thr, best_cost = None, float("inf")
    for t in grid:
        c = expected_cost(y_true, y_score, t, costs)
        if c < best_cost:
            best_cost, best_thr = c, t
    return float(best_thr), float(best_cost)

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected, actual = np.array(expected), np.array(actual)
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, qs))
    edges[0] = -np.inf; edges[-1] = np.inf
    e_hist = np.histogram(expected, bins=edges)[0].astype(float)
    a_hist = np.histogram(actual, bins=edges)[0].astype(float)
    e_prop = np.clip(e_hist / e_hist.sum(), 1e-6, 1.0)
    a_prop = np.clip(a_hist / a_hist.sum(), 1e-6, 1.0)
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))

def save_json(d: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

def load_config(path: str = "config.yaml") -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)
