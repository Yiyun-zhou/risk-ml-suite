# src/thresholding.py
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

# 计算期望成本
def expected_cost(y_true, y_proba, thr, costs: dict) -> float:
    pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return (
        costs.get("false_positive", 1.0) * fp
        + costs.get("false_negative", 10.0) * fn
    )

# 搜索最优阈值
def optimize_threshold(y_true, y_proba, costs: dict, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99) # threshold grid
    best_thr, best_cost = None, np.inf
    for t in grid:
        c = expected_cost(y_true, y_proba, t, costs)
        if c < best_cost:
            best_cost, best_thr = c, t
    return best_thr, best_cost

# 灵敏度分析（返回成本曲线 DataFrame）
def sensitivity_analysis(y_true, y_proba, costs: dict, fn_factor=(0.5,1.0,1.5), grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    rows = []
    for f in fn_factor: # 比例参数 fn_factor 来测试：如果 FN 成本比现在高 50% 或低 50%，最优阈值会不会大幅变化
        mod_cost = costs.copy()
        mod_cost["false_negative"] *= f
        for t in grid:
            c = expected_cost(y_true, y_proba, t, mod_cost)
            rows.append({"thr":t, "fn_factor":f, "cost":c})
    return pd.DataFrame(rows)

# 保存灵敏度曲线图
def plot_sensitivity(df: pd.DataFrame, outpath: str|Path, opt_thr: float|None=None):
    plt.figure()
    for f, d in df.groupby("fn_factor"):
        plt.plot(d["thr"], d["cost"], label=f"FN x{f}")
    if opt_thr is not None:
        plt.axvline(opt_thr, linestyle="--", color="k", label=f"opt_thr={opt_thr:.2f}")
    plt.xlabel("threshold"); plt.ylabel("expected cost")
    plt.title("Cost-sensitive threshold sensitivity")
    plt.legend(); plt.grid(True)
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    

# 提取几个关键节点的指标
def threshold_nodes(y_true, y_proba, thresholds):
    rows = []
    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        P = tp / max(tp+fp,1); R = tp / max(tp+fn,1)
        rows.append({"thr":thr,"precision":P,"recall":R,"tp":tp,"fp":fp,"fn":fn,"tn":tn})
    return pd.DataFrame(rows)