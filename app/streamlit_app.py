import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np
#from src.drift import psi_report

from pathlib import Path
import sys, json, joblib, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
model = joblib.load(ROOT / "models" / "lgbm_kaggle.joblib")
feature_cols = json.loads((ROOT/"models"/"kaggle_features.json").read_text())


# 载入数据 & 模型
train =  pd.read_csv("data/processed/train.csv", parse_dates=["event_time"])
valid = pd.read_csv("data/processed/valid.csv", parse_dates=["event_time"])
test  = pd.read_csv("data/processed/test.csv",  parse_dates=["event_time"])

st.set_page_config(page_title="Risk ML Dashboard", layout="wide")


MODEL_PATH = ROOT / "models" / "lgbm_kaggle.joblib"

def unwrap_pipeline(model):
    prep, clf = None, model
    if hasattr(model, "named_steps"):
        prep = model.named_steps.get("prep", None)
        clf = model.named_steps.get("clf", model)
    if hasattr(clf, "base_estimator_"):
        base = getattr(clf, "base_estimator_", None)
        if base is not None:
            clf = base
    return prep, clf

@st.cache_data(show_spinner=False)  # 关键：缓存按 model_path + n_samples + feature_cols 内容
def compute_shap_from_path(model_path_str, df, feature_cols, n_samples=400):
    model = joblib.load(model_path_str)  # 在缓存函数内部加载
    prep, clf = unwrap_pipeline(model)

    X_df = df[feature_cols].sample(min(n_samples, len(df)), random_state=42)
    X_used = prep.transform(X_df) if prep is not None else X_df.values

    try:
        import lightgbm as lgb, xgboost as xgb
        if isinstance(clf, (lgb.LGBMClassifier, xgb.XGBClassifier)):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_used)
            if isinstance(sv, list): sv = sv[1]
            exp_value = explainer.expected_value
            if isinstance(exp_value, (list, np.ndarray)):
                exp_value = exp_value[1]
            model_type = "tree"
            return sv, X_used, float(exp_value), model_type
    except Exception:
        pass

    from sklearn.linear_model import LogisticRegression
    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, X_used)
        sv = explainer.shap_values(X_used)
        exp_value = explainer.expected_value
        return sv, X_used, float(np.array(exp_value).mean()), "linear"

    # 兜底：permutation（对整个 pipeline）
    f = lambda X_df_input: model.predict_proba(X_df_input)[:, 1]
    bg = df[feature_cols].sample(min(200, len(df)), random_state=0)
    explainer = shap.Explainer(f, bg, algorithm="permutation")
    ex = explainer(X_df)
    sv = np.array(ex.values)
    exp_value = float(np.array(ex.base_values).mean())
    return sv, X_df.values, exp_value, "perm"



# --- 页面选择 ---
menu = ["KPI 总览", "阈值优化", "漂移监控", "模型解释"]
choice = st.sidebar.radio("选择", menu)

# --- 1. KPI 总览 ---
if choice == "KPI 总览":
    st.header("📊 KPI 总览")
    fraud_rate = test["Class"].mean()
    st.metric("Fraud Rate", f"{fraud_rate:.3%}")

    # 假设我们在 reports/drift_metrics_hourly.csv 里有 rolling 指标
    metrics = pd.read_csv("reports/drift_metrics.csv")
    st.line_chart(metrics.set_index("hour")[["auc_pr", "roc_auc"]])

# --- 2. 阈值优化 ---
elif choice == "阈值优化":
    st.header("⚖️ 阈值优化")
    from sklearn.metrics import precision_recall_fscore_support

    X = test[feature_cols]
    proba = model.predict_proba(X)[:, 1]
    
    X, y = test[feature_cols], test["Class"]
    proba = model.predict_proba(X)[:,1]

    thr = st.slider("选择阈值", 0.0, 1.0, 0.5, 0.01)
    pred = (proba >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary")

    st.write(f"Precision={p:.3f} | Recall={r:.3f} | F1={f:.3f}")

    fig, ax = plt.subplots()
    ax.hist(proba, bins=50, alpha=0.7)
    ax.axvline(thr, color="red", linestyle="--", label=f"Threshold={thr}")
    ax.legend()
    st.pyplot(fig)

# --- 3. 漂移监控 ---
elif choice == "漂移监控":
    st.header("📈 特征漂移监控")
    train = pd.read_csv("data/processed/train.csv", parse_dates=["event_time"])
    psi_df = pd.read_csv("reports/PSI_feature_metrics.csv")

    st.dataframe(psi_df)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(psi_df["feature"], psi_df["psi"], color="steelblue")
    ax.axvline(0.1, color="orange", linestyle="--")
    ax.axvline(0.25, color="red", linestyle="--")
    st.pyplot(fig)

# --- 4. 模型解释 (SHAP) ---
elif choice == "模型解释":
    st.header("🔍 模型解释 (SHAP)")

    # 路径
    SHAP_DIR = ROOT / "reports"
    FIG_DIR  = SHAP_DIR / "figures"
    # 读 meta / values / X
    import json, numpy as np, pandas as pd, matplotlib.pyplot as plt

    sv_path   = SHAP_DIR / "shap_values_sample.npy"

    shap_values = np.load(sv_path)              # [N, d]
   
    # === 全局图：直接展示已生成的图片 ===
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("全局重要性（Bar）")
        st.image(str(FIG_DIR / "shap_summary_bar.png"), use_column_width=True)
    with col2:
        st.subheader("全局分布（Dot）")
        st.image(str(FIG_DIR / "shap_summary_dot.png"), use_column_width=True)

    # === 局部解释函数：Top-K 条形图 ===
    def plot_local_explanation(sv_row: np.ndarray, feature_cols, k=8, base=0.0):
        top_idx = np.argsort(np.abs(sv_row))[::-1][:k]
        feats = np.array(feature_cols)[top_idx]
        vals  = sv_row[top_idx]
        fig = plt.figure(figsize=(6, 4))
        colors = ["tab:red" if v > 0 else "tab:blue" for v in vals]
        plt.barh(feats[::-1], vals[::-1], color=colors[::-1])
        plt.axvline(0, color="k", linewidth=0.8)
        plt.title(f"Top-{k} features(base={base:.3f})")
        plt.tight_layout()
        return fig

    st.subheader("局部解释（单笔交易）")
    k = st.slider("Top-K 特征", 3, 20, 8, 1)
    idx_local = st.number_input("选择样本 index（0 ~ N-1）", min_value=0, max_value=len(shap_values)-1, value=0)

    sv_i = shap_values[idx_local]
    fig = plot_local_explanation(sv_i, feature_cols, k=k)
    st.pyplot(fig, clear_figure=True)
