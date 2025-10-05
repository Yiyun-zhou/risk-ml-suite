# -*- coding: utf-8 -*-
"""
生成双语 Risk Memo PDF
用法：
  python -m src.generate_docs

依赖：
  pip install reportlab pandas numpy
"""
from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# 注册支持中文的字体
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))


# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parents[1]  # 项目根目录 risk-ml-suite/
REPORTS = ROOT / "reports"
FIGDIR  = REPORTS / "figures"

METRICS_CANDIDATES = [
    REPORTS / "model_metrics.csv",
    ROOT / "model_metrics.csv",
]

SHAP_VALUES = REPORTS / "shap_values_sample.npy"
#SHAP_X      = REPORTS / "shap_X_sample.csv"
#SHAP_META   = REPORTS / "shap_meta.json"

FIG_CANDIDATES = {
    "PR": FIGDIR / "pr_curve_xgb_lgbm.png",
    "Calib": FIGDIR / "calibration_curves.png",
    "Rolling": FIGDIR / "rolling_metrics.csv.png",
    "PSI": FIGDIR / "psi_features.png",
    "SHAP_BAR": FIGDIR / "shap_summary_bar.png",
    "SHAP_DOT": FIGDIR / "shap_summary_dot.png",
}

# ---------- 小工具 ----------
def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

def safe_img(path: Path, width=480):
    if path and path.exists():
        return Image(str(path), width=width, height=width*0.62)  # 约 16:10
    return None

def load_metrics():
    csv_path = first_existing(METRICS_CANDIDATES)
    if not csv_path:
        return None
    df = pd.read_csv(csv_path)
    # 取表现最好的模型（test_auc_pr 最大）
    if "test_auc_pr" in df.columns:
        df = df.sort_values("test_auc_pr", ascending=False)
    return df

def load_shap_topk(k=10):
    if not SHAP_VALUES.exists(): #and SHAP_META.exists()):
        return None
    sv = np.load(SHAP_VALUES)  # [N, d]
    #meta = json.loads(SHAP_META.read_text())
    #feats = meta.get("feature_cols", [])
    feats = json.loads((ROOT/"models"/"kaggle_features.json").read_text())
    mean_abs = np.mean(np.abs(sv), axis=0)
    order = np.argsort(mean_abs)[::-1][:min(k, len(mean_abs))]
    rows = [{"rank": i+1, "feature": feats[idx], "mean_|SHAP|": float(mean_abs[idx])} for i, idx in enumerate(order)]
    return pd.DataFrame(rows)

# ---------- 文本样式 ----------
styles = getSampleStyleSheet()
Body = ParagraphStyle(
    "Body", parent=styles["BodyText"],
    fontName="STSong-Light",  # ← 改这里
    fontSize=10.5, leading=15, alignment=TA_LEFT
)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18,
                    fontName="STSong-Light", spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14,
                    fontName="STSong-Light", spaceAfter=4)


def p(text): return Paragraph(text, Body)
def h1(text): return Paragraph(text, H1)
def h2(text): return Paragraph(text, H2)

# ---------- 主函数 ----------
def build_story():
    story = []

    # 读取指标 & SHAP
    metrics = load_metrics()
    shap_top = load_shap_topk(k=10)

    # 取一个代表行
    rep = metrics.iloc[0].to_dict() if metrics is not None and len(metrics) else {}
    mdl_name = rep.get("model", "Model")
    test_ap  = rep.get("test_auc_pr", None)
    test_auc = rep.get("test_roc_auc", None)
    test_ks  = rep.get("test_ks", None)
    va_ap    = rep.get("valid_auc_pr", None)
    va_auc   = rep.get("valid_roc_auc", None)
    va_ks    = rep.get("valid_ks", None)

    # 封面/摘要
    story += [
        h1("Risk Management Memo 风控备忘录（中英双语）"),
        Spacer(1, 6),
        p(
        "【中文摘要】本备忘录记录了信用卡欺诈检测模型的目标、数据、方法、表现、稳定性监控与解释结果。"
        "模型主指标以 AUC-PR 为准，同时报告 KS 与校准。我们采用成本敏感阈值优化与 PSI 漂移监控，"
        "并在 Dashboard 中提供阈值滑杆、KPI 趋势与 SHAP 解释。"),
        Spacer(1, 2),
        p(
        "<i>[EN Summary]</i> This memo documents objectives, data, methods, performance, calibration, "
        "stability monitoring, and explainability for the credit-card fraud detection model. "
        "We use AUC-PR as the primary metric, report KS and calibration quality, optimize the decision "
        "threshold under a cost-sensitive framework, and monitor data drift via PSI. The Streamlit "
        "dashboard provides threshold control, KPI trends, and SHAP explanations."),
        Spacer(1, 12),

        h2("1. Objective 目标"),
        p("【中】通过监督学习识别高风险交易，减少漏报造成的资金损失，并控制误报带来的运营成本。"),
        p("<i>[EN]</i> Identify high-risk transactions via supervised learning to reduce FN losses "
          "while controlling FP operational costs."),
        Spacer(1, 6),

        h2("2. Data & Processing 数据与处理"),
        p("【中】数据源：Kaggle Credit Card Fraud。特征：Amount + V1..V28（PCA 匿名化），按时间分层切分 "
          "为 train/valid/test；仅在训练集拟合变换，避免信息泄露。类别极不均衡，以 AUC-PR 为主指标。"),
        p("<i>[EN]</i> Source: Kaggle Credit Card Fraud. Features: Amount + V1..V28. "
          "Temporal/stratified split into train/valid/test. All transforms fitted on train to avoid leakage. "
          "Severe class imbalance; AUC-PR is the primary metric."),
        Spacer(1, 6),

        h2("3. Model & Threshold 模型与阈值"),
        p(f"【中】当前报告的代表模型：<b>{mdl_name}</b>。在验证集上进行成本敏感阈值优化（FN≫FP），"
          "并将最优阈值固定后于测试集上报告指标。"),
        p("<i>[EN]</i> The representative model above was tuned with a cost-sensitive threshold "
          "on validation and then evaluated on test with the chosen threshold fixed."),
        Spacer(1, 6),
    ]

    # 指标表格（valid & test）
    if metrics is not None and len(metrics):
        tbl_data = [["", "Valid AUC-PR", "Valid ROC-AUC", "Valid KS", "Test AUC-PR", "Test ROC-AUC", "Test KS"]]
        row = [
            rep.get("model", "Model"),
            f"{va_ap:.3f}" if va_ap is not None else "-",
            f"{va_auc:.3f}" if va_auc is not None else "-",
            f"{va_ks:.3f}" if va_ks is not None else "-",
            f"{test_ap:.3f}" if test_ap is not None else "-",
            f"{test_auc:.3f}" if test_auc is not None else "-",
            f"{test_ks:.3f}" if test_ks is not None else "-",
        ]
        tbl_data.append(row)
        t = Table(tbl_data, colWidths=[90, 80, 85, 65, 80, 85, 65])
        t.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ]))
        story += [t, Spacer(1, 10)]

    # 插图（若存在就嵌入）
    story += [h2("4. Key Plots 关键图表")]
    for label, path in FIG_CANDIDATES.items():
        img = safe_img(path)
        if img:
            cap_cn = {
                "PR": "PR 曲线与最优阈值",
                "Calib": "概率校准曲线",
                "Rolling": "按小时滚动指标",
                "PSI": "特征 PSI 漂移监控",
                "SHAP_BAR": "SHAP 全局重要性（Bar）",
                "SHAP_DOT": "SHAP 全局分布（Dot）",
            }.get(label, label)
            cap_en = {
                "PR": "PR curve with optimal threshold",
                "Calib": "Calibration curves",
                "Rolling": "Hourly rolling metrics",
                "PSI": "Feature PSI drift",
                "SHAP_BAR": "SHAP global importance (bar)",
                "SHAP_DOT": "SHAP global distribution (dot)",
            }.get(label, label)
            story += [p(f"【中】{cap_cn}  |  <i>[EN]</i> {cap_en}"), Spacer(1, 4), img, Spacer(1, 10)]

    # 校准与稳定性
    story += [
        h2("5. Calibration & Stability 校准与稳定性"),
        p("【中】通过 Platt/Isotonic 进行概率校准，Brier Score 下降说明概率更可靠，可用于概率分层策略。"
          "建立 PSI 阈值：PSI<0.1 稳定；0.1–0.25 轻微漂移；>0.25 严重漂移需排查与复训。"),
        p("<i>[EN]</i> Platt/Isotonic calibration improved the Brier score, enabling probability-tiered decisions. "
          "PSI thresholds: <0.1 stable; 0.1–0.25 slight drift; >0.25 major drift triggering investigation/retraining."),
        Spacer(1, 6),
    ]

    # 可解释性（SHAP Top）
    story += [h2("6. Explainability (SHAP) 可解释性")]
    if shap_top is not None and len(shap_top):
        tbl = [["Rank / 排名", "Feature / 特征", "mean(|SHAP|)"]]
        for _, r in shap_top.iterrows():
            tbl.append([int(r["rank"]), r["feature"], f'{r["mean_|SHAP|"]:.4f}'])
        t2 = Table(tbl, colWidths=[90, 240, 140])
        t2.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("ALIGN", (0,0), (-1,0), "CENTER"),
        ]))
        story += [p("【中】Top 特征从业务侧提示了风险信号，配合规则引擎做优先审核。"
                    "<i>[EN]</i> Top features highlight risk signals for priority review and rule-engine actions."),
                  Spacer(1, 4), t2, Spacer(1, 10)]
    else:
        story += [p("【中】（未检测到 SHAP 预计算文件） | <i>[EN]</i> SHAP precomputed files not found."), Spacer(1, 6)]

    # 治理与复训计划
    story += [
        h2("7. Governance & Retraining 模型治理与再训练"),
        p("【中】治理与上线：阈值选择基于验证集的成本最小化，测试集仅用于报告；建立线上监控（PSI/AUC-PR），"
          "触发条件：PSI>0.25 或 Recall@P≥0.90 下滑超过阈值。复训频率：季度或触发式。回滚：保留前一版模型，灰度发布。"),
        p("<i>[EN]</i> Governance: threshold chosen on validation via expected cost minimization; "
          "test is for reporting only. Online monitoring with PSI/AUC-PR. Triggers: PSI>0.25 or "
          "Recall@P≥0.90 deterioration beyond a set threshold. Retrain quarterly or upon trigger. "
          "Rollback with previous version and canary releases."),
        Spacer(1, 6),

        h2("8. Limitations & Fairness 局限与公平"),
        p("【中】数据仅覆盖两天交易，时间代表性有限；匿名特征难以直接给出业务语义。建议在更广时段与多来源数据上复验，"
          "并补充公平性评估（如群体间误报/漏报差异）。"),
        p("<i>[EN]</i> Data covers only two days; temporal representativeness is limited. PCA-anonymized features "
          "lack direct business semantics. Validate on wider periods and evaluate fairness (group-wise FP/FN gaps)."),
        Spacer(1, 6),

        p("【完】本备忘录与 Dashboard 一起，形成了可复现、可解释、可监控的风控项目样例。"
          "<i>[EN]</i> Together with the dashboard, this memo completes a reproducible, explainable and monitorable risk ML demo."),
    ]
    return story

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    out_pdf = REPORTS / "risk_memo.pdf"

    doc = SimpleDocTemplate(
        str(out_pdf), pagesize=A4,
        leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    story = build_story()
    doc.build(story)
    print(f"[OK] Generated: {out_pdf}")

if __name__ == "__main__":
    main()
