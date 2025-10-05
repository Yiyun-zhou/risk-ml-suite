# 📊 End-to-End Credit Risk Model with Risk Monitoring Dashboard  
（端到端信用风险 / 欺诈检测模型与风控监控系统）

---

## Overview

This repository demonstrates a full-stack **credit risk / fraud detection framework**,  
spanning data preprocessing, model training, cost-sensitive thresholding, interpretability, stability monitoring, and a dashboard for real-time oversight.

该仓库实现了从数据管道、建模、成本敏感阈值优化、可解释性分析，到稳定性监控与可视化 Dashboard 的完整端到端风控系统。

---

## Project Structure

```text
risk-ml-suite/
├── app/                        # Streamlit 风控监控系统前端（dashboard）
│   └── streamlit_app.py
├── data/                       # 原始与处理后数据（train / valid / test）
├── models/                     # 导出模型与特征映射 json
├── reports/                    # 报告与可视化产出
│   ├── figures/                # 各类图表（PR, PSI, SHAP, drift 等）
│   ├── risk_memo.pdf           # 风控备忘录（中英双语）
│   └── model_card.md           # 模型卡（文档化说明）
├── src/                        # 核心源码
│   ├── data_prep.py            # 数据清洗、特征构建、泄露检查
│   ├── train.py                # 模型建立函数（LR / XGB / LGBM 等）
│   ├── eval.py                 # 指标计算、阈值搜索、灵敏度分析
│   ├── drift.py                # 稳定性 / 漂移监控 (PSI / rolling) 
│   ├── explain.py              # 模型解释 (SHAP) 支持函数
│   └── generate_docs.py        # 自动生成风控备忘录与模型卡脚本
├── tests/                      # 单元测试（可检测数据泄露、指标正确性等）
├── config.yaml                 # 全局路径 / 配置（如数据路径、超参等）
├── leakage_checklist.yaml      # 潜在泄露字段清单供人工审查
├── Makefile                    # 一键复现命令（setup、train、report、dashboard）
├── requirements.txt            # Python 依赖包列表
├── .gitignore                   # 忽略列表（包括 .venv、__pycache__ 等）
└── README.md                   # 本文档

---

## ⚙️ Quickstart

```bash
# 1️⃣ Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ (Optional) Download dataset from Kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip

# 4️⃣ Run full pipeline
make all

# 5️⃣ Launch Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## 🧠 Methodology

| Step | Description | Key Outputs |
| --- | --- | --- |
| EDA & Data Validation | Data profiling, leakage screening, and imbalance assessment | Data quality report, class ratio |
| Baseline Modeling | Logistic Regression (class_weight='balanced') | ROC-AUC ≈ 0.98, AUC-PR ≈ 0.83 |
| Advanced Models | XGBoost / LightGBM with stratified CV | Performance comparison across models |
| Cost-Sensitive Threshold Optimization | Business-driven expected cost minimization | Optimal recall–precision tradeoff |
| Probability Calibration | Platt & Isotonic calibration, Brier score evaluation | Calibrated probabilities for policy layers |
| Explainability (SHAP) | Global & local feature importance, business-driven insights | Feature thresholds & risk driver analysis |
| Stability & Drift | PSI-based population drift & rolling KS monitoring | Drift report & alarm trigger (PSI > 0.25) |
| Governance & Documentation | Risk memo (PDF) + Model card | Traceability, re-train policy, rollback plan |

---

## 🧩 Key Features

- **Cost-sensitive decisioning** → dynamically optimized threshold minimizing expected loss
- **Explainable AI (SHAP)** → feature-level reasoning for risk policy design
- **Population Stability Index (PSI)** → drift detection & retraining trigger
- **Streamlit dashboard** → real-time visualization of KPIs, PSI, SHAP summary
- **Model governance pipeline** → automated Risk Memo PDF + bilingual Model Card
- **Reproducible build** → make all and fixed random seeds for consistent results


---

## 🖥️ Practical Applications

- Fraud or credit default detection in fintech and banking risk systems
- Cost-sensitive decision optimization for loan approval or payment anomaly detection
- Real-time model monitoring in production-grade MLOps pipelines
- Explainability & model governance compliance under Basel III / EU AI Act


- 金融科技与银行业风控系统中的欺诈与违约检测；
- 放贷审批、支付风控中的成本敏感策略优化；
- 生产级 MLOps 系统中的实时模型监控；
- 满足 Basel III / EU AI Act 的可解释与治理合规要求。

---

## 🧑‍💻 Author

Yiyun (Sarah) Zhou

MSc Quantitative Finance, ETH Zurich & University of Zurich

Focus: Risk Analytics, Quantitative Modeling, and Machine Learning for Finance

📧 Contact: yiyun1.zhou@gmail.com



