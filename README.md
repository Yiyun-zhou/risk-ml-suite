# 📊 End-to-End Credit Risk Model with Risk Monitoring Dashboard  
*(端到端信用风险建模与风控监控系统)*

---

## 🌐 Overview

This repository implements an **end-to-end credit risk / fraud detection system**,  
covering the full lifecycle from **data preparation, model training, threshold optimization, interpretability, to MLOps-style monitoring dashboard**.

该项目完整复现了一个**端到端的信用风险（违约/欺诈）建模与监控系统**，  
涵盖了从数据准备、建模、成本敏感阈值优化、可解释性分析到稳定性监控与可视化 Dashboard 的全过程。  

---

## 🧭 Project Structure

risk-ml-suite/
├── app/                     # Streamlit dashboard for risk monitoring
├── data/                    # Raw / processed data splits (train, valid, test)
├── models/                  # Serialized models (.joblib)
├── reports/
│   ├── figures/             # Evaluation plots (ROC, PR, KS, PSI, SHAP)
│   ├── risk_memo.pdf        # Risk management memo (bilingual)
│   └── model_card.md        # Model documentation card
├── src/
│   ├── data_prep.py         # Data preprocessing & leakage protection
│   ├── train.py             # Baseline and advanced models (LR, XGB, LGBM)
│   ├── eval.py              # Metrics, threshold tuning, sensitivity 
├── tests/                   # Unit tests (leakage, PSI, metric validation)
├── config.yaml              # Global config for paths & experiment setup
├── leakage_checklist.yaml   # Potential leakage fields to exclude
├── Makefile                 # One-click reproducibility scripts
├── requirements.txt         # Python dependencies
└── README.md

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

- Cost-sensitive decisioning → dynamically optimized threshold minimizing expected loss
- Explainable AI (SHAP) → feature-level reasoning for risk policy design
- Population Stability Index (PSI) → drift detection & retraining trigger
- Streamlit dashboard → real-time visualization of KPIs, PSI, SHAP summary
- Model governance pipeline → automated Risk Memo PDF + bilingual Model Card
- Reproducible build → make all and fixed random seeds for consistent results


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



