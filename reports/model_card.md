# Model Card｜信用卡欺诈检测 / Credit Card Fraud Detection

**Model name**: `lgbm_kaggle` (Pipeline: ColumnTransformer + LGBM / fallback LR)  
**Owners**: Sarah Zhou
**Last updated**: 2025-10-05
**Repository**: risk-ml-suite

---

## 1. Overview / 概览
**Purpose（目的）**  
- **CN**：用于在交易发生时识别高风险欺诈交易，降低漏报带来的资金损失，同时控制误报造成的运营成本。  
- **EN**: Identify high-risk fraudulent transactions at or near real-time to reduce FN loss while controlling FP operational costs.

**Primary metric（主指标）**  
- **AUC-PR**（极度不均衡任务）；辅以 **KS**、**ROC-AUC** 与 **Recall@Precision≥X**。  
- 目标（示例）：AUC-PR ≥ 0.90（或同类前 20%），Recall@P≥0.90 / 0.75 报告。

---

## 2. Intended Use / 预期用途
- **CN**：作为风控引擎的打分模型，结合阈值与规则引擎执行：自动拦截 / 人工审核 / 自动放行。  
- **EN**: Score input records to trigger decision flows: auto-block / manual review / auto-approve based on thresholds.

---

## 3. Data / 数据
- **Source（来源）**: Kaggle Credit Card Fraud Dataset（匿名化 PCA 特征 V1..V28 + Amount）。  
- **Time coverage（时间范围）**: 原始数据覆盖两天；本项目通过索引构造 `event_time` 便于滚动评估展示。We contructed a col `event_time` to capture the actual time of transaction for rolling evaluation on the model   
- **Split（切分）**: Train / Valid / Test；仅在 Train 拟合变换，避免信息泄露。  
- **Target（目标）**: `Class`（1 = fraud, 0 = non-fraud）。  
- **Imbalance（不均衡）**: 极度稀少正类（fraud rate ≈ 0.1%–0.2%）。

**Data quality & leakage controls（质量与防泄露）**  
- 剔除潜在后验字段（退款、投诉结果等，若有）。Remove all posterior features (if exists)  
- 所有标准化/过采样均在 Train 内执行；Valid/Test 仅转换。All standardised procedures are conducted in train set.

---

## 4. Model & Training / 模型与训练
- **Algorithms（算法）**: LightGBM（首选/Best）/ Logistic Regression（基线/Baseline）；`class_weight="balanced"`。  
- **Pipeline**: `ColumnTransformer(num -> StandardScaler) → Classifier`。  
- **Hyperparameters（关键超参）**: `n_estimators≈400, num_leaves≈31, learning_rate≈0.05, subsample≈0.8` 等。  
- **Calibration（概率校准）**: Platt / Isotonic（Brier Score 对比与曲线见附图）。  
- **Thresholding（阈值）**: 在 Valid 上进行**成本敏感阈值优化**（Cost_FN ≫ Cost_FP），Test 仅用于报告。

---

## 5. Performance / 指标表现
> （以下数字建议由脚本自动填入；未填前用你目前的观测值或留空）

| Split | AUC-PR | ROC-AUC | KS |
|------:|:------:|:-------:|:--:|
| Valid | 0.863 | 0.987 | 0.919 |
| Test  | 0.775  | 0.973  | 0.838  |

**Operating points（典型工作点）**  
- Recall@Precision≥0.90 = `0.750`（Test）  
- Recall@Precision≥0.75 = `0.750`（Test）  

**Calibration（校准）**  
- Brier(raw / platt / isotonic) = `0.000437	 / 0.000410	 / 0.000402	`  
- 见 `reports/figures/calibration_curves.png`。

> 解释：AUC-PR 反映**排序区分能力**；Brier 反映**概率可靠性**。二者互补：Isotonic 可降低 Brier，但可能微幅影响 AUC-PR。

---

## 6. Business Thresholding / 业务阈值策略
- **CN**：基于 FN≫FP 的成本矩阵，搜索阈值以最小化期望成本；再在 Test 固定阈值报告表现。并提供“概率分层策略”：  
  - p ≥ 0.9 → 自动拦截；0.5 ≤ p < 0.9 → 人工审核；p < 0.5 → 放行（阈值可按监管/KPI 调整）。  
- **EN**: Threshold chosen via expected-cost minimization (FN≫FP) on validation. Probability-tiered policy as above.
- 最佳阈值: 0.01 成本: 9085.0


---

## 7. Stability & Drift / 稳定性与漂移监控
- **Rolling metrics**: 按小时/天计算 AUC-PR/ROC-AUC/KS 趋势（`rolling_metrics_hourly.png`）。  
- **PSI**: Train vs Test 以及小时级 PSI（`psi_features.png`/`psi_hourly.csv`）；阈值：  
  - PSI < 0.10 稳定；0.10–0.25 轻微漂移；> 0.25 严重漂移（触发调查/复训）。  

---

## 8. Explainability / 可解释性
- **Global（全局）**: SHAP Top 特征（`shap_summary_bar.png`/`shap_summary_dot.png`）。  
- **Local（局部）**: 单笔交易 Top-K 特征贡献条形图（Dashboard 可交互展示）。  
- **Note**: 解释基于**原始模型**（未校准包），概率校准仅影响数值映射，不改变特征关系。

---

## 9. Ethical & Fairness / 伦理与公平
- **CN**：避免使用敏感属性（性别/种族等）；若业务落地需进行群体分布与误报/漏报差异评估，设置告警阈值与补救流程。  
- **EN**: Sensitive attributes excluded. For deployment, conduct group-wise FP/FN gap analysis with thresholds and remediation.

---

## 10. Limitations / 局限
- 数据仅覆盖两天，时间代表性有限；匿名化特征缺乏直观业务语义。  
- 指标在更长区间与多地域/渠道数据上需复验。

---

## 11. Monitoring & Retraining / 监控与再训练
- **Online**: 实时监控检出率、AUC-PR、Recall@P≥0.90、PSI。  
- **Triggers**: PSI > 0.25 或 Recall@P≥0.90 下滑超过阈值。  
- **Retrain**: 季度或触发式复训；保留灰度/回滚机制。

---

## 12. Versioning / 版本
- `models/lgbm_kaggle.joblib`（与 `kaggle_features.json` 对齐）  
- 生成记录：`model_metrics.csv`、`shap_meta.json`、`reports/figures/*`

