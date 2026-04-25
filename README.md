# 📡 Customer Churn Prediction — Telco Industry

A complete end-to-end machine learning pipeline for predicting customer churn in the telecommunications industry, featuring automated data cleaning, domain-driven feature engineering, multi-model comparison with and without SMOTE, stacking ensemble, and SHAP-based interpretability.

---

## 📊 Results Summary

| Model | Setting | Accuracy | F1 (Churn) | ROC-AUC | PR-AUC |
|-------|---------|----------|------------|---------|--------|
| Logistic Regression | No SMOTE | 0.80 | 0.64 | **0.856** | **0.665** |
| Decision Tree | No SMOTE | 0.76 | 0.54 | 0.701 | 0.388 |
| Random Forest | No SMOTE | 0.83 | 0.57 | 0.846 | 0.639 |
| XGBoost | No SMOTE | 0.79 | 0.52 | 0.824 | 0.591 |
| Logistic Regression | With SMOTE | 0.81 | 0.64 | 0.855 | 0.657 |
| Decision Tree | With SMOTE | 0.80 | 0.58 | 0.724 | 0.434 |
| Random Forest | With SMOTE | 0.83 | **0.62** | 0.846 | 0.641 |
| XGBoost | With SMOTE | 0.82 | 0.59 | 0.847 | 0.635 |
| Stacking Ensemble | RF + XGB + LR | **0.83** | 0.55 | 0.843 | 0.634 |

> **Key finding:** Logistic Regression achieves the best discriminative power (ROC-AUC 0.856), while Random Forest + SMOTE provides the best balance of accuracy and churn recall.

---

## 🗂️ Project Structure

```
├── telco-customer-churn.ipynb   # Main notebook (full pipeline)
├── README.md
└── cleaned_data.csv             # Output of preprocessing pipeline (generated at runtime)
```

---

## ⚙️ Pipeline Overview

### 1. Data Cleaning
- Detects and replaces proxy null values (`NA`, `None`, empty strings, etc.)
- Drops columns with >50% missing values
- Mean imputation for numeric, mode imputation for categorical columns
- IQR-based outlier removal (1.5× fence)
- Min-Max normalization

### 2. Feature Engineering (15 new features)

| Category | Features |
|----------|----------|
| Customer Lifecycle | `tenure_group`, `avg_monthly_spend` |
| Spending Behavior | `high_value`, `totalcharges_group`, `monthly_contract_interaction`, `tenure_charge_interaction` |
| Service Engagement | `num_services`, `service_diversity`, `has_support`, `security_services` |
| Risk Indicators | `contract_risk`, `auto_payment`, `paperless_flag`, `senior_alone` |
| Composite | `engagement_score` |

### 3. Preprocessing & Feature Selection
- `StandardScaler` for numeric features
- `OneHotEncoder` for categorical features
- `SelectFromModel` (Random Forest, median threshold)

### 4. Models
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Stacking Ensemble (RF + XGBoost → Logistic Regression meta-learner)

Each model is trained in two settings: **standard** and **SMOTE-augmented**.

### 5. Evaluation
- Accuracy, F1-score (churn class), ROC-AUC, PR-AUC
- Precision-Recall Curve (Stacking Ensemble)
- Feature Importance (Random Forest)
- SHAP values (XGBoost)

---

## 📦 Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn shap
```

---

## 🚀 How to Run

1. Download the dataset from Kaggle:  
   [Telco Customer Churn — IBM Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

2. Place the CSV file at:
   ```
   /kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```
   Or update the path in the notebook's **Load Data** cell.

3. Run all cells in `telco-customer-churn.ipynb`.

---

## 🔍 Key Insights

- **Contract type** is the strongest churn driver — month-to-month customers churn at significantly higher rates
- **Tenure** is strongly negatively correlated with churn — longer customers are far less likely to leave
- **High monthly charges** combined with short tenure represent the highest-risk customer segment
- **SMOTE** improves minority class recall across all models with minimal accuracy trade-off
- **Engineered features** (`engagement_score`, `contract_risk`) rank among the top 15 predictors

---

## 📁 Dataset

**Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Size:** 7,043 rows × 21 features  
**Target:** `Churn` (Yes/No) — ~26.5% positive class

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
