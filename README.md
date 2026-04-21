# 🧠 Customer Churn Prediction using Machine Learning

This project builds an end-to-end **machine learning framework for predicting customer churn** using the Telco Customer dataset.  
The goal is to identify customers at risk of leaving and understand the key drivers behind churn using interpretable AI techniques.

---

## 🚀 Project Objectives

- Predict customer churn (binary classification problem)
- Handle class imbalance using SMOTE and class weighting
- Compare multiple machine learning models
- Build an interpretable and business-driven solution
- Identify key factors influencing churn using feature importance

---

## 📊 Dataset

- Telco Customer Churn dataset (Kaggle)
- Features include:
  - Customer demographics
  - Account information
  - Services used
  - Billing & contract details

---

## 🛠️ Feature Engineering

Key engineered features:

- Customer lifecycle segmentation (tenure groups)
- Revenue-based features (avg monthly spend, total charges groups)
- Behavioral features (number of services, engagement score)
- Interaction features (tenure × charges, contract × charges)
- Risk indicators (contract risk, senior-alone flag)

---

## 🤖 Models Used

### Baseline Models:
- Logistic Regression
- Decision Tree

### Advanced Models:
- Random Forest
- XGBoost

### Ensemble Method:
- Stacking (Random Forest + XGBoost → Logistic Regression)

---

## ⚖️ Handling Class Imbalance

- SMOTE (Synthetic Minority Oversampling Technique)
- Class weighting for baseline models
- Comparison between SMOTE vs non-SMOTE approaches

---

## 📈 Evaluation Metrics

Due to class imbalance, multiple metrics were used:

- F1-score (primary metric)
- ROC-AUC
- Precision / Recall
- PR-AUC

> “Given the class imbalance, F1-score and ROC-AUC were used as primary evaluation metrics.”

---

## 🧠 Explainability (XAI)

Feature importance analysis revealed:

### Key Drivers of Churn:
- Contract type (Month-to-month is highest risk)
- Monthly charges
- Customer tenure
- Service usage (OnlineSecurity, TechSupport)
- Engagement score

---

## 🏆 Key Results

- Logistic Regression achieved highest ROC-AUC (~0.85)
- XGBoost provided best balance between precision and recall
- SMOTE had limited impact on performance improvement
- Feature importance confirmed strong business interpretability

---

## 💡 Business Insights

- Customers with **month-to-month contracts** are significantly more likely to churn
- High monthly charges combined with low tenure increase churn risk
- Lack of value-added services increases customer attrition
- Improving onboarding and service adoption can reduce churn

---

## 📌 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib


---

## 🚀 Future Work

- SHAP-based explainability (local + global)
- Hyperparameter tuning (Optuna)
- Deployment as churn prediction API
- Dashboard for business users

---

## 👨‍💻 Author

Data Science & Machine Learning Project  
Focused on **business-driven AI and predictive analytics**
