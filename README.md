# 💳 Payment Fraud Detection & Risk Scoring Application

## Project Overview
This project is an **end-to-end machine learning application** for detecting fraudulent financial transactions and assessing transaction risk.

It combines a trained classification model with an **interactive Streamlit dashboard** that enables:
- Real-time fraud prediction
- Probability-based risk scoring
- Model interpretability through feature importance
- Exploratory fraud analytics

The application is designed to support **fraud monitoring and decision-making** in financial systems.

---

## Business Problem
Financial institutions face significant losses due to fraudulent transactions.  
The challenge is to:
- Detect fraud accurately
- Minimize false positives
- Provide interpretable risk scores for decision support

This application addresses these needs by combining predictive modeling with explainable analytics.

---

## Key Features

### 🔍 Fraud Prediction
- Upload a CSV file containing transaction data
- The model generates:
  - `fraud_prediction` (0 = Not Fraud, 1 = Fraud)
  - `fraud_probability`
  - `risk_level` (Low / Medium / High)
- Adjustable decision threshold via the Streamlit sidebar



### 📊 Visual Analytics
The dashboard provides:
- Top 10 feature importance visualization
- Fraud vs Non-Fraud distribution
- Risk-level distribution
- Confusion matrix and model evaluation metrics

These visualizations help users **understand model behavior and performance**.



## Machine Learning Approach
- Supervised classification model trained on historical transaction data
- Feature engineering to capture transaction patterns
- Model evaluation using:
  - Confusion matrix
  - Classification metrics (precision, recall, etc.)
- Threshold tuning to balance fraud detection and false positives



## Risk Scoring Logic
Fraud probabilities are mapped to risk categories:
- **High Risk** — high probability of fraud
- **Medium Risk** — moderate probability
- **Low Risk** — low probability

This enables **actionable decision-making**, not just binary predictions.



## Application Workflow
1. User uploads transaction data (CSV)
2. Data is processed and passed to the trained model
3. Fraud predictions and probabilities are generated
4. Results and analytics are displayed in an interactive dashboard



## Applications
- Fraud monitoring and investigation
- Risk assessment for financial transactions
- Model experimentation and threshold tuning
- Demonstration of applied ML in finance



## Tech Stack
- **Python** — data processing and modeling  
- **scikit-learn** — machine learning  
- **Streamlit** — interactive dashboard  
- **pandas / numpy** — data manipulation  
- **matplotlib / seaborn** — visualization  
