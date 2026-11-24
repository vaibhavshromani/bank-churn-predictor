# 📉 Bank Customer Churn Predictor

A compact ML project: predict which bank customers are likely to leave and suggest retention actions.

## Features
- Random Forest classifier in a Scikit-learn Pipeline (scaling + one-hot encoding)
- Class-imbalance handling with `class_weight="balanced"`
- Feature importance chart (what drives churn)
- Rule-based retention recommendations for the business team
- Streamlit UI

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The model trains automatically on first launch (a few seconds).

## Deploy free on Streamlit Community Cloud
1. Push this folder to GitHub.
2. At https://share.streamlit.io, click **New app**, select the repo, main file `app.py`, and deploy.

## Files
- `churn_model.py`: synthetic data generation and model training
- `app.py`: Streamlit app

## Notes
Data is synthetic. To use real data (e.g. Kaggle "Bank Customer Churn"), rename its columns to match `NUM` and `CAT` in `churn_model.py`.
