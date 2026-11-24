import os

import joblib
import pandas as pd
import streamlit as st

from churn_model import CAT, MODEL_PATH, NUM, train

st.set_page_config(page_title="Bank Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource(show_spinner="Training model on first run...")
def get_bundle():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else train()


def retention_actions(c: dict) -> list[str]:
    tips = []
    if c["is_active"] == "No":
        tips.append("Re-engage with a personalised offer or app notification (inactive customer).")
    if c["num_products"] >= 3:
        tips.append("Review product bundle; many products can signal fees or complexity frustration.")
    if c["monthly_txns"] < 8:
        tips.append("Low usage: offer cashback or rewards on everyday spending.")
    if c["balance"] > 150000:
        tips.append("High-balance customer: assign a relationship manager and offer premium rates.")
    if c["age"] >= 55:
        tips.append("Offer retirement planning and fixed-deposit options.")
    return tips or ["No specific red flags; maintain regular engagement."]


bundle = get_bundle()
model = bundle["model"]

st.title("📉 Bank Customer Churn Predictor")
st.caption("Predict which customers are likely to leave and what the bank can do about it.")
tab1, tab2 = st.tabs(["🔮 Predict", "📈 Model Insights"])

with tab1:
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Age", 18, 80, 45)
    tenure = c1.slider("Tenure (years)", 0, 15, 3)
    balance = c1.number_input("Account balance (₹)", 0.0, 2_000_000.0, 120000.0, step=5000.0)
    products = c2.selectbox("Number of products", [1, 2, 3, 4], index=2)
    txns = c2.slider("Transactions per month", 0, 60, 5)
    score = c2.slider("Credit score", 300, 900, 620)
    salary = c3.number_input("Annual salary (₹)", 0.0, 5_000_000.0, 250000.0, step=10000.0)
    geo = c3.selectbox("Region", ["North", "South", "East", "West"])
    card = c3.radio("Has credit card", ["Yes", "No"], horizontal=True)
    active = c3.radio("Active member", ["Yes", "No"], index=1, horizontal=True)

    if st.button("Predict churn risk", type="primary"):
        cust = dict(age=age, tenure_years=tenure, balance=balance, num_products=products, monthly_txns=txns,
                    credit_score=score, salary=salary, geography=geo, has_credit_card=card, is_active=active)
        p = float(model.predict_proba(pd.DataFrame([cust])[NUM + CAT])[0, 1])
        level = "🔴 High" if p >= 0.6 else "🟠 Medium" if p >= 0.35 else "🟢 Low"
        st.metric("Churn probability", f"{p:.1%}", level)
        st.progress(min(p, 1.0))
        st.subheader("Suggested retention actions")
        for t in retention_actions(cust):
            st.write(f"- {t}")

with tab2:
    a, b = st.columns(2)
    a.metric("ROC-AUC", f"{bundle['auc']:.3f}")
    b.metric("Churn rate in data", f"{bundle['churn_rate']:.1%}")
    st.subheader("What drives churn? (feature importance)")
    st.bar_chart(bundle["importance"].head(10))
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(bundle["report"]).T.round(3))
    st.caption("Data is synthetic. Replace `make_data()` in churn_model.py with a real dataset such as Kaggle's Bank Customer Churn.")
