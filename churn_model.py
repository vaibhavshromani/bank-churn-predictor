"""Synthetic bank customer data + churn model training."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["age", "tenure_years", "balance", "num_products", "monthly_txns", "credit_score", "salary"]
CAT = ["geography", "has_credit_card", "is_active"]
MODEL_PATH = "model.joblib"


def make_data(n=8000, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "tenure_years": rng.integers(0, 15, n),
        "balance": rng.gamma(2.0, 40000, n).round(2),
        "num_products": rng.choice([1, 2, 3, 4], n, p=[.5, .4, .07, .03]),
        "monthly_txns": rng.poisson(18, n),
        "credit_score": rng.normal(650, 90, n).clip(300, 900).astype(int),
        "salary": rng.lognormal(12.2, 0.5, n).round(2),
        "geography": rng.choice(["North", "South", "East", "West"], n),
        "has_credit_card": rng.choice(["Yes", "No"], n, p=[.7, .3]),
        "is_active": rng.choice(["Yes", "No"], n, p=[.55, .45]),
    })
    z = (-2.2 + 0.03 * (df.age - 40) + 1.1 * (df.is_active == "No") + 0.9 * (df.num_products >= 3)
         - 0.06 * df.monthly_txns + 0.9 * (df.balance > 150000) - 0.004 * (df.credit_score - 650)
         + 0.5 * (df.geography == "East") - 0.05 * df.tenure_years)
    df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def train():
    df = make_data()
    X, y = df[NUM + CAT], df["churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pre = ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42))]).fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    report = classification_report(yte, pipe.predict(Xte), output_dict=True)
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
    imp.index = [i.split("__", 1)[1] for i in imp.index]
    bundle = {"model": pipe, "auc": auc, "report": report, "importance": imp, "churn_rate": float(y.mean())}
    joblib.dump(bundle, MODEL_PATH)
    return bundle


if __name__ == "__main__":
    b = train()
    print(f"ROC-AUC: {b['auc']:.3f} | churn rate: {b['churn_rate']:.1%}")
