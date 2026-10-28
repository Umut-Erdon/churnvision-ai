from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_PATH = Path(__file__).with_name("data") / "telco_churn.csv"
TARGET, ID = "Churn", "customerID"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load_data(source=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(source)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["ChurnFlag"] = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["TenureGroup"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 100], labels=["0-12 ay", "13-24 ay", "25-48 ay", "49+ ay"])
    return df


def _pipeline(cat_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", NUMERIC),
    ])
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=250, l2_regularization=1.0, random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def train_and_score(df: pd.DataFrame):
    """Hold-out metrics + out-of-fold risk for every customer (no in-sample scoring)."""
    drop = {ID, TARGET, "ChurnFlag", "TenureGroup"}
    cat_cols = [c for c in df.columns if c not in drop and c not in NUMERIC]
    X, y = df[cat_cols + NUMERIC], df["ChurnFlag"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = _pipeline(cat_cols).fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_te, p)), "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred)), "recall": float(recall_score(y_te, pred)),
        "cm": confusion_matrix(y_te, pred), "n_test": len(y_te), "base_rate": float(y.mean()),
    }
    imp = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=4, random_state=42)
    importance = pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean}).sort_values("importance", ascending=False)
    oof = cross_val_predict(_pipeline(cat_cols), X, y, cv=5, method="predict_proba")[:, 1]
    scored = df.copy()
    scored["risk"] = oof
    scored["risk_level"] = pd.cut(oof, [-0.01, 0.3, 0.6, 1.01], labels=["Düşük", "Orta", "Yüksek"])
    return scored.sort_values("risk", ascending=False), metrics, importance
