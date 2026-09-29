"""Order-profitability classification and explainability utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


FEATURES = ["sales", "quantity", "discount", "category", "sub_category", "region", "segment",
            "ship_mode", "order_month_num", "order_year"]


def prepare_model_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build predictors available at order time; profit is target only, preventing leakage."""
    data = frame.sort_values("order_date").copy()
    data["order_month_num"] = data.order_date.dt.month
    data["order_year"] = data.order_date.dt.year
    return data[FEATURES], (data.profit > 0).astype(int)


def _pipeline(model: object) -> Pipeline:
    categorical = [c for c in FEATURES if c in {"category", "sub_category", "region", "segment", "ship_mode"}]
    numeric = [c for c in FEATURES if c not in categorical]
    processor = ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric),
                                   ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                                     ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    return Pipeline([("preprocess", processor), ("model", model)])


def train_models(frame: pd.DataFrame, random_state: int = 42) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Chronologically split data (latest 20% held out), cross-validate training data."""
    X, y = prepare_model_data(frame)
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    models = {"Random Forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=3, class_weight="balanced",
                                                        random_state=random_state, n_jobs=-1),
              "XGBoost": XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, eval_metric="logloss", random_state=random_state,
                                         n_jobs=1)}
    metrics, fitted = [], {}
    cv = TimeSeriesSplit(n_splits=3)
    for name, estimator in models.items():
        pipe = _pipeline(estimator)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1).mean()
        pipe.fit(X_train, y_train)
        probabilities = pipe.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics.append({"model": name, "CV ROC-AUC": cv_auc, "Test ROC-AUC": roc_auc_score(y_test, probabilities),
                        "Average precision": average_precision_score(y_test, probabilities), "F1": f1_score(y_test, predictions),
                        "Accuracy": accuracy_score(y_test, predictions)})
        fitted[name] = pipe
    test_data = X_test.copy()
    test_data["target"] = y_test
    return fitted, pd.DataFrame(metrics).sort_values("Test ROC-AUC", ascending=False), test_data
