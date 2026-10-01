"""Leakage-safe monthly-sales forecasting models and evaluation."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def monthly_sales(frame: pd.DataFrame) -> pd.Series:
    """Produce a complete monthly sales series, including zero-sales months."""
    series = frame.groupby("order_month")["sales"].sum().sort_index()
    return series.asfreq("MS", fill_value=0.0)


def temporal_split(series: pd.Series, test_months: int = 12) -> tuple[pd.Series, pd.Series]:
    if len(series) <= test_months + 24:
        raise ValueError("Need at least 24 training months plus the test horizon.")
    return series.iloc[:-test_months], series.iloc[-test_months:]


def seasonal_naive(train: pd.Series, horizon: int, season: int = 12) -> pd.Series:
    values = [train.iloc[-season + (i % season)] for i in range(horizon)]
    return pd.Series(values, index=pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS"))


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict:
    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    mape = (np.abs((aligned.actual - aligned.predicted) / aligned.actual.replace(0, np.nan)).dropna().mean() * 100)
    return {"MAPE": float(mape), "RMSE": float(mean_squared_error(aligned.actual, aligned.predicted) ** 0.5)}


def fit_forecasts(series: pd.Series, test_months: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Compare seasonal naive, additive Holt-Winters, and SARIMA(1,1,1)(1,1,1,12).

    The parsimonious SARIMA specification uses one regular and one seasonal
    difference to address trend/annual seasonality; its orders are kept fixed
    before testing, avoiding test-set tuning.
    """
    train, test = temporal_split(series, test_months)
    base = seasonal_naive(train, len(test))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12,
                                  initialization_method="estimated").fit(optimized=True)
        hw_pred = hw.forecast(len(test))
        sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        result = sarima.get_forecast(len(test))
    sarima_pred = result.predicted_mean
    comparison = pd.DataFrame([{"model": "Seasonal naive", **evaluate(test, base)},
                               {"model": "Holt-Winters", **evaluate(test, hw_pred)},
                               {"model": "SARIMA", **evaluate(test, sarima_pred)}]).sort_values("MAPE")
    forecast = pd.DataFrame({"actual": test, "seasonal_naive": base, "holt_winters": hw_pred,
                             "sarima": sarima_pred})
    ci = result.conf_int()
    forecast["lower_ci"] = ci.iloc[:, 0]
    forecast["upper_ci"] = ci.iloc[:, 1]
    return comparison, forecast, sarima
