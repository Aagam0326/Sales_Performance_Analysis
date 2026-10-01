"""Business KPI calculations for cleaned Superstore transactions."""
from __future__ import annotations
import pandas as pd


def calculate_kpis(frame: pd.DataFrame) -> dict:
    """Return core KPIs calculated from line-level transactions."""
    orders = frame.groupby("order_id", as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"),
                                                        customer_id=("customer_id", "first"), order_date=("order_date", "min"))
    revenue = frame["sales"].sum()
    profit = frame["profit"].sum()
    customer_orders = orders.groupby("customer_id")["order_id"].nunique()
    return {"revenue": float(revenue), "profit": float(profit),
            "profit_margin": float(profit / revenue) if revenue else float("nan"),
            "aov": float(orders["sales"].mean()), "orders": int(len(orders)),
            "customers": int(orders["customer_id"].nunique()),
            "repeat_customer_rate": float((customer_orders > 1).mean())}


def monthly_kpis(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly sales/profit series with month-over-month and year-over-year growth."""
    monthly = (frame.groupby("order_month", as_index=False)
               .agg(revenue=("sales", "sum"), profit=("profit", "sum"), orders=("order_id", "nunique"))
               .sort_values("order_month"))
    monthly["profit_margin"] = monthly["profit"] / monthly["revenue"]
    monthly["aov"] = monthly["revenue"] / monthly["orders"]
    monthly["mom_growth"] = monthly["revenue"].pct_change()
    monthly["yoy_growth"] = monthly["revenue"].pct_change(12)
    return monthly
