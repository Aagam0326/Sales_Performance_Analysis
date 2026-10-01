"""Execute the complete reproducible analysis after the raw CSV is supplied."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.backend.data_cleaning import clean_superstore, save_cleaned_data
from src.backend.forecasting import fit_forecasts, monthly_sales
from src.backend.kpis import calculate_kpis, monthly_kpis
from src.backend.modeling import train_models

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _charts(data: pd.DataFrame, monthly: pd.DataFrame, forecast: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    category = data.groupby("category", as_index=False).agg(Sales=("sales", "sum"), Profit=("profit", "sum"))
    fig, ax = plt.subplots(figsize=(9, 5)); category.set_index("category").plot(kind="bar", ax=ax)
    ax.set(title="Revenue and Profit by Category", xlabel="Category", ylabel="USD"); _save(fig, "category_revenue_profit.png")
    subcat = data.groupby("sub_category", as_index=False).profit.sum().sort_values("profit")
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=subcat, y="sub_category", x="profit", ax=ax)
    ax.set(title="Profit by Sub-Category", xlabel="Profit (USD)", ylabel="Sub-Category"); _save(fig, "subcategory_profit.png")
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(monthly.order_month, monthly.revenue, marker="o")
    ax.set(title="Monthly Revenue", xlabel="Month", ylabel="Revenue (USD)"); _save(fig, "monthly_revenue.png")
    fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(data=data, x="discount", y="profit", scatter_kws={"alpha": .25}, line_kws={"color": "crimson"}, ax=ax)
    ax.set(title="Discount versus Profit", xlabel="Discount", ylabel="Profit (USD)"); _save(fig, "discount_profit.png")
    region = data.groupby("region", as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"))
    region["profit_margin"] = region["profit"] / region["sales"]
    interactive = px.bar(region, x="region", y="profit_margin", text_auto=".1%",
                         title="Interactive Profit Margin by Region", labels={"profit_margin": "Profit margin"})
    interactive.update_layout(template="plotly_white")
    interactive.write_html(FIGURES / "regional_margin.html", include_plotlyjs="cdn")
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(forecast.index, forecast.actual, label="Actual", marker="o"); ax.plot(forecast.index, forecast.sarima, label="SARIMA", marker="o")
    ax.fill_between(forecast.index, forecast.lower_ci, forecast.upper_ci, alpha=.2, label="95% interval")
    ax.set(title="SARIMA Forecast versus Actual Sales", xlabel="Month", ylabel="Sales (USD)"); ax.legend(); _save(fig, "forecast_vs_actual.png")


def _insights(data: pd.DataFrame, kpis: dict, comparison: pd.DataFrame, model_metrics: pd.DataFrame) -> str:
    by_discount = data.groupby("discount").agg(sales=("sales", "sum"), profit=("profit", "sum")); by_discount["margin"] = by_discount.profit / by_discount.sales
    negative_discounts = by_discount[by_discount.margin < 0]
    threshold = float(negative_discounts.index.min()) if not negative_discounts.empty else float("nan")
    losers = data.groupby("sub_category").profit.sum().sort_values().head(3)
    regions = data.groupby("region").agg(sales=("sales", "sum"), profit=("profit", "sum")); regions["margin"] = regions.profit / regions.sales
    best = comparison.iloc[0]; baseline = comparison.loc[comparison.model == "Seasonal naive"].iloc[0]
    improvement = (baseline["MAPE"] - best["MAPE"]) / baseline["MAPE"] * 100
    top_model = model_metrics.iloc[0]
    return f"""# Actionable insights

1. Revenue was **${kpis['revenue']:,.0f}** across **{kpis['orders']:,} orders**, while profit margin was **{kpis['profit_margin']:.1%}**. Use this as the executive baseline.
2. Repeat customers represented **{kpis['repeat_customer_rate']:.1%}** of customers. Target retention offers to one-time buyers before broad discounting.
3. The first observed discount tier with aggregate negative margin is **{threshold:.0%}**. Require margin review for promotions at or beyond that tier.
4. The three weakest sub-categories by total profit were **{', '.join(f'{name} (${value:,.0f})' for name, value in losers.items())}**. Reprice, bundle, or renegotiate these lines.
5. Regional margins varied from **{regions.margin.min():.1%}** to **{regions.margin.max():.1%}**. Allocate commercial attention to low-margin regions, not just high-revenue ones.
6. **{best.model}** won the held-out forecast comparison with **{best.MAPE:.2f}% MAPE**, a **{improvement:.1f}%** reduction versus seasonal naive ({baseline.MAPE:.2f}%). Use it for monthly planning.
7. The best profitability classifier was **{top_model.model}**, with held-out ROC-AUC **{top_model['Test ROC-AUC']:.3f}**. Use it as a decision-support flag, not an automatic pricing rule.
"""


def main() -> None:
    raw = ROOT / "data" / "raw" / "Sample - Superstore.csv"
    data, audit = clean_superstore(raw)
    save_cleaned_data(data, ROOT / "data" / "processed" / "superstore_cleaned.csv")
    kpi = calculate_kpis(data); monthly = monthly_kpis(data)
    comparison, forecast, _ = fit_forecasts(monthly_sales(data))
    models, model_metrics, _ = train_models(data)
    _charts(data, monthly, forecast)
    metrics = {"cleaning_audit": audit, "kpis": kpi, "forecasting": comparison.to_dict(orient="records"), "modeling": model_metrics.to_dict(orient="records")}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (ROOT / "reports" / "insights.md").write_text(_insights(data, kpi, comparison, model_metrics), encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
