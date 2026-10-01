"""Streamlit dashboard for the Superstore sales analysis."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.backend.data_cleaning import clean_superstore
from src.backend.kpis import calculate_kpis, monthly_kpis
from src.backend.modeling import train_models

DATA_PATH = ROOT / "data" / "raw" / "Sample - Superstore.csv"

st.set_page_config(page_title="Superstore Sales Dashboard", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="Loading and cleaning Superstore data…")
def load_data(path: str, modified: float) -> pd.DataFrame:
    # Include the modification time in Streamlit's cache key so changed CSVs reload.
    frame, _ = clean_superstore(path)
    return frame


@st.cache_resource(show_spinner="Training the profitability prediction model…")
def load_profitability_model(path: str, modified: float):
    """Train and cache the validated champion model for interactive scoring."""
    frame, _ = clean_superstore(path)
    models, metrics, _ = train_models(frame, random_state=42)
    champion_name = metrics.iloc[0]["model"]
    return models[champion_name], metrics, champion_name


st.title("Superstore Sales Performance")
st.caption("Revenue, profit, customer, and product trends from the Superstore dataset.")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Place the source CSV at data/raw/Sample - Superstore.csv and reload the app.")
    st.stop()

data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)

with st.sidebar:
    st.header("Filters")
    categories = sorted(data["category"].dropna().unique())
    regions = sorted(data["region"].dropna().unique())
    selected_categories = st.multiselect("Category", categories, default=categories)
    selected_regions = st.multiselect("Region", regions, default=regions)
    min_date = data["order_date"].min().date()
    max_date = data["order_date"].max().date()
    date_range = st.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered = data[data["category"].isin(selected_categories) & data["region"].isin(selected_regions)]
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    filtered = filtered[filtered["order_date"].ge(start) & filtered["order_date"].lt(end)]

if filtered.empty:
    st.warning("No records match these filters. Adjust the selections in the sidebar.")
    st.stop()

kpis = calculate_kpis(filtered)
cards = st.columns(5)
cards[0].metric("Revenue", f"${kpis['revenue']:,.0f}")
cards[1].metric("Profit", f"${kpis['profit']:,.0f}")
cards[2].metric("Profit margin", f"{kpis['profit_margin']:.1%}")
cards[3].metric("Orders", f"{kpis['orders']:,}")
cards[4].metric("Customers", f"{kpis['customers']:,}")

monthly = monthly_kpis(filtered)
left, right = st.columns(2)
with left:
    st.subheader("Monthly revenue and profit")
    trend = monthly.melt(id_vars="order_month", value_vars=["revenue", "profit"], var_name="measure", value_name="amount")
    fig = px.line(trend, x="order_month", y="amount", color="measure", markers=True,
                  labels={"order_month": "Order month", "amount": "USD", "measure": ""})
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Revenue by category")
    category = filtered.groupby("category", as_index=False).agg(revenue=("sales", "sum"), profit=("profit", "sum"))
    fig = px.bar(category, x="category", y="revenue", color="profit", color_continuous_scale="RdYlGn",
                 labels={"category": "Category", "revenue": "Revenue (USD)", "profit": "Profit (USD)"})
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Profit by sub-category")
    subcategory = filtered.groupby("sub_category", as_index=False).profit.sum().sort_values("profit")
    fig = px.bar(subcategory, x="profit", y="sub_category", orientation="h",
                 labels={"profit": "Profit (USD)", "sub_category": "Sub-category"})
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Sales and profit by region")
    region = filtered.groupby("region", as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"))
    region["profit_margin"] = region["profit"] / region["sales"]
    fig = px.bar(region, x="region", y="profit_margin", text="profit_margin",
                 labels={"region": "Region", "profit_margin": "Profit margin"}, text_auto=".1%")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.header("Order profitability prediction")
st.caption(
    "Estimate whether a proposed order line is likely to be profitable. "
    "This is decision support based on historical data, not an automatic pricing decision."
)

with st.expander("How this prediction works", expanded=False):
    st.markdown(
        "The model predicts **profit > 0** using order details available before the outcome: "
        "sales, quantity, discount, product hierarchy, region, segment, shipping mode, "
        "and order timing. Profit and profit margin are excluded to prevent leakage. "
        "The model is evaluated using a chronological holdout."
    )

model, model_metrics, champion_name = load_profitability_model(
    str(DATA_PATH), DATA_PATH.stat().st_mtime
)
champion_metrics = model_metrics.iloc[0]
metric_columns = st.columns(3)
metric_columns[0].metric("Champion", champion_name)
metric_columns[1].metric("Held-out ROC-AUC", f"{champion_metrics['Test ROC-AUC']:.3f}")
metric_columns[2].metric("Held-out F1", f"{champion_metrics['F1']:.3f}")

with st.form("profitability_prediction_form"):
    left, middle, right = st.columns(3)
    with left:
        sales = st.number_input("Sales value (USD)", min_value=0.01, value=250.00, step=10.00)
        quantity = st.number_input("Quantity", min_value=1, value=2, step=1)
        discount = st.select_slider(
            "Discount", options=sorted(data["discount"].dropna().unique()), value=0.0,
            format_func=lambda value: f"{value:.0%}"
        )
        category = st.selectbox("Category", sorted(data["category"].dropna().unique()))
        subcategory_options = sorted(data.loc[data["category"] == category, "sub_category"].dropna().unique())
        sub_category = st.selectbox("Sub-category", subcategory_options)
    with middle:
        region = st.selectbox("Region", sorted(data["region"].dropna().unique()))
        segment = st.selectbox("Customer segment", sorted(data["segment"].dropna().unique()))
        ship_mode = st.selectbox("Ship mode", sorted(data["ship_mode"].dropna().unique()))
    with right:
        order_date = st.date_input(
            "Proposed order date", value=data["order_date"].max().date(),
            min_value=data["order_date"].min().date()
        )
        st.info("A probability at or above 50% is classified as likely profitable.")
    submitted = st.form_submit_button("Predict profitability", type="primary")

if submitted:
    proposed_order = pd.DataFrame([{
        "sales": float(sales), "quantity": int(quantity), "discount": float(discount),
        "category": category, "sub_category": sub_category, "region": region,
        "segment": segment, "ship_mode": ship_mode,
        "order_month_num": order_date.month, "order_year": order_date.year,
    }])
    probability = float(model.predict_proba(proposed_order)[0, 1])
    likely_profitable = probability >= 0.50
    result_columns = st.columns(2)
    result_columns[0].metric("Probability of positive profit", f"{probability:.1%}")
    result_columns[1].metric(
        "Prediction", "Likely profitable" if likely_profitable else "Potential loss risk",
        delta="Above 50% decision threshold" if likely_profitable else "Below 50% decision threshold",
        delta_color="normal" if likely_profitable else "inverse",
    )
    if likely_profitable:
        st.success("The proposed order is predicted to be profitable. Confirm final pricing and costs before approval.")
    else:
        st.warning(
            "The proposed order has elevated loss risk. Review discount level, price, shipping choice, "
            "and product economics before approving the offer."
        )
    with st.expander("Prediction input used by the model"):
        st.dataframe(proposed_order, use_container_width=True, hide_index=True)

with st.expander("View filtered transaction data"):
    st.dataframe(filtered, use_container_width=True, hide_index=True)
