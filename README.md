# Sales Performance Analysis & Forecasting

A portfolio-grade Python analysis of the Kaggle Superstore dataset that translates transaction-level sales into commercial recommendations, a monthly forecast, and an order-profitability model.

## Business questions

- Which categories, products, customers, and geographies drive revenue and profit?
- When do discounts erode margin, and where should promotional controls tighten?
- How does demand vary through the year, and what will monthly sales be next?
- Can an order be flagged as likely profitable before fulfilment?

## Dataset and methodology

Source: [Superstore Dataset Final](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final). Download `Sample - Superstore.csv` into `data/raw/`; the source is read-only. The pipeline handles latin-1/CP1252 encoding, standardizes headers, parses dates, audits nulls/duplicates/IQR outliers, and writes only `data/processed/superstore_cleaned.csv`.

The notebooks form a narrative: cleaning, EDA, KPI dashboard, time-series forecasting, then leakage-safe classification and SHAP interpretation. Forecasting uses a chronological 12-month holdout and compares seasonal naive, additive Holt-Winters, and a pre-specified SARIMA(1,1,1)(1,1,1,12). Modeling uses chronological train/test data and time-series cross-validation; profit is excluded from predictors.

## Headline results

The executed pipeline processed 9,994 transactions (2014–2017), producing **$2,297,201 revenue**, **$286,397 profit**, and a **12.5% profit margin** across 5,009 orders. Aggregate margin becomes negative at the **30% discount** tier; Tables alone lost **$17,725**.

On the final 12 months held out chronologically, **Holt-Winters won** with **22.59% MAPE** and $12,541 RMSE—**7.9% lower MAPE** than the 24.52% seasonal-naive baseline. The order-profitability champion, **XGBoost**, achieved **0.982 ROC-AUC**, 0.996 average precision, and 0.963 F1 on the latest 20% holdout. All exact results are available in `reports/metrics.json` and the recommendations in `reports/insights.md`.

![Category performance](reports/figures/category_revenue_profit.png)
![Forecast](reports/figures/forecast_vs_actual.png)

The interactive regional-margin chart is saved as `reports/figures/regional_margin.html`.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Place the Kaggle CSV at data/raw/Sample - Superstore.csv
python -m src.run_analysis
jupyter nbconvert --to notebook --execute notebooks/01_data_loading_cleaning.ipynb --output executed_01.ipynb
```

Run each notebook in numeric order. They use the same reusable modules and expect the raw CSV path above.

## Repository layout

`src/` holds reusable, commented functions; `notebooks/` provides the portfolio narrative; `reports/figures/` contains exported charts; and `reports/insights.md` contains evidence-backed recommendations.

## Future work

Add external calendar/holiday and economic drivers, automate monthly retraining, monitor forecast drift, and validate promotion recommendations with controlled experiments.
