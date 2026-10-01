# 📊 Sales Performance Analysis & Forecasting

An end-to-end sales analytics project — data cleaning, exploratory analysis, KPI dashboards, time-series forecasting, and a machine learning model with SHAP-based interpretability — built on the Superstore retail dataset.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%7C%20SARIMA-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-9cf.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Description](#-description)
- [Project Structure](#-project-structure)
- [Tech Used](#️-tech-used)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Models Used](#-models-used)
- [Performance](#-performance)
- [Roadmap](#-roadmap)
- [Author](#-author)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 📌 Description

Retail businesses generate rich transactional data, but the real value comes from turning it into decisions: which categories are actually profitable, where discounting is eroding margin, what next quarter's revenue will likely look like, and which orders are at risk of being unprofitable.

This project covers the full analytics pipeline:

- **Data cleaning** — type fixes, deduplication, outlier and validity checks
- **Exploratory Data Analysis** — category/region/segment breakdowns, seasonality, discount-vs-profit relationship
- **KPI dashboard** — revenue, profit margin, AOV, MoM/YoY growth, repeat-customer rate
- **Forecasting** — SARIMA and Holt-Winters models compared against a seasonal-naive baseline
- **ML + explainability** — Random Forest / XGBoost predicting order-level profitability, explained with SHAP

---

## 📂 Project Structure

```
sales-performance-analysis/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                   # original Superstore dataset (untouched)
│   └── processed/              # cleaned dataset
│
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_kpi_dashboard.ipynb
│   ├── 04_forecasting.ipynb
│   └── 05_ml_model_shap.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── kpis.py
│   ├── forecasting.py
│   ├── modeling.py
│   └── run_analysis.py
│
├── reports/
│   ├── figures/                # exported charts (used in this README)
│   ├── insights.md             # business recommendations backed by numbers
│   └── kpi_summary.csv
│
└── images/                     # standalone screenshots for the README
```

---

## 🛠️ Tech Used

| Category | Technologies |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Forecasting | Statsmodels (SARIMA, Holt-Winters) |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Notebook Environment | Jupyter |

---

## 📊 Dataset

**Source:** [Superstore Sales dataset (Kaggle)](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

Order-level retail transaction data covering roughly 4 years, including:

- Order and ship dates, ship mode
- Customer, segment, region, state, city
- Product category, sub-category, product name
- Sales, quantity, discount, profit

**Summary**

| Field | Description |
|---|---|
| `sales` | Revenue for the order line (USD) |
| `profit` | Profit for the order line (USD) — can be negative |
| `discount` | Discount applied, 0–1 |
| `category` / `sub_category` | Product classification |
| `segment` | Consumer / Corporate / Home Office |
| `region` | East / West / Central / South |

Engineered features added during cleaning: `order_year`, `order_month`, `order_year_month`, `order_quarter`, `profit_margin`, `fulfillment_days`.

---

## 🔍 Exploratory Data Analysis

Key questions explored in [`02_eda.ipynb`](notebooks/02_eda.ipynb):

- Which categories and sub-categories drive revenue vs. which are actually losing money
- How profit margin varies by region and segment
- Monthly/quarterly seasonality in sales
- The discount threshold at which average profit turns negative
- Top and bottom performing products
- Customer repeat-purchase rate

![Sales & Profit by Category](reports/figures/sales_profit_by_category.png)
![Discount vs. Profit](reports/figures/discount_vs_profit.png)

Full write-up of the five non-obvious, quantified insights is in [`reports/insights.md`](reports/insights.md).

---

## 🤖 Models Used

**Forecasting** (monthly revenue):
- Baseline: Seasonal Naive
- SARIMA(1,1,1)(1,1,1,12)
- Holt-Winters (triple exponential smoothing, additive trend + seasonality)
- Evaluated on a time-based holdout using MAPE and RMSE

**Classification** (order-line profitability — `profit > 0`):
- Random Forest Classifier
- XGBoost Classifier
- 5-fold stratified cross-validation, evaluated on ROC-AUC, precision, recall, F1
- SHAP TreeExplainer for feature-level interpretability (summary plot + dependence plots for discount and category)

---

## 📈 Performance

| Forecasting Model | MAPE | RMSE |
|---|---|---|
| Seasonal Naive (baseline) | *fill in after running `04_forecasting.ipynb`* | |
| SARIMA | | |
| Holt-Winters | | |

| Classification Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | *fill in after running `05_ml_model_shap.ipynb`* | | | |
| XGBoost | | | | |

![Forecast vs. Actual](reports/figures/forecast_vs_actual.png)
![SHAP Summary](reports/figures/shap_summary.png)

> Run notebooks 4 and 5 end to end and copy the printed metrics into the tables above — keeping real numbers here (not placeholders) is what makes this project credible on a resume.

---

## 🧭 Roadmap

- [ ] Fill in actual forecasting and classification metrics above
- [ ] Add a lightweight Streamlit dashboard for interactive KPI exploration
- [ ] Extend forecasting to per-category/per-region granularity
- [ ] Add a SQL version of the KPI queries (`sql/kpi_queries.sql`) for the analytics-engineering angle
- [ ] Deploy the KPI dashboard (Streamlit Community Cloud)
- [ ] Add automated tests for `src/` modules

---

## 👤 Author

**Aagam Shah**

- GitHub: [@Aagam0326](https://github.com/Aagam0326)
- LinkedIn: [Aagam Shah](https://www.linkedin.com/in/aagam-shah-v322006/)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

---

## ⭐ Acknowledgements

Dataset provided via Kaggle (Superstore Sales). Built with Pandas, Statsmodels, Scikit-learn, XGBoost, SHAP, and the broader open-source data science community.