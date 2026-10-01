# ✈️ Airline Delay Prediction System

A machine learning project that predicts whether a flight will be delayed, built on historical airline operational data using an **XGBoost classifier** — with an interactive **Streamlit** app for live predictions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Demo](#️-demo)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Model](#-model)
- [Current Performance](#-current-performance)
- [Streamlit App](#️-streamlit-app)
- [Roadmap](#-roadmap)
- [Author](#-author)
- [License](#-license)

---

## 📌 Project Overview

Flight delays are costly for airlines and frustrating for passengers. This project analyzes airline operational data — carrier, route, scheduled departure time, distance, and day of week — to predict the likelihood of a delay before it happens.

**Current status:** core pipeline and model are working end-to-end; performance tuning and explainability are in active progress (see [Roadmap](#-roadmap)).

The project covers:

- Data cleaning and feature engineering
- Exploratory Data Analysis (EDA)
- XGBoost model training and hyperparameter tuning
- Model evaluation with visualizations
- An interactive Streamlit web app for live predictions

---

## 🖼️ Demo

| Dashboard | Prediction Output |
|---|---|
| ![Dashboard](images/dashboard.png) | ![Prediction](images/prediction.png) |


---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| Model Serialization | Joblib / Pickle |
| Notebook Environment | Jupyter |

---

## 📂 Project Structure

```
Airline_Delay_Predictor/
│
├── data/
│   ├── raw_data/
│   ├── raw_data_documentation.txt
│   ├── train_sets_documentation.txt
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── airline_delay_model.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── outputs/
│   ├── best_params.json
│   ├── metrics.json
│   ├── model_evaluation.png
│   └── tuning_summary.json
│
├── images/
│   ├── dashboard.png
│   └── prediction.png
│
├── src/
│   ├── backend/
│   └── frontend/
│       └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset

The dataset contains airline operational records, including:

- Airline carrier
- Departure & arrival airport
- Scheduled departure time
- Distance traveled
- Day of week
- Flight delay status (target)

**Target variable**

- `0` → On-time flight
- `1` → Delayed flight

---

## 🔍 Exploratory Data Analysis

EDA was used to:

- Identify delay patterns across carriers, routes, and times
- Analyze feature distributions
- Detect missing values
- Study correlations between features
- Understand class imbalance (delays are the minority class)

![Delay Distribution by Carrier](images/eda_carrier_delays.png)

Full notebook: [`notebooks/eda.ipynb`](notebooks/eda.ipynb)

---

## 🤖 Model

**XGBoost Classifier** — chosen for:

- Strong performance on tabular data
- Fast training
- Built-in handling of imbalanced classes (`scale_pos_weight`)
- Good baseline accuracy with room to tune

### Training pipeline

1. Data cleaning
2. Feature engineering
3. Train/test split
4. Categorical encoding
5. Model training
6. Hyperparameter tuning
7. Evaluation
8. Model saving

---

## 📈 Current Performance

| Metric | Score |
|---|---|
| Accuracy | 72% |
| Precision (Delayed) | 35% |
| Recall (Delayed) | 58% |
| F1-score (Delayed) | 44% |

```
          precision    recall  f1-score   support

 On-time       0.88      0.75      0.81
 Delayed       0.35      0.58      0.44

accuracy                           0.72
```

![Model Evaluation — Confusion Matrix & ROC Curve](outputs/model_evaluation.png)

**Honest take:** accuracy looks fine, but the model is currently better at catching on-time flights than delays — precision on the "Delayed" class is the main thing being worked on next, since false positives are a real cost in deployment. Class imbalance handling and feature engineering are the planned next steps to improve this (see [Roadmap](#-roadmap)).

Additional evaluation artifacts (tuning summary, best params) are saved in [`outputs/`](outputs/).

---

## 🖥️ Streamlit App

The app lets users:

- Enter flight details
- Get a delay probability prediction
- View model outputs and basic airline insights

![Streamlit App Screenshot](images/streamlit_app.png)

### Run it locally

```bash
git clone https://github.com/Aagam0326/Airline_Delay_Prediction.git
cd Airline_Delay_Predictor

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run src/frontend/app.py
```

---

## 🧭 Roadmap

Planned next, in priority order:

- [ ] Improve precision/recall on the "Delayed" class (class imbalance handling, feature engineering)
- [ ] Add SHAP explainability for feature-level interpretability
- [ ] Real-time flight data API integration
- [ ] Docker support
- [ ] Cloud deployment (Streamlit Community Cloud / Hugging Face Spaces)
- [ ] CI/CD pipeline
- [ ] MLflow experiment tracking

---

## 👤 Author

**Aagam Shah**

- GitHub: [@Aagam0326](https://github.com/Aagam0326)
- LinkedIn: [Aagam Shah](https://www.linkedin.com/in/aagam-shah-v322006/)
- Email: [Aagam Shah](aagamvshah0306@gmail.com)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

---

## ⭐ Acknowledgements

Built with Scikit-learn, XGBoost, Streamlit, and the broader open-source data science community.