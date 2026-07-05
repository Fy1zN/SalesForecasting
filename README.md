# 📊 End-to-End Sales Forecasting & Demand Intelligence System

An internship capstone project that builds a complete retail demand-forecasting pipeline — from raw transactional data to a deployed, interactive business dashboard. The system predicts future product demand, detects anomalous sales patterns, segments products by demand behavior, and presents findings through a live Streamlit application.

**Live Dashboard:** [Add your Streamlit app link here]
**Notebook:** `analysis.ipynb`
**Business Report:** `summary.docx`

---

## 🎯 Problem Statement

Retail and e-commerce businesses depend on accurately answering one question: *how much of each product will sell next month, and is there enough stock to meet that demand?* Overstocking wastes capital and storage; understocking loses sales and customers.

This project builds a forecasting system covering time-series analysis, machine learning, anomaly detection, product segmentation, and deployment — the same category of system data science teams build and maintain in production.

---

## 🧠 What This Project Does

| Capability | Approach |
|---|---|
| Data exploration & feature engineering | Pandas, time-based feature extraction (year, month, week, quarter, season) |
| Time series decomposition & stationarity testing | `statsmodels` — trend/seasonal/residual decomposition, ADF test, differencing |
| Sales forecasting (3 models compared) | SARIMA, Facebook Prophet, XGBoost (lag + rolling-window features) |
| Segment-level forecasting | XGBoost applied to 3 product categories + 2 regions |
| Anomaly detection | Isolation Forest + Z-Score, cross-validated against each other |
| Product demand segmentation | K-Means clustering (Elbow Method) + PCA visualization |
| Deployment | 4-page interactive Streamlit dashboard |
| Business reporting | 2-page executive summary for non-technical stakeholders |

---

## 📈 Key Results

- **Best forecasting model:** XGBoost (MAE 17,720 / RMSE 19,906 / MAPE 18.0%), outperforming SARIMA and Prophet on this dataset.
- **Strongest seasonality:** November and December are consistently the highest-selling months every year, driven by holiday demand.
- **Most consistent regional growth:** the East region, with a year-over-year growth standard deviation of ~1.8% versus 25–37% for other regions.
- **21 anomalous weeks** identified via Isolation Forest, cross-checked with a Z-Score method; top spikes align with Black Friday / Cyber Monday periods.
- **3 demand clusters** identified across 17 product sub-categories, each with a tailored stocking strategy.

Full details, charts, and model diagnostics are documented in `analysis.ipynb`.

---

## 🗂️ Repository Structure

```
SalesForecasting_KrishMalhotra/
├── analysis.ipynb        # Full analysis: all 8 project tasks with explanations
├── app.py                 # Streamlit dashboard (4 pages)
├── requirements.txt        # Python dependencies
├── summary.docx            # 2-page executive business report
├── train.csv                # Superstore Sales dataset (primary data source)
├── vgsales.csv               # Video Game Sales dataset (multi-source merge exercise)
└── charts/                    # Exported chart images (.png)
```

---

## 🖥️ Dashboard Overview

The Streamlit app (`app.py`) has four pages:

1. **Sales Overview** — yearly/monthly sales trends, filterable by region and category
2. **Forecast Explorer** — pick a category or region, choose a forecast horizon, view XGBoost predictions vs. actuals with MAE/RMSE
3. **Anomaly Report** — visualizes detected anomalies on the weekly sales timeline with a supporting data table
4. **Product Demand Segments** — PCA-projected cluster visualization with stocking strategy per segment

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/Fy1zN/SalesForecasting.git
cd SalesForecasting

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The notebook (`analysis.ipynb`) can be opened in Jupyter, VS Code, or Google Colab to review the full analysis end-to-end.

---

## 🛠️ Tools & Libraries

Python · Pandas · NumPy · Statsmodels · Prophet · XGBoost · Scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit

---

## 📌 Data Sources

- [Superstore Sales Dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) — primary dataset, 4 years of daily retail sales
- [Video Game Sales Dataset](https://www.kaggle.com/datasets/gregorut/videogamesales) — used for a multi-source data merging exercise

---

## ⚠️ Known Limitations

- With only ~4 years of monthly history, all three forecasting models tend to underestimate the magnitude of extreme seasonal peaks (particularly November). Forecasts should be treated as a directional floor rather than a precise ceiling for holiday-season planning.
- The video game sales dataset has minimal record coverage after 2016, limiting the depth of the multi-source merge exercise — documented in the notebook as a data-quality finding rather than a business insight.

---

## 👤 Author

**Krish Malhotra**
Internship Project — Week 3 & 4
