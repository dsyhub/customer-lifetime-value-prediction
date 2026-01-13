# High-Value Customer Churn Prediction

## Obejctive

Identify customers at risk of leaving (chruning) in the next 30 days to target them with retention campaigns.

## Tech Stack

- **SQL (BigQuery):** Feature engineering and data extraction.
- **Python (Pandas, XGBoost):** Data processing and predictive modeling.
- **Streamlit:** Interactive dashboard for stakeholders.

## Structure

```
ecommerce-churn-prediction/
│
├── README.md              # The "Cover Letter" of my project (Executive Summary)
├── .gitignore
├── requirements.txt       # List of Python libraries (pandas, xgboost, etc.)
│
├── sql/                   # Store SQL queries here
│   └── feature_engineering.sql
│
├── data/                  # Local data storage (Git will ignore this folder)
│   ├── raw/               # The CSV you download from BigQuery
│   └── processed/         # Cleaned data ready for modeling
│
├── notebooks/             # Jupyter Notebooks
│   ├── 01_data_extraction.ipynb
│   ├── 02_eda_and_modeling.ipynb
│   └── 03_evaluation_and_insights.ipynb
│
└── src/                   # (Optional) Python scripts for the Streamlit app later
    └── app.py
```
