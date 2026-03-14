# Customer Lifetime Value Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-006600)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.51-blueviolet)

Two-stage machine learning pipeline that predicts which e-commerce customers will purchase again, how much they'll spend, and segments them into actionable marketing tiers — validated against a 183-day holdout window.

---

## Key Results

- **Top 20% of predicted CLV captures 68.8% of actual holdout revenue** — the model correctly identifies the customers who matter most
- **Brier score 0.1775** on held-out test data (29% reduction vs. naive baseline), confirming calibrated probabilities feed directly into reliable CLV estimates
- **Revenue calibration ratio of 0.895** — total predicted CLV undershoots actual holdout revenue by 10.5%, a conservative and operationally safe bias
- **4,918 customers scored and segmented into 4 tiers** with differentiated campaign budgets and break-even lift thresholds ready for A/B testing

---

## Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                        UCI Online Retail II                           │
│                   1M rows → 777K clean → 4,918 customers              │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                    Temporal split: 2011-06-09
                    Calibration (69%) │ Holdout (31%)
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│   STAGE 1: Purchase      │          │   STAGE 2: Expected      │
│   Propensity             │          │   Revenue                │
│                          │          │                          │
│  Calibrated XGBoost      │          │  Spend-tier terciles     │
│  (Optuna 50 trials,      │          │  via pooled daily rates  │
│   isotonic calibration)  │          │                          │
│                          │          │  Low:  $402/customer     │
│  PR-AUC: 0.8358          │          │  Mid:  $851/customer     │
│  Brier:  0.1775          │          │  High: $2,866/customer   │
│                          │          │                          │
│  Output: P(purchase)     │          │  Output: E[revenue]      │
└────────────┬─────────────┘          └────────────┬─────────────┘
             │                                     │
             └────────────────┬────────────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │   CLV = P(purchase)      │
                 │       × E[revenue]       │
                 │                          │
                 │   Annualized: ×365/183   │
                 └────────────┬─────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │   4-TIER SEGMENTATION    │
                 │                          │
                 │  High Value (20%)  → $5  │
                 │  Growing   (38%)   → $15 │
                 │  At-Risk   (14.5%) → $10 │
                 │  Low Value (27.5%) → $2  │
                 └──────────────────────────┘
```

**Why two stages?** With only ~2,500 buyers in the calibration window, individual-level revenue regression is unreliable. Spend-tier group averages are more stable and interpretable, yielding a 0.895 revenue calibration ratio against the holdout.

---

## Dashboard

> **Screenshot placeholder** — _add dashboard screenshot here_

Launch with `streamlit run src/app.py`

| Tab                      | What it shows                                                                                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Executive Summary**    | KPI cards (total customers, predicted revenue, top-20% capture, Brier score), segment distribution, revenue-by-segment chart, segment profiles table |
| **Customer Explorer**    | Look up any customer by ID or score a new customer with manual feature entry. SHAP waterfall shows top-5 prediction drivers with direction labels.   |
| **Campaign Sensitivity** | Per-segment budget sliders, break-even incremental lift table, ROI sensitivity chart with break-even dots — no fabricated conversion rates           |

---

## Project Structure

```
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb      # Data cleaning, temporal split, 13 features
│   ├── 02_purchase_propensity_model.ipynb      # 4-model comparison, Optuna tuning, SHAP
│   └── 03_customer_lifetime_value_segmentation.ipynb  # CLV computation, holdout validation
├── src/
│   └── app.py                                  # Streamlit dashboard (4 tabs)
├── models/
│   ├── purchase_propensity_model.pkl           # Calibrated XGBoost (isotonic, joblib)
│   └── label_encoders.pkl                      # Country LabelEncoder dict
├── data/
│   ├── raw/
│   │   └── online_retail_II.xlsx               # UCI source (not committed — see below)
│   └── processed/
│       ├── clv_data.csv                        # 4,918 × 17 — features from NB01
│       ├── stage1_scored.csv                   # 4,918 × 20 — with P(purchase) from NB02
│       └── clv_final.csv                       # 4,918 × 26 — CLV + segments from NB03
├── requirements.txt
└── .streamlit/config.toml                      # Theme and server config
```

---

## How to Run

**1. Clone and install**

```bash
git clone https://github.com/dsyhub/customer-lifetime-value-prediction.git
cd customer-lifetime-value-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Download the dataset**

Download `online_retail_II.xlsx` from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it in `data/raw/`.

**3. Run notebooks in order**

```
notebooks/01_exploratory_data_analysis.ipynb        → outputs data/processed/clv_data.csv
notebooks/02_purchase_propensity_model.ipynb        → outputs models/*.pkl, data/processed/stage1_scored.csv
notebooks/03_customer_lifetime_value_segmentation.ipynb → outputs data/processed/clv_final.csv
```

**4. Launch the dashboard**

```bash
streamlit run src/app.py
```

> Processed data and models are committed to the repo, so you can skip steps 2-3 and go straight to the dashboard.

---

## Methodology Highlights

### Two-stage evaluation philosophy

Model selection and Optuna tuning optimize **PR-AUC** (discrimination — can the model rank customers?) via 3-fold CV on training data only. Final evaluation uses **Brier score** (calibration — are the probabilities accurate?) because CLV is a dollar-weighted expectation: `P(purchase) × E[revenue]`. A well-ranked but miscalibrated model produces wrong CLV estimates. Isotonic calibration (5-fold) bridges the gap.

### Why Brier score over log loss

Log loss penalizes confident wrong predictions exponentially, which is useful during training (it's XGBoost's implicit loss). But for CLV, we care about **average probability accuracy across the portfolio**, not worst-case confidence. Brier score measures mean squared error of probabilities — directly interpretable as "how far off are our purchase probability estimates, on average?" The test-set Brier of 0.1775 vs. naive baseline of 0.2496 means the model reduces probability error by 29% (Brier Skill Score).

### Segmentation cascade and the At-Risk tradeoff

Segments are assigned via priority-ordered rules, not simple CLV quartiles:

1. **High Value** — top 20% by CLV regardless of purchase probability (protect your best customers)
2. **At-Risk** — P(purchase) < 0.20 _and_ not High Value (low engagement signal)
3. **Growing** — middle 40% by CLV with P(purchase) ≥ 0.20
4. **Low Value** — bottom 40% by CLV with P(purchase) ≥ 0.20

The 0.20 threshold pools 713 customers (14.5%) into At-Risk. Sensitivity analysis shows this is a deliberate choice: lowering to 0.10 captures only 230 customers (too few for a campaign), raising to 0.30 dilutes the segment with 1,109 customers who have reasonable engagement.

### Campaign sensitivity instead of fabricated conversion rates

Rather than guessing "this campaign will convert 5% of At-Risk customers," the dashboard computes **break-even incremental lift** per segment. For example, a $10/customer At-Risk campaign breaks even at 4.16% incremental lift (1 in 24 customers). This gives marketers a concrete threshold to evaluate against their own A/B test data.

---

## Limitations & Next Steps

### Limitations

- **Revenue tiers use historical spending levels.** Customers who shift spending behavior over time land in the wrong tier. A regression model would help but requires a larger buyer base than the current ~2,500 to avoid overfitting.
- **Linear annualization assumes stable purchase rates.** The 183-day holdout (Jun-Dec) includes holiday season, so annualized CLV likely overstates non-holiday months and understates holiday concentration.
- **B2B contamination.** Bulk/wholesale orders inflate high-end revenue tiers. Separating B2C from B2B records would improve precision.
- **Single-market bias.** 91.6% of customers are UK-based. The model may not generalize to other geographies without recalibration.

### Next steps

- Replace spend-tier averages with a regression model as the customer base grows
- A/B test the campaign budgets per segment to measure true incremental lift and close the feedback loop
- Add seasonality-aware annualization (e.g., monthly CLV with seasonal adjustment factors)
- Separate B2B and B2C customers for tier-specific modeling

---

## Tech Stack

| Category       | Tools                                                                           |
| -------------- | ------------------------------------------------------------------------------- |
| ML & Tuning    | XGBoost, scikit-learn, Optuna (50-trial Bayesian search), LightGBM (comparison) |
| Explainability | SHAP (TreeExplainer, top-5 feature waterfall)                                   |
| Data           | pandas, NumPy, SciPy                                                            |
| Visualization  | Plotly (dashboard), matplotlib + seaborn (notebooks)                            |
| App            | Streamlit (cached data/model loading, custom theme)                             |

---

## Data Source

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — 1,067,371 transactions from a UK-based online retailer, Dec 2009 - Dec 2011.
