# Customer Lifetime Value Prediction

A two-stage CLV model predicting **12-month revenue per customer** using purchase propensity classification + spend-tier expected revenue, validated with a 183-day temporal holdout, and operationalized into a 4-tier segmentation and campaign ROI tool.

**Data:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — real UK e-commerce transactions (Dec 2009 – Dec 2011, ~4,900 customers after cleaning)

---

## Technical Approach

### Framework: Two-Stage Purchase Propensity + CLV

**Stage 1 — Purchase Propensity (Classification):**
XGBoost classifier (tuned with Optuna, 50 trials) predicting whether a customer will purchase in the holdout window. Handles class imbalance with `scale_pos_weight` + isotonic probability calibration. Selected via 4-model comparison (Logistic Regression, Random Forest, XGBoost, LightGBM).

**Stage 2 — Expected Revenue (Spend Tiers):**
Customers are binned by historical `monetary_value` into spend tiers (terciles). Each tier maps to the average holdout revenue per buyer within that tier. Individual-level regression was attempted but produced negative R² — revenue per transaction is inherently noisy.

**CLV = P(purchase) x E[revenue | purchase]**

### Key Design Decisions

1. **One-time buyers (~31%) are NOT filtered out** — they receive low `p_purchase` scores, which is the correct signal
2. **Spend-tier revenue over individual regression** — tier averages provide meaningful differentiation (Low: £527, Mid: £940, High: £3,138) without overfitting
3. **Isotonic probability calibration** — `scale_pos_weight` distorts raw probabilities; calibration restores them to match true positive rates
4. **Temporal holdout = 183 days** — long enough to capture meaningful re-purchase signal in retail e-commerce
5. **Undiscounted CLV** for main output (simpler stakeholder communication)
6. **4-model comparison** — evaluated LR, RF, XGBoost, LightGBM before selecting XGBoost + Optuna tuning

### Data Split

```
Calibration period   <- features computed here
▐████████████████████████▌  2011-06-09
                             ▐█████████▌  <- holdout validation
                                          2011-12-09 (cutoff)
```

---

## Model Performance

| Metric | Test Set |
| ------ | -------- |
| PR-AUC | 0.87 |
| ROC-AUC | 0.85 |
| Positive rate | 52% |

**Revenue calibration:** Predicted total CLV within 0.3% of actual holdout revenue (ratio: 1.00).

**Spend tier differentiation:** CV = 91.4% across tiers — the two-stage design produces meaningful revenue separation.

---

## Model Inputs

| Feature Group | Features |
| ------------- | -------- |
| Purchase History | `frequency`, `recency`, `T`, `monetary_value`, `total_orders`, `avg_order_value`, `days_since_last_order` |
| Shopping Behavior | `unique_products`, `avg_basket_size`, `purchase_regularity`, `cancellation_rate`, `days_active` |
| Geography | `country` |
| Derived | `recency_ratio` (recency / T) |

---

## 4-Tier Segmentation

| Segment | Definition | Count | Budget/Customer | Action |
| ------- | ---------- | ----- | --------------- | ------ |
| **High Value** | Top 20% CLV | 984 (20%) | $0 (organic) | Protect margin — no discounts |
| **Growing** | Middle 40% + p_purchase >= 0.20 | 1,850 (38%) | $15 | Personalized offers |
| **At-Risk** | p_purchase < 0.20 (any CLV) | 957 (19%) | $10 | Win-back campaign |
| **Low Value** | Bottom 40% + p_purchase >= 0.20 | 1,127 (23%) | $2 (email) | Email-only |

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Notebook Pipeline

Run notebooks in order:

```
01_data_extraction.ipynb       -> data/raw/clv_data.csv
02_purchase_propensity.ipynb   -> models/purchase_propensity_model.pkl
                                  data/processed/stage1_scored.csv
03_clv_regression.ipynb        -> data/processed/clv_scored.csv
04_clv_validation.ipynb        -> validation metrics + lift curve
05_clv_segmentation.ipynb      -> data/processed/clv_final.csv
```

### Streamlit Dashboard

```bash
streamlit run src/app.py
```

**Tab 1 — Portfolio Overview:** Segment breakdown (pie + bar charts), KPI cards, segment profile table

**Tab 2 — Single Customer:**

- Mode A: look up existing customer by ID
- Mode B: manual feature entry → live CLV inference + segment classification

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   ├── clv_data.csv               # RFM + transaction + holdout labels
│   │   └── online_retail_II.xlsx       # UCI source data
│   └── processed/
│       ├── stage1_scored.csv          # + p_purchase
│       ├── clv_scored.csv             # + spend_tier, expected_revenue, clv_180d, clv_12m
│       └── clv_final.csv              # + segment assignments
├── models/
│   ├── purchase_propensity_model.pkl  # Calibrated XGBoost classifier
│   └── label_encoders.pkl             # LabelEncoders for categorical features
├── notebooks/
│   ├── 01_data_extraction.ipynb       # UCI data → clv_data.csv
│   ├── 02_purchase_propensity.ipynb   # Stage 1: 4-model comparison + Optuna tuning
│   ├── 03_clv_regression.ipynb        # Stage 2: spend-tier revenue + combined CLV
│   ├── 04_clv_validation.ipynb        # Temporal holdout backtesting
│   ├── 05_clv_segmentation.ipynb      # 4-tier segmentation + campaign ROI
│   └── archive/                       # Legacy BG/NBD notebooks (for reference)
├── sql/
│   └── archive/                       # Legacy TheLook SQL (for reference)
├── src/
│   └── app.py                         # Streamlit CLV dashboard
└── requirements.txt
```

---

## Limitations

- **Dataset age:** UCI Online Retail II covers Dec 2009 – Dec 2011. E-commerce behavior has shifted significantly since then (mobile, social commerce, subscription models). The methodology is transferable but the specific model coefficients would not generalize to modern data.
- **No engagement data:** Unlike modern e-commerce platforms, this dataset lacks browsing behavior, session data, cart abandonment, and email engagement signals. These features typically improve purchase propensity models.
- **UK-centric:** ~90% of transactions are from the UK. Country-level features have limited variation, and the model may not generalize well to geographically diverse customer bases.
- **B2B component:** The dataset contains a mix of retail and wholesale (bulk) transactions. Some "customers" are businesses placing large orders, which inflates monetary values and complicates segmentation.
- **No discount/promotion data:** Campaign ROI projections assume a fixed cost structure. Without actual promotion response data, the ROI estimates are illustrative rather than actionable.
- **Annualization assumption:** 12-month CLV is extrapolated from a 183-day holdout window using a linear scaling factor (365/183). This assumes stationarity in purchase behavior across seasons.

---

## References

- Chen, T., & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System.](https://arxiv.org/abs/1603.02754) _KDD 2016_.
- Fader, P.S., Hardie, B.G.S., & Lee, K.L. (2005). ["Counting Your Customers" the Easy Way.](https://www.jstor.org/stable/30036675) _Marketing Science_, 24(2), 275-284. (Background on BG/NBD; see `notebooks/archive/` for initial exploration.)
- Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository — Online Retail II.](https://archive.ics.uci.edu/dataset/502/online+retail+ii) University of California, Irvine.
