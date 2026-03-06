# Customer Lifetime Value Prediction

A two-stage CLV model predicting **12-month revenue per customer** using purchase propensity classification + spend-tier expected revenue, validated with a 180-day temporal holdout, and operationalized into a 4-tier segmentation and campaign ROI tool.

**Data:** Google BigQuery [TheLook E-Commerce](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce) public dataset (~52K customers)

---

## Why CLV Over Binary Churn

| Dimension           | Binary Churn Model                 | This CLV Model                          |
| ------------------- | ---------------------------------- | --------------------------------------- |
| Output              | Will they buy again? (yes/no)      | How much will they spend? ($)           |
| Business use        | Retention trigger                  | Budget allocation + campaign ROI        |
| Baseline difficulty | PR-AUC ≈ 0.16 (barely above naive) | Two-stage with interpretable components |
| Segmentation        | Binary risk tier                   | 4 tiers with dollar-denominated value   |

---

## Technical Approach

### Framework: Two-Stage Purchase Propensity + CLV

**Stage 1 — Purchase Propensity (Classification):**
XGBoost classifier predicting whether a customer will purchase in the holdout window. Handles class imbalance (12.3% positive) with `scale_pos_weight` + isotonic probability calibration.

**Stage 2 — Expected Revenue (Spend Tiers):**
Customers are binned by historical `monetary_value` into spend tiers (terciles). Each tier maps to the average holdout revenue per buyer within that tier. Individual-level regression was attempted but produced negative R² — revenue per transaction is inherently noisy.

**CLV = P(purchase) × E[revenue | purchase]**

### Key Design Decisions

1. **One-time buyers (~69%) are NOT filtered out** — they receive low `p_purchase` scores, which is the correct signal
2. **Spend-tier revenue over individual regression** — R² = -0.03 on individual regression; tier averages provide meaningful differentiation without overfitting
3. **Isotonic probability calibration** — `scale_pos_weight` distorts raw probabilities; calibration restores them to match true positive rates
4. **Temporal holdout = 180 days** — long enough to capture meaningful re-purchase signal in fashion e-commerce
5. **Undiscounted CLV** for main output (simpler stakeholder communication)

### Data Split

```
Calibration period   <- features computed here
▐████████████████████████▌  2025-04-04
                             ▐█████████▌  <- holdout validation
                                          2025-10-01 (cutoff)
```

---

## Model Inputs

| Feature Group    | Features                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Purchase History | `frequency`, `recency`, `T`, `monetary_value`, `total_orders`, `avg_order_value`, `days_since_last_order`                 |
| Demographics     | `customer_tenure_days`, `age`, `gender`, `traffic_source`, `country`                                                      |
| Engagement       | `total_sessions`, `total_events`, `days_since_last_visit`, `avg_events_per_session`, `cart_events`, `product_view_events` |
| Derived          | `recency_ratio` (recency / T)                                                                                             |

---

## 4-Tier Segmentation

| Segment        | Definition                      | Budget/Customer | Action                        |
| -------------- | ------------------------------- | --------------- | ----------------------------- |
| **High Value** | Top 20% CLV                     | $0 (organic)    | Protect margin — no discounts |
| **Growing**    | Middle 40% + p_purchase >= 0.05 | $15             | Personalized offers           |
| **At-Risk**    | p_purchase < 0.05 (any CLV)     | $10             | Win-back campaign             |
| **Low Value**  | Bottom 40% + p_purchase >= 0.05 | $2 (email)      | Email-only                    |

---

## Validation Checklist

- [ ] Purchase propensity PR-AUC significantly above baseline (0.123)
- [ ] Calibration curve: predicted probabilities match actual rates
- [ ] Lift curve: top 20% CLV captures disproportionate holdout revenue
- [ ] One-time buyer CLV < repeat buyer CLV (sanity check)
- [ ] Total predicted CLV within reasonable range of actual holdout revenue

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

Requires a Google Cloud account with BigQuery access and the `GOOGLE_APPLICATION_CREDENTIALS` environment variable set.

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
- Mode B: manual feature entry -> live CLV inference + segment classification

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   └── clv_data.csv               # Purchase history + engagement + holdout labels
│   └── processed/
│       ├── stage1_scored.csv          # + p_purchase
│       ├── clv_scored.csv             # + spend_tier, expected_revenue, clv_180d, clv_12m
│       └── clv_final.csv              # + segment assignments
├── models/
│   ├── purchase_propensity_model.pkl  # Calibrated XGBoost classifier
│   └── label_encoders.pkl             # LabelEncoders for categorical features
├── notebooks/
│   ├── 01_data_extraction.ipynb       # BigQuery -> clv_data.csv
│   ├── 02_purchase_propensity.ipynb   # Stage 1: purchase propensity classifier
│   ├── 03_clv_regression.ipynb        # Stage 2: spend-tier revenue + combined CLV
│   ├── 04_clv_validation.ipynb        # Temporal holdout backtesting
│   ├── 05_clv_segmentation.ipynb      # 4-tier segmentation + campaign ROI
│   └── archive/                       # Legacy BG/NBD notebooks (for reference)
├── sql/
│   └── clv_features.sql               # Parameterized feature extraction SQL
├── src/
│   └── app.py                         # Streamlit CLV dashboard
└── requirements.txt
```

---

## References

- Chen, T., & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System.](https://arxiv.org/abs/1603.02754) _KDD 2016_.
- Fader, P.S., Hardie, B.G.S., & Lee, K.L. (2005). ["Counting Your Customers" the Easy Way.](https://www.jstor.org/stable/30036675) _Marketing Science_, 24(2), 275-284. (Background on BG/NBD; see `notebooks/archive/` for initial exploration.)
