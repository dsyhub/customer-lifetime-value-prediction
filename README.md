# Customer Lifetime Value Prediction

A probabilistic CLV model predicting **12-month revenue per customer** using the BG/NBD + Gamma-Gamma framework, validated with a 180-day temporal holdout, and operationalized into a 4-tier segmentation and campaign ROI tool.

**Data:** Google BigQuery [TheLook E-Commerce](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce) public dataset (~30K customers)

---

## Why CLV Over Binary Churn

| Dimension | Binary Churn Model | This CLV Model |
|-----------|-------------------|----------------|
| Output | Will they buy again? (yes/no) | How much will they spend? ($) |
| Business use | Retention trigger | Budget allocation + campaign ROI |
| Baseline difficulty | PR-AUC ≈ 0.16 (barely above naive) | Probabilistic with interpretable params |
| Segmentation | Binary risk tier | 4 tiers with dollar-denominated value |

---

## Technical Approach

### Framework: BG/NBD + Gamma-Gamma

The **Beta-Geometric / Negative Binomial Distribution (BG/NBD)** model predicts future purchase frequency by jointly modeling:
- Each customer's latent purchase rate (Poisson process with gamma heterogeneity)
- Dropout probability (beta-geometric process)

The **Gamma-Gamma** model predicts average transaction value, conditional on being an active buyer.

**CLV = predicted_purchases_12m × expected_avg_spend**

### Key Design Decisions

1. **One-time buyers (~88%) are NOT filtered out** — BG/NBD handles them natively; their low CLV output is the correct, informative answer
2. **Gamma-Gamma fallback for one-time buyers** — use observed `avg_order_value` rather than exclude them from CLV calculation
3. **Undiscounted CLV** for main output (simpler stakeholder communication)
4. **Temporal holdout = 180 days** — long enough to capture meaningful re-purchase signal in fashion e-commerce
5. **XGBoost complement** — adds engagement signals (sessions, cart events) that the probabilistic model ignores; provides feature importance and a second CLV estimate

### Data Split

```
Calibration period   ← features computed here
▐████████████████████████▌  2025-04-04
                             ▐█████████▌  ← holdout validation
                                          2025-10-01 (cutoff)
```

---

## BG/NBD Inputs

| Feature | Definition |
|---------|------------|
| `frequency` | Repeat purchases = total_orders − 1 |
| `recency` | Days from first to last purchase (calibration period) |
| `T` | Customer age in days at calibration end |
| `monetary_value` | Avg revenue per repeat transaction |

---

## 4-Tier Segmentation

| Segment | Definition | Budget/Customer | Action |
|---------|------------|-----------------|--------|
| **High Value** | Top 20% CLV | $0 (organic) | Protect margin — no discounts |
| **Growing** | Middle 40% + p_alive ≥ 0.3 | $15 | Personalized offers |
| **At-Risk** | p_alive < 0.3 (any CLV) | $10 | Win-back campaign |
| **Low Value** | Bottom 40% + p_alive ≥ 0.3 | $2 (email) | Email-only |

---

## Validation Checklist

- [ ] BG/NBD converges (all params positive, reasonable SEs)
- [ ] Gamma-Gamma: freq/monetary correlation < 0.3
- [ ] Lift curve: top 20% CLV captures ≥50% of holdout revenue
- [ ] `plot_period_transactions` bars align with actuals
- [ ] One-time buyer CLV < repeat buyer CLV (sanity check)

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
01_data_extraction.ipynb   → data/raw/clv_data.csv
02_clv_bgnbd.ipynb         → models/bgnbd_model.pkl, models/gg_model.pkl
                              data/processed/clv_scored.csv
03_clv_validation.ipynb    → validation metrics + lift curve
04_clv_segmentation.ipynb  → models/xgb_clv_model.pkl
                              data/processed/clv_final.csv
```

### Streamlit Dashboard

```bash
streamlit run src/app.py
```

**Tab 1 — Portfolio Overview:** Segment breakdown (pie + bar charts), KPI cards, segment profile table

**Tab 2 — Single Customer:**
- Mode A: look up existing customer by ID
- Mode B: manual BG/NBD input entry → live CLV inference + segment classification

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   └── clv_data.csv           # BG/NBD inputs + holdout labels
│   └── processed/
│       ├── clv_scored.csv         # + p_alive, predicted purchases, CLV
│       └── clv_final.csv          # + segment, XGBoost CLV
├── models/
│   ├── bgnbd_model.pkl            # Fitted BG/NBD model
│   ├── gg_model.pkl               # Fitted Gamma-Gamma model
│   └── xgb_clv_model.pkl          # XGBoost CLV regressor
├── notebooks/
│   ├── 01_data_extraction.ipynb   # BigQuery → clv_data.csv
│   ├── 02_clv_bgnbd.ipynb         # BG/NBD + Gamma-Gamma fitting
│   ├── 03_clv_validation.ipynb    # Temporal holdout backtesting
│   └── 04_clv_segmentation.ipynb  # Segmentation + campaign ROI
├── sql/
│   └── clv_features.sql           # Parameterized BG/NBD feature SQL
├── src/
│   └── app.py                     # Streamlit CLV dashboard
└── requirements.txt
```

---

## References

- Fader, P.S., Hardie, B.G.S., & Lee, K.L. (2005). ["Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model.](https://www.jstor.org/stable/30036675) *Marketing Science*, 24(2), 275–284.
- Fader, P.S., & Hardie, B.G.S. (2013). [The Gamma-Gamma Model of Monetary Value.](http://www.brucehardie.com/notes/025/gamma_gamma.pdf)
- [lifetimes Python library](https://lifetimes.readthedocs.io/) — BG/NBD and Gamma-Gamma implementation
