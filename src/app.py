"""
CLV Dashboard — Customer Lifetime Value Predictor
Run from repo root: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_PATH = REPO_ROOT / "data" / "processed" / "clv_final.csv"
PROPENSITY_PATH = REPO_ROOT / "models" / "purchase_propensity_model.pkl"
ENCODERS_PATH = REPO_ROOT / "models" / "label_encoders.pkl"

# ---------------------------------------------------------------------------
# Segment config (colours + recommended actions)
# ---------------------------------------------------------------------------
SEGMENT_CONFIG = {
    "High Value": {
        "color": "#2196F3",
        "action": "VIP loyalty — no discounts, protect margin",
        "icon": "💎",
    },
    "Growing": {
        "color": "#4CAF50",
        "action": "Personalized offer — growth potential justifies investment",
        "icon": "📈",
    },
    "At-Risk": {
        "color": "#FF5722",
        "action": "Win-back campaign — act before permanent dropout",
        "icon": "⚠️",
    },
    "Low Value": {
        "color": "#9E9E9E",
        "action": "Email-only touch — minimal budget, monitor for growth",
        "icon": "📧",
    },
}

CLV_TOP20_PCT = 0.80  # CLV percentile threshold for High Value
CLV_BOTTOM40_PCT = 0.40  # CLV percentile threshold for Low Value
P_PURCHASE_THRESHOLD = 0.20  # p_purchase cutoff for At-Risk (base rate ~52%)
HOLDOUT_DAYS = 183  # Calibration → cutoff window length


# ---------------------------------------------------------------------------
# Data + model loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_portfolio():
    if not FINAL_PATH.exists():
        return None
    return pd.read_csv(FINAL_PATH)


@st.cache_resource
def load_models():
    if not PROPENSITY_PATH.exists() or not ENCODERS_PATH.exists():
        return None, None
    clf = joblib.load(PROPENSITY_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return clf, encoders


def classify_segment(clv_12m, p_purchase, top20_threshold, bottom40_threshold):
    """Assign 4-tier segment based on CLV and p_purchase."""
    if clv_12m > top20_threshold:
        return "High Value"
    elif p_purchase < P_PURCHASE_THRESHOLD:
        return "At-Risk"
    elif clv_12m > bottom40_threshold:
        return "Growing"
    else:
        return "Low Value"


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CLV Dashboard",
    page_icon="💡",
    layout="wide",
)

st.title("Customer Lifetime Value Dashboard")
st.caption(
    "Two-stage CLV model (purchase propensity + spend tiers) · "
    "UCI Online Retail II · "
    "Calibration: before 2011-06-09 | Holdout: 2011-06-09 → 2011-12-09"
)

tab1, tab2 = st.tabs(["📊 Portfolio Overview", "🔍 Single Customer"])


# ===========================================================================
# TAB 1: PORTFOLIO OVERVIEW
# ===========================================================================
with tab1:
    df = load_portfolio()

    if df is None:
        st.warning(
            "No portfolio data found. Run the notebook pipeline first:\n\n"
            "1. `01_data_extraction.ipynb` → generates `data/raw/clv_data.csv`\n"
            "2. `02_purchase_propensity.ipynb` → Stage 1 classifier\n"
            "3. `03_clv_regression.ipynb` → Stage 2 revenue + combined CLV\n"
            "4. `05_clv_segmentation.ipynb` → generates `data/processed/clv_final.csv`"
        )
        st.stop()

    # ---- KPI cards --------------------------------------------------------
    total_customers = len(df)
    total_clv = df["clv_12m"].sum()
    median_clv = df["clv_12m"].median()
    high_value_n = (df["segment"] == "High Value").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Predicted 12-Month Revenue", f"${total_clv:,.0f}")
    c3.metric("Median CLV", f"${median_clv:.2f}")
    c4.metric("High-Value Customers", f"{high_value_n:,}")

    st.divider()

    # ---- Charts -----------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Customer Count by Segment")
        seg_counts = df["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]

        # Ensure consistent order
        seg_order = ["High Value", "Growing", "At-Risk", "Low Value"]
        seg_counts["segment"] = pd.Categorical(
            seg_counts["segment"], categories=seg_order, ordered=True
        )
        seg_counts = seg_counts.sort_values("segment")

        fig_pie = px.pie(
            seg_counts,
            names="segment",
            values="count",
            color="segment",
            color_discrete_map={s: SEGMENT_CONFIG[s]["color"] for s in SEGMENT_CONFIG},
            hole=0.45,
        )
        fig_pie.update_layout(
            legend_title_text="Segment",
            margin=dict(t=20, b=20, l=20, r=20),
            height=380,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Predicted 12-Month Revenue by Segment")
        seg_rev = df.groupby("segment")["clv_12m"].sum().reset_index()
        seg_rev.columns = ["segment", "predicted_revenue"]
        seg_rev["segment"] = pd.Categorical(
            seg_rev["segment"], categories=seg_order, ordered=True
        )
        seg_rev = seg_rev.sort_values("segment")

        fig_bar = px.bar(
            seg_rev,
            x="segment",
            y="predicted_revenue",
            color="segment",
            color_discrete_map={s: SEGMENT_CONFIG[s]["color"] for s in SEGMENT_CONFIG},
            labels={"predicted_revenue": "Predicted Revenue ($)", "segment": "Segment"},
            text_auto=".3s",
        )
        fig_bar.update_layout(
            showlegend=False,
            yaxis_tickprefix="$",
            margin=dict(t=20, b=20, l=20, r=20),
            height=380,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Segment details table --------------------------------------------
    st.subheader("Segment Profile")
    seg_profile = (
        df.groupby("segment")
        .agg(
            Customers=("user_id", "count"),
            Avg_CLV=("clv_12m", "mean"),
            Median_CLV=("clv_12m", "median"),
            Total_Revenue=("clv_12m", "sum"),
            Avg_p_purchase=("p_purchase", "mean"),
            Avg_frequency=("frequency", "mean"),
        )
        .round(2)
    )

    seg_profile.index = pd.CategoricalIndex(
        seg_profile.index, categories=seg_order, ordered=True
    )
    seg_profile = seg_profile.sort_index()
    seg_profile["Revenue_Share_%"] = (
        seg_profile["Total_Revenue"] / seg_profile["Total_Revenue"].sum() * 100
    ).round(1)
    seg_profile["Recommended_Action"] = [
        SEGMENT_CONFIG.get(s, {}).get("action", "") for s in seg_profile.index
    ]

    st.dataframe(
        seg_profile.style.format(
            {
                "Avg_CLV": "${:.2f}",
                "Median_CLV": "${:.2f}",
                "Total_Revenue": "${:,.0f}",
                "Avg_p_purchase": "{:.3f}",
                "Avg_frequency": "{:.2f}",
                "Revenue_Share_%": "{:.1f}%",
            }
        ),
        use_container_width=True,
    )


# ===========================================================================
# TAB 2: SINGLE CUSTOMER
# ===========================================================================
with tab2:
    st.subheader("Single-Customer CLV Lookup / Scorer")
    mode = st.radio(
        "Mode",
        ["Look up existing customer by ID", "Score new customer (manual entry)"],
        horizontal=True,
    )

    # ------------------------------------------------------------------
    # MODE A: Look up by ID
    # ------------------------------------------------------------------
    if mode == "Look up existing customer by ID":
        df = load_portfolio()
        if df is None:
            st.warning("Run the notebook pipeline first to generate `clv_final.csv`.")
            st.stop()

        customer_id = st.number_input(
            "Customer ID",
            min_value=int(df["user_id"].min()),
            max_value=int(df["user_id"].max()),
            step=1,
        )
        if st.button("Look Up"):
            row = df[df["user_id"] == customer_id]
            if row.empty:
                st.error(f"Customer {customer_id} not found in dataset.")
            else:
                r = row.iloc[0]
                seg = r.get("segment", "Unknown")
                cfg = SEGMENT_CONFIG.get(seg, {})

                st.success(f"{cfg.get('icon', '')} **Segment: {seg}**")
                st.info(f"Recommended action: {cfg.get('action', 'N/A')}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted 12m CLV", f"${r['clv_12m']:.2f}")
                m2.metric("P(purchase)", f"{r['p_purchase']:.3f}")
                m3.metric("Spend Tier", f"{r.get('spend_tier', 'N/A')}")
                m4.metric("Frequency (hist.)", f"{int(r['frequency'])}")

                with st.expander("Full customer profile"):
                    display_cols = [
                        "user_id",
                        "frequency",
                        "recency",
                        "T",
                        "monetary_value",
                        "total_orders",
                        "avg_order_value",
                        "days_since_last_order",
                        "unique_products",
                        "avg_basket_size",
                        "purchase_regularity",
                        "cancellation_rate",
                        "days_active",
                        "country",
                        "p_purchase",
                        "spend_tier",
                        "expected_revenue_if_purchase",
                        "clv_180d",
                        "clv_12m",
                        "segment",
                    ]
                    display_cols = [c for c in display_cols if c in r.index]
                    st.dataframe(r[display_cols].to_frame().T, use_container_width=True)

    # ------------------------------------------------------------------
    # MODE B: Manual entry → live model inference
    # ------------------------------------------------------------------
    else:
        clf, encoders = load_models()
        if clf is None:
            st.warning(
                "Models not found. Run `02_purchase_propensity.ipynb` to generate:\n"
                "- `models/purchase_propensity_model.pkl`\n"
                "- `models/label_encoders.pkl`"
            )
            st.stop()

        # Load portfolio data for threshold + tier computation
        df_portfolio = load_portfolio()
        if df_portfolio is not None:
            top20_threshold = df_portfolio["clv_12m"].quantile(0.80)
            bottom40_threshold = df_portfolio["clv_12m"].quantile(0.40)
            # Compute tier revenue averages from portfolio
            tier_avg_map = (
                df_portfolio[df_portfolio["actual_holdout_transactions"] > 0]
                .groupby("spend_tier")["actual_holdout_revenue"]
                .mean()
                .to_dict()
            )
        else:
            top20_threshold = 100.0
            bottom40_threshold = 20.0
            tier_avg_map = {"Low Spend": 50.0, "Mid Spend": 500.0, "High Spend": 2000.0}

        st.markdown("Enter customer features to score:")

        # --- Purchase history features ---
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Purchase History**")
            frequency = st.number_input(
                "Frequency (repeat purchases)",
                min_value=0,
                max_value=500,
                value=3,
                help="Total orders minus 1 (0 = one-time buyer)",
            )
            recency = st.number_input(
                "Recency (days, first → last purchase)",
                min_value=0,
                max_value=3650,
                value=120,
                help="Days from first to last purchase",
            )
            T = st.number_input(
                "T (customer age in days)",
                min_value=8,
                max_value=3650,
                value=365,
                help="Days from first purchase to calibration end",
            )
            monetary_value = st.number_input(
                "Monetary Value (avg £ per transaction)",
                min_value=0.01,
                max_value=50000.0,
                value=350.0,
                help="Average order value on repeat transactions",
            )
            total_orders = st.number_input(
                "Total Orders", min_value=1, max_value=500, value=4
            )
            avg_order_value = st.number_input(
                "Avg Order Value (£)", min_value=0.01, max_value=50000.0, value=350.0
            )
            days_since_last_order = st.number_input(
                "Days Since Last Order", min_value=0, max_value=3650, value=60
            )

        with col2:
            st.markdown("**Shopping Behavior**")
            unique_products = st.number_input(
                "Unique Products Purchased",
                min_value=1,
                max_value=5000,
                value=50,
                help="Distinct product codes (StockCodes) ordered",
            )
            avg_basket_size = st.number_input(
                "Avg Basket Size (items/order)",
                min_value=0.1,
                max_value=1000.0,
                value=20.0,
                help="Average number of items per order",
            )
            purchase_regularity = st.number_input(
                "Purchase Regularity (std dev of inter-purchase days)",
                min_value=0.0,
                max_value=500.0,
                value=30.0,
                help="Lower = more regular purchasing pattern (0 for ≤ 2 orders)",
            )
            cancellation_rate = st.number_input(
                "Cancellation Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                help="Proportion of invoices that were cancellations",
            )
            days_active = st.number_input(
                "Days Active",
                min_value=0,
                max_value=3650,
                value=120,
                help="Days between first and last purchase",
            )

            # Country dropdown using actual encoder classes
            country_list = list(encoders["country"].classes_)
            default_idx = country_list.index("United Kingdom") if "United Kingdom" in country_list else 0
            country = st.selectbox("Country", country_list, index=default_idx)

        if recency > T:
            st.error(f"Recency ({recency}) cannot exceed T ({T}). Please adjust.")
            st.stop()

        if st.button("Score Customer", type="primary"):
            # Encode country
            country_enc = encoders["country"].transform([country])[0]

            # Derived feature
            recency_ratio = recency / T if T > 0 else 0

            # Build feature vector (must match model training order)
            features = {
                "frequency": frequency,
                "recency": recency,
                "T": T,
                "monetary_value": monetary_value,
                "total_orders": total_orders,
                "avg_order_value": avg_order_value,
                "days_since_last_order": days_since_last_order,
                "unique_products": unique_products,
                "avg_basket_size": avg_basket_size,
                "purchase_regularity": purchase_regularity,
                "cancellation_rate": cancellation_rate,
                "days_active": days_active,
                "recency_ratio": recency_ratio,
                "country_enc": country_enc,
            }
            X_input = pd.DataFrame([features])

            # Stage 1: purchase propensity
            p_purchase = clf.predict_proba(X_input)[:, 1][0]

            # Stage 2: tier-based expected revenue
            # Determine spend tier from monetary_value
            if df_portfolio is not None:
                tier_thresholds = (
                    df_portfolio["monetary_value"].quantile([1 / 3, 2 / 3]).values
                )
                if monetary_value <= tier_thresholds[0]:
                    spend_tier = "Low Spend"
                elif monetary_value <= tier_thresholds[1]:
                    spend_tier = "Mid Spend"
                else:
                    spend_tier = "High Spend"
            else:
                spend_tier = "Mid Spend"

            expected_rev = tier_avg_map.get(spend_tier, 500.0)

            # Combined CLV
            clv_180d = p_purchase * expected_rev
            clv_12m = clv_180d * (365 / HOLDOUT_DAYS)

            # Segment
            segment = classify_segment(
                clv_12m, p_purchase, top20_threshold, bottom40_threshold
            )
            cfg = SEGMENT_CONFIG[segment]

            # Display results
            st.divider()
            st.success(f"{cfg['icon']} **Segment: {segment}**")
            st.info(f"Recommended action: {cfg['action']}")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Predicted 12m CLV", f"${clv_12m:.2f}")
            r2.metric("P(purchase)", f"{p_purchase:.3f}")
            r3.metric("Spend Tier", spend_tier)
            r4.metric("E[Rev | Purchase]", f"${expected_rev:.2f}")

            # Visual: gauge-style CLV bar
            st.markdown("**CLV relative to portfolio thresholds:**")
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=clv_12m,
                    number={"prefix": "$", "valueformat": ",.2f"},
                    gauge={
                        "axis": {"range": [0, top20_threshold * 1.5]},
                        "bar": {"color": cfg["color"]},
                        "steps": [
                            {"range": [0, bottom40_threshold], "color": "#F5F5F5"},
                            {
                                "range": [bottom40_threshold, top20_threshold],
                                "color": "#E8F5E9",
                            },
                            {
                                "range": [top20_threshold, top20_threshold * 1.5],
                                "color": "#E3F2FD",
                            },
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 3},
                            "thickness": 0.75,
                            "value": top20_threshold,
                        },
                    },
                    title={"text": "12-Month CLV ($)"},
                )
            )
            fig.update_layout(height=300, margin=dict(t=30, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)
