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
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parents[1]
FINAL_PATH = REPO_ROOT / "data" / "processed" / "clv_final.csv"
BGNBD_PATH = REPO_ROOT / "models" / "bgnbd_model.pkl"
GG_PATH    = REPO_ROOT / "models" / "gg_model.pkl"

# ---------------------------------------------------------------------------
# Segment config (colours + recommended actions)
# ---------------------------------------------------------------------------
SEGMENT_CONFIG = {
    "High Value": {
        "color":  "#2196F3",
        "action": "VIP loyalty — no discounts, protect margin",
        "icon":   "💎",
    },
    "Growing": {
        "color":  "#4CAF50",
        "action": "Personalized offer — growth potential justifies investment",
        "icon":   "📈",
    },
    "At-Risk": {
        "color":  "#FF5722",
        "action": "Win-back campaign — act before permanent dropout",
        "icon":   "⚠️",
    },
    "Low Value": {
        "color":  "#9E9E9E",
        "action": "Email-only touch — minimal budget, monitor for growth",
        "icon":   "📧",
    },
}

CLV_TOP20_PCT    = 0.80   # CLV percentile threshold for High Value
CLV_BOTTOM40_PCT = 0.40   # CLV percentile threshold for Low Value
PALIVE_THRESHOLD = 0.30   # p_alive cutoff for At-Risk


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
    if not BGNBD_PATH.exists() or not GG_PATH.exists():
        return None, None
    bgf = joblib.load(BGNBD_PATH)
    ggf = joblib.load(GG_PATH)
    return bgf, ggf


def classify_segment(clv_12m, p_alive, top20_threshold, bottom40_threshold):
    """Assign 4-tier segment based on CLV and p_alive."""
    if clv_12m > top20_threshold:
        return "High Value"
    elif p_alive < PALIVE_THRESHOLD:
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
    "BG/NBD + Gamma-Gamma probabilistic CLV model • TheLook E-Commerce • "
    "Calibration: before 2025-04-04 | Holdout: 2025-04-04 → 2025-10-01"
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
            "2. `02_clv_bgnbd.ipynb` → fits models, generates `data/processed/clv_scored.csv`\n"
            "3. `04_clv_segmentation.ipynb` → generates `data/processed/clv_final.csv`"
        )
        st.stop()

    # ---- KPI cards --------------------------------------------------------
    total_customers = len(df)
    total_clv       = df["clv_12m"].sum()
    median_clv      = df["clv_12m"].median()
    high_value_n    = (df["segment"] == "High Value").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",         f"{total_customers:,}")
    c2.metric("Predicted 12-Month Revenue", f"${total_clv:,.0f}")
    c3.metric("Median CLV",              f"${median_clv:.2f}")
    c4.metric("High-Value Customers",    f"{high_value_n:,}")

    st.divider()

    # ---- Charts -----------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Customer Count by Segment")
        seg_counts = df["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]

        # Ensure consistent order
        seg_order   = ["High Value", "Growing", "At-Risk", "Low Value"]
        seg_colors  = [SEGMENT_CONFIG[s]["color"] for s in seg_order if s in seg_counts["segment"].values]
        seg_counts["segment"] = pd.Categorical(seg_counts["segment"], categories=seg_order, ordered=True)
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
        seg_rev["segment"] = pd.Categorical(seg_rev["segment"], categories=seg_order, ordered=True)
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
    seg_profile = df.groupby("segment").agg(
        Customers        = ("user_id", "count"),
        Avg_CLV          = ("clv_12m", "mean"),
        Median_CLV       = ("clv_12m", "median"),
        Total_Revenue    = ("clv_12m", "sum"),
        Avg_p_alive      = ("p_alive", "mean"),
        Avg_frequency    = ("frequency", "mean"),
    ).round(2)

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
        seg_profile.style.format({
            "Avg_CLV":       "${:.2f}",
            "Median_CLV":    "${:.2f}",
            "Total_Revenue": "${:,.0f}",
            "Avg_p_alive":   "{:.3f}",
            "Avg_frequency": "{:.2f}",
            "Revenue_Share_%": "{:.1f}%",
        }),
        use_container_width=True,
    )


# ===========================================================================
# TAB 2: SINGLE CUSTOMER
# ===========================================================================
with tab2:
    bgf, ggf = load_models()

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
            "Customer ID", min_value=int(df["user_id"].min()),
            max_value=int(df["user_id"].max()), step=1
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
                m1.metric("Predicted 12m CLV",   f"${r['clv_12m']:.2f}")
                m2.metric("P(alive)",             f"{r['p_alive']:.3f}")
                m3.metric("Predicted Purchases",  f"{r['predicted_purchases_12m']:.2f}")
                m4.metric("Frequency (hist.)",    f"{int(r['frequency'])}")

                with st.expander("Full customer profile"):
                    display_cols = [
                        "user_id", "frequency", "recency", "T", "monetary_value",
                        "total_orders", "avg_order_value", "days_since_last_order",
                        "p_alive", "predicted_purchases_12m", "expected_avg_spend",
                        "clv_12m", "segment",
                    ]
                    display_cols = [c for c in display_cols if c in r.index]
                    st.dataframe(r[display_cols].to_frame().T, use_container_width=True)

    # ------------------------------------------------------------------
    # MODE B: Manual entry → live model inference
    # ------------------------------------------------------------------
    else:
        if bgf is None or ggf is None:
            st.warning(
                "Models not found. Run `02_clv_bgnbd.ipynb` to generate:\n"
                "- `models/bgnbd_model.pkl`\n- `models/gg_model.pkl`"
            )
            st.stop()

        # Load portfolio data for threshold computation (if available)
        df_portfolio = load_portfolio()
        if df_portfolio is not None:
            top20_threshold    = df_portfolio["clv_12m"].quantile(0.80)
            bottom40_threshold = df_portfolio["clv_12m"].quantile(0.40)
        else:
            # Fallback thresholds (can be updated after running notebooks)
            top20_threshold    = 100.0
            bottom40_threshold = 20.0

        st.markdown("Enter BG/NBD inputs to score a new or hypothetical customer:")

        col1, col2 = st.columns(2)
        with col1:
            frequency = st.number_input(
                "Frequency (repeat purchases)", min_value=0, max_value=100, value=2,
                help="Total orders minus 1 (0 = one-time buyer)"
            )
            recency = st.number_input(
                "Recency (days, first → last purchase)", min_value=0, max_value=3650, value=90,
                help="Days from first to last purchase"
            )
        with col2:
            T = st.number_input(
                "T (customer age in days)", min_value=8, max_value=3650, value=365,
                help="Days from first purchase to calibration end"
            )
            monetary_value = st.number_input(
                "Monetary Value (avg $ per transaction)", min_value=0.01, max_value=10000.0, value=60.0,
                help="Average order value (repeat transactions)"
            )

        if recency > T:
            st.error(f"Recency ({recency}) cannot exceed T ({T}). Please adjust.")
            st.stop()

        if st.button("Score Customer", type="primary"):
            # BG/NBD predictions
            p_alive = bgf.conditional_probability_alive(frequency, recency, T)
            pred_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
                365, frequency, recency, T
            )

            # Gamma-Gamma spend estimate
            if frequency > 0:
                expected_spend = float(
                    ggf.conditional_expected_average_profit(
                        pd.Series([frequency]),
                        pd.Series([monetary_value]),
                    ).iloc[0]
                )
            else:
                expected_spend = monetary_value  # fallback for one-time buyers

            clv_12m = pred_purchases * expected_spend

            # Segment classification
            segment = classify_segment(clv_12m, p_alive, top20_threshold, bottom40_threshold)
            cfg = SEGMENT_CONFIG[segment]

            # Display results
            st.divider()
            st.success(f"{cfg['icon']} **Segment: {segment}**")
            st.info(f"Recommended action: {cfg['action']}")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Predicted 12m CLV",   f"${clv_12m:.2f}")
            r2.metric("P(alive)",             f"{p_alive:.3f}")
            r3.metric("Predicted Purchases",  f"{pred_purchases:.2f}")
            r4.metric("Expected Spend/Order", f"${expected_spend:.2f}")

            # Visual: gauge-style CLV bar
            st.markdown("**CLV relative to portfolio thresholds:**")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=clv_12m,
                number={"prefix": "$", "valueformat": ",.2f"},
                gauge={
                    "axis": {"range": [0, top20_threshold * 1.5]},
                    "bar":  {"color": cfg["color"]},
                    "steps": [
                        {"range": [0, bottom40_threshold],     "color": "#F5F5F5"},
                        {"range": [bottom40_threshold, top20_threshold], "color": "#E8F5E9"},
                        {"range": [top20_threshold, top20_threshold * 1.5], "color": "#E3F2FD"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75,
                        "value": top20_threshold,
                    },
                },
                title={"text": "12-Month CLV ($)"},
            ))
            fig.update_layout(height=300, margin=dict(t=30, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)
