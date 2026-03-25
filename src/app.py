"""
CLV Dashboard: Customer Lifetime Value Predictor
Run from repo root: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
from pathlib import Path
from sklearn.metrics import brier_score_loss

from src.clv_logic import (
    FEATURE_COLS,
    SEGMENT_ORDER,
    SEGMENT_CONFIG,
    DEFAULT_BUDGETS,
    CLV_TOP20_PCT,
    CLV_BOTTOM40_PCT,
    P_PURCHASE_THRESHOLD,
    HOLDOUT_DAYS,
    classify_segment,
    compute_clv_12m,
    assign_spend_tier,
    compute_break_even_lift,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_PATH = REPO_ROOT / "data" / "processed" / "clv_final.csv"
PROPENSITY_PATH = REPO_ROOT / "models" / "purchase_propensity_model.pkl"
ENCODERS_PATH = REPO_ROOT / "models" / "label_encoders.pkl"

# ---------------------------------------------------------------------------
# UI-specific constants
# ---------------------------------------------------------------------------
# Human-readable SHAP direction labels
SHAP_LABELS = {
    "frequency": lambda v: (
        f"{int(v)} repeat purchases" if v > 0 else "zero repeat purchases"
    ),
    "recency": lambda v: f"{int(v)} days since last purchase",
    "T": lambda v: f"{int(v)}-day customer tenure",
    "monetary_value": lambda v: f"${v:,.0f} avg spend",
    "avg_order_value": lambda v: f"${v:,.0f} avg order",
    "unique_products": lambda v: f"{int(v)} unique products",
    "avg_basket_size": lambda v: f"{v:.1f} items/order",
    "interpurchase_std": lambda v: f"{v:.0f}-day purchase variability",
    "is_one_time_buyer": lambda v: "one-time buyer" if v == 1 else "repeat buyer",
    "cancellation_rate": lambda v: f"{v:.0%} cancellation rate",
    "recency_ratio": lambda v: f"{v:.2f} dormancy ratio",
    "country_enc": lambda v: f"country code {int(v)}",
}


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


@st.cache_resource
def load_shap_explainer():
    clf, _ = load_models()
    if clf is None:
        return None
    # Extract base estimator from CalibratedClassifierCV (model-agnostic)
    base_est = clf.calibrated_classifiers_[0].estimator
    if hasattr(base_est, "feature_importances_"):
        return shap.TreeExplainer(base_est)
    else:
        df = load_portfolio()
        if df is None:
            return None
        X_bg = df[FEATURE_COLS].sample(min(100, len(df)), random_state=42)
        return shap.LinearExplainer(base_est, X_bg)


def compute_shap_for_row(X_row):
    """Compute SHAP values for a single-row DataFrame. Returns array or None."""
    explainer = load_shap_explainer()
    if explainer is None:
        return None
    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values[0] if shap_values.ndim > 1 else shap_values


def render_shap_panel(shap_vals, feature_values):
    """Render top-5 SHAP drivers as a horizontal bar chart + direction labels."""
    if shap_vals is None:
        st.info("SHAP explainer not available.")
        return

    abs_vals = np.abs(shap_vals)
    top5_idx = np.argsort(abs_vals)[-5:][::-1]

    names = [FEATURE_COLS[i] for i in top5_idx]
    impacts = [shap_vals[i] for i in top5_idx]
    colors = ["#16A34A" if v > 0 else "#DC2626" for v in impacts]

    fig = go.Figure(
        go.Bar(
            y=names,
            x=impacts,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in impacts],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="SHAP Impact on P(purchase)",
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(t=10, b=30, l=120, r=60),
    )
    st.plotly_chart(fig, width="stretch")

    # Direction labels
    for i in top5_idx:
        feat = FEATURE_COLS[i]
        val = feature_values[i]
        direction = SHAP_LABELS.get(feat, lambda v: f"{v}")(val)
        sign = "+" if shap_vals[i] > 0 else "-"
        color = "#16A34A" if shap_vals[i] > 0 else "#DC2626"
        st.markdown(
            f"<span style='color:{color};font-weight:600'>{sign}</span> "
            f"<code>{feat}</code>: {direction}",
            unsafe_allow_html=True,
        )


def render_customer_profile(r):
    """Render customer profile as key-value grid."""
    profile_items = [
        ("Frequency", f"{int(r['frequency'])}"),
        ("Recency", f"{int(r['recency'])} days"),
        ("Tenure", f"{int(r['T'])} days"),
        ("Monetary Value", f"${r['monetary_value']:,.0f}"),
        ("Avg Order Value", f"${r['avg_order_value']:,.0f}"),
        ("Unique Products", f"{int(r['unique_products'])}"),
        ("Cancellation Rate", f"{r['cancellation_rate']:.0%}"),
    ]
    for label, val in profile_items:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:8px 0;border-bottom:1px solid #F3F4F6'>"
            f"<span style='color:#6B7280;font-size:14px'>{label}</span>"
            f"<span style='font-weight:600;font-size:14px'>{val}</span></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CLV Dashboard",
    page_icon="💡",
    layout="wide",
)

st.title("Customer Lifetime Value Dashboard")
st.caption("Two-stage CLV model · UCI Online Retail II · Holdout: Jun-Dec 2011")

with st.expander("About this project", expanded=False):
    st.markdown(
        "**Two-stage Customer Lifetime Value prediction** for e-commerce retention.\n\n"
        "- **Stage 1:** Calibrated XGBoost classifier predicts purchase probability\n"
        "- **Stage 2:** Spend-tier revenue estimation via pooled daily spend rates\n"
        "- **CLV:** P(purchase) x E[revenue | purchase], annualized\n"
        "- **Segmentation:** 4-tier customer segments with actionable campaign strategies\n\n"
        "Built with: Python, XGBoost, Scikit-learn, Streamlit, Plotly\n\n"
        "[GitHub Repository](https://github.com/dsyhub/customer-lifetime-value-prediction)"
    )

with st.sidebar:
    st.markdown("---")
    st.markdown(
        "[GitHub](https://github.com/dsyhub/customer-lifetime-value-prediction)"
    )

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Executive Summary",
        "🔍 Customer Explorer",
        "💰 Campaign Sensitivity",
        "📋 About",
    ]
)


# ===========================================================================
# TAB 1: EXECUTIVE SUMMARY
# ===========================================================================
with tab1:
    df = load_portfolio()

    if df is None:
        st.warning(
            "No portfolio data found. Run the notebook pipeline first:\n\n"
            "1. `01_exploratory_data_analysis.ipynb` → data extraction\n"
            "2. `02_purchase_propensity_model.ipynb` → Stage 1 classifier\n"
            "3. `03_customer_lifetime_value_segmentation.ipynb` → CLV scoring → `clv_final.csv`"
        )
        st.stop()

    # ---- KPI cards --------------------------------------------------------
    total_customers = len(df)
    total_clv = df["clv_12m"].sum()

    # Top 20% capture: share of actual holdout revenue captured by top-20% CLV customers
    top20_n = int(len(df) * 0.20)
    top20_rev = df.nlargest(top20_n, "clv_12m")["actual_holdout_revenue"].sum()
    total_holdout_rev = df["actual_holdout_revenue"].sum()
    top20_capture = top20_rev / total_holdout_rev if total_holdout_rev > 0 else 0

    # Brier score on test set only (avoids in-sample bias from training data)
    test_mask = (
        df["is_test"] if "is_test" in df.columns else pd.Series(True, index=df.index)
    )
    test_df_brier = df[test_mask]
    brier = brier_score_loss(
        test_df_brier["purchased_in_holdout"], test_df_brier["p_purchase"]
    )
    base_rate = test_df_brier["purchased_in_holdout"].mean()
    brier_baseline = base_rate * (1 - base_rate)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Predicted 12-Month Revenue", f"${total_clv:,.0f}")
    c3.metric(
        "Top 20% Capture",
        f"{top20_capture:.1%}",
        help="Share of actual holdout revenue captured by top-20% CLV customers",
    )
    c4.metric(
        "Brier Score",
        f"{brier:.3f}",
        help=f"Probability calibration accuracy (lower is better). Naive baseline: {brier_baseline:.3f}",
    )

    st.divider()

    # ---- Pipeline overview ------------------------------------------------
    st.subheader("Pipeline Overview")
    steps = [
        ("Stage 1", "Propensity", "Calibrated XGBoost → P(purchase)", "#2563EB"),
        ("Stage 2", "Revenue", "Pooled daily spend rate → E[rev | buy]", "#059669"),
        ("CLV", "Segments", "P(buy) × E[rev], annualized → 4 tiers", "#7C3AED"),
    ]
    steps_html = "<div style='display:flex;gap:12px;flex-wrap:wrap;max-width:900px'>"
    for label, title, desc, color in steps:
        steps_html += (
            f"<div style='flex:1;min-width:220px;max-width:300px;border-left:3px solid {color};"
            f"padding:8px 12px;background:var(--secondary-background-color, #F9FAFB);"
            f"border-radius:0 6px 6px 0'>"
            f"<div style='font-weight:600;font-size:14px;color:{color}'>{label}: {title}</div>"
            f"<div style='font-size:13px;color:#6B7280;margin-top:2px'>{desc}</div>"
            f"</div>"
        )
    steps_html += "</div>"
    st.markdown(steps_html, unsafe_allow_html=True)

    with st.expander("Model Card"):
        # ---- Flow diagram (HTML/CSS) ----------------------------------------
        flow_steps = [
            ("Raw Transactions", "#6B7280"),
            ("Feature Engineering\n(RFM + Behavioral)", "#6B7280"),
            ("Stage 1: Calibrated XGBoost\n(Purchase Propensity)", "#2563EB"),
            ("Stage 2: Spend-Tier\nRevenue Estimation", "#2563EB"),
            ("CLV = P(purchase)\n\u00d7 E[revenue]", "#059669"),
            ("4-Tier\nSegmentation", "#059669"),
        ]
        boxes_html = ""
        for i, (label, color) in enumerate(flow_steps):
            label_html = label.replace("\n", "<br>")
            boxes_html += (
                f"<div style='display:flex;align-items:center'>"
                f"<div style='background:{color};color:white;border-radius:8px;"
                f"padding:10px 16px;font-size:13px;font-weight:500;"
                f"text-align:center;min-width:120px;line-height:1.4'>"
                f"{label_html}</div>"
            )
            if i < len(flow_steps) - 1:
                boxes_html += (
                    "<div style='font-size:20px;color:#9CA3AF;"
                    "margin:0 6px'>\u2192</div>"
                )
            boxes_html += "</div>"
        st.markdown(
            f"<div style='display:flex;align-items:center;flex-wrap:wrap;"
            f"gap:4px;margin-bottom:16px'>{boxes_html}</div>",
            unsafe_allow_html=True,
        )

        # ---- Model card details in two columns ------------------------------
        mc_left, mc_right = st.columns(2)
        with mc_left:
            st.markdown(
                "**Model**\n"
                "- XGBoost classifier + isotonic calibration (5-fold)\n"
                "- Optuna tuning: 50 trials, 3-fold CV (PR-AUC)\n"
                "- 12 input features (RFM, behavioral, country)\n"
                "- Temporal holdout split: 2011-06-09 (183 days)"
            )
        with mc_right:
            st.markdown(
                "**Performance**\n"
                "- Brier score: ~0.18 (naive baseline ~0.25)\n"
                "- PR-AUC: ~0.84\n"
                "- Spend tiers: Low $402 / Mid $851 / High $2,866 (pooled daily spend rate \u00d7 183 days)\n"
                "- CLV annualized via 365/183 scaling factor"
            )

    # Segment legend pills
    seg_counts = df["segment"].value_counts()
    pills_html = ""
    for seg in SEGMENT_ORDER:
        cnt = seg_counts.get(seg, 0)
        pct = cnt / total_customers * 100
        color = SEGMENT_CONFIG[seg]["color"]
        pills_html += (
            f"<span style='display:inline-flex;align-items:center;gap:6px;"
            f"background:#F9FAFB;border-radius:6px;padding:6px 12px;margin-right:8px'>"
            f"<span style='width:10px;height:10px;border-radius:50%;"
            f"background:{color};display:inline-block'></span>"
            f"<span style='font-size:13px;font-weight:500;color:#374151'>"
            f"{seg}: {cnt:,} ({pct:.0f}%)</span></span>"
        )
    st.markdown(pills_html, unsafe_allow_html=True)

    # Stacked segment bar
    seg_bar_data = pd.DataFrame(
        [{"segment": seg, "count": seg_counts.get(seg, 0)} for seg in SEGMENT_ORDER]
    )
    fig_bar = px.bar(
        seg_bar_data,
        x="count",
        y=[""] * len(seg_bar_data),
        color="segment",
        orientation="h",
        color_discrete_map={s: SEGMENT_CONFIG[s]["color"] for s in SEGMENT_CONFIG},
        category_orders={"segment": SEGMENT_ORDER},
        text=seg_bar_data["count"].apply(lambda c: f"{c / total_customers:.0%}"),
    )
    fig_bar.update_layout(
        barmode="stack",
        showlegend=False,
        height=70,
        margin=dict(t=5, b=5, l=0, r=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    fig_bar.update_traces(
        textposition="inside", textfont_size=12, textfont_color="white"
    )
    st.plotly_chart(fig_bar, width="stretch")

    st.divider()

    # ---- Charts -----------------------------------------------------------
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Predicted 12-Month Revenue by Segment")
        seg_rev = df.groupby("segment")["clv_12m"].sum().reset_index()
        seg_rev.columns = ["segment", "predicted_revenue"]
        seg_rev["segment"] = pd.Categorical(
            seg_rev["segment"], categories=SEGMENT_ORDER, ordered=True
        )
        seg_rev = seg_rev.sort_values("segment")

        # Revenue share for labels
        total_rev = seg_rev["predicted_revenue"].sum()
        seg_rev["label"] = seg_rev.apply(
            lambda row: (
                f"${row['predicted_revenue'] / 1e6:.1f}M ({row['predicted_revenue'] / total_rev:.1%})"
                if row["predicted_revenue"] >= 1e6
                else f"${row['predicted_revenue'] / 1e3:.0f}K ({row['predicted_revenue'] / total_rev:.1%})"
            ),
            axis=1,
        )

        fig_rev = px.bar(
            seg_rev,
            y="segment",
            x="predicted_revenue",
            color="segment",
            orientation="h",
            color_discrete_map={s: SEGMENT_CONFIG[s]["color"] for s in SEGMENT_CONFIG},
            text="label",
        )
        fig_rev.update_layout(
            showlegend=False,
            xaxis_tickprefix="$",
            xaxis_title="Predicted Revenue ($)",
            yaxis_title="",
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
        )
        fig_rev.update_traces(
            textposition="inside", textfont_size=12, textfont_color="white"
        )
        st.plotly_chart(fig_rev, width="stretch")

    with col_right:
        st.subheader("Segment Profiles")
        seg_profile = (
            df.groupby("segment")
            .agg(
                Avg_CLV=("clv_12m", "mean"),
                Avg_p_purchase=("p_purchase", "mean"),
            )
            .round(2)
        )
        seg_profile.index = pd.CategoricalIndex(
            seg_profile.index, categories=SEGMENT_ORDER, ordered=True
        )
        seg_profile = seg_profile.sort_index()

        st.dataframe(
            seg_profile.style.format(
                {"Avg_CLV": "${:,.0f}", "Avg_p_purchase": "{:.0%}"}
            ),
            width="stretch",
            height=220,
        )


# ===========================================================================
# TAB 2: CUSTOMER EXPLORER
# ===========================================================================
with tab2:
    st.subheader("Customer Explorer")
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

        customer_ids = sorted(df["user_id"].unique())
        customer_id = st.selectbox(
            "Select Customer ID",
            customer_ids,
            index=0,
        )

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
            m1.metric("P(purchase)", f"{r['p_purchase']:.3f}")
            m2.metric("Predicted 12m CLV", f"${r['clv_12m']:.2f}")
            m3.metric("Segment", seg)
            m4.metric("Spend Tier", f"{r.get('spend_tier', 'N/A')}")

            st.divider()
            profile_col, shap_col = st.columns(2)

            with profile_col:
                st.markdown("**Customer Profile**")
                render_customer_profile(r)

            with shap_col:
                st.markdown("**Prediction Drivers (SHAP)**")
                X_single = row[FEATURE_COLS]
                shap_vals = compute_shap_for_row(X_single)
                render_shap_panel(shap_vals, X_single.values[0])

    # ------------------------------------------------------------------
    # MODE B: Manual entry → live model inference
    # ------------------------------------------------------------------
    else:
        clf, encoders = load_models()
        if clf is None:
            st.warning(
                "Models not found. Run `02_purchase_propensity_model.ipynb` to generate:\n"
                "- `models/purchase_propensity_model.pkl`\n"
                "- `models/label_encoders.pkl`"
            )
            st.stop()

        # Load portfolio data for threshold + tier computation
        df_portfolio = load_portfolio()
        if df_portfolio is not None:
            top20_threshold = df_portfolio["clv_12m"].quantile(0.80)
            bottom40_threshold = df_portfolio["clv_12m"].quantile(0.40)
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
                "Recency (days since last order)",
                min_value=0,
                max_value=3650,
                value=60,
                help="Days from last purchase to calibration end",
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
            avg_order_value = st.number_input(
                "Avg Order Value (£)",
                min_value=0.01,
                max_value=50000.0,
                value=350.0,
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
            interpurchase_std = st.number_input(
                "Inter-purchase Std Dev (days)",
                min_value=0.0,
                max_value=500.0,
                value=30.0,
                help="Std dev of days between orders (0 for one-time buyers)",
            )
            is_one_time_buyer = st.selectbox(
                "One-Time Buyer",
                [0, 1],
                index=0,
                help="1 = customer made only one purchase in calibration period",
            )
            cancellation_rate = st.number_input(
                "Cancellation Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                help="Proportion of invoices that were cancellations",
            )
            country_list = list(encoders["country"].classes_)
            default_idx = (
                country_list.index("United Kingdom")
                if "United Kingdom" in country_list
                else 0
            )
            country = st.selectbox("Country", country_list, index=default_idx)

        if recency > T:
            st.error(
                f"Recency ({recency} days since last order) cannot exceed T ({T} customer age). Please adjust."
            )
            st.stop()

        if st.button("Score Customer", type="primary"):
            country_enc = encoders["country"].transform([country])[0]
            recency_ratio = recency / T if T > 0 else 0

            features = {
                "frequency": frequency,
                "recency": recency,
                "T": T,
                "monetary_value": monetary_value,
                "avg_order_value": avg_order_value,
                "unique_products": unique_products,
                "avg_basket_size": avg_basket_size,
                "interpurchase_std": interpurchase_std,
                "is_one_time_buyer": is_one_time_buyer,
                "cancellation_rate": cancellation_rate,
                "recency_ratio": recency_ratio,
                "country_enc": country_enc,
            }
            X_input = pd.DataFrame([features])

            # Stage 1: purchase propensity
            p_purchase = clf.predict_proba(X_input)[:, 1][0]

            # Stage 2: tier-based expected revenue
            if df_portfolio is not None:
                tier_thresholds = (
                    df_portfolio["monetary_value"].quantile([1 / 3, 2 / 3]).values
                )
                spend_tier = assign_spend_tier(
                    monetary_value, tier_thresholds[0], tier_thresholds[1]
                )
            else:
                spend_tier = "Mid Spend"

            expected_rev = tier_avg_map.get(spend_tier, 500.0)

            # Combined CLV
            clv_12m = compute_clv_12m(p_purchase, expected_rev)

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
            r1.metric("P(purchase)", f"{p_purchase:.3f}")
            r2.metric("Predicted 12m CLV", f"${clv_12m:.2f}")
            r3.metric("Spend Tier", spend_tier)
            r4.metric("E[Rev | Purchase]", f"${expected_rev:.2f}")

            # Two-panel: profile + SHAP
            profile_col, shap_col = st.columns(2)

            with profile_col:
                st.markdown("**Input Summary**")
                for feat, val in features.items():
                    label_fn = SHAP_LABELS.get(feat, lambda v: f"{v}")
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"padding:6px 0;border-bottom:1px solid #F3F4F6'>"
                        f"<span style='color:#6B7280;font-size:13px'>{feat}</span>"
                        f"<span style='font-weight:600;font-size:13px'>{label_fn(val)}</span></div>",
                        unsafe_allow_html=True,
                    )

            with shap_col:
                st.markdown("**Prediction Drivers (SHAP)**")
                shap_vals = compute_shap_for_row(X_input)
                render_shap_panel(shap_vals, X_input.values[0])

            # CLV gauge
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
            st.plotly_chart(fig, width="stretch")


# ===========================================================================
# TAB 3: CAMPAIGN SENSITIVITY
# ===========================================================================
with tab3:
    df = load_portfolio()

    if df is None:
        st.warning("Run the notebook pipeline first to generate `clv_final.csv`.")
        st.stop()

    # Segment aggregates
    seg_agg = df.groupby("segment").agg(
        n_customers=("user_id", "count"),
        avg_clv=("clv_12m", "mean"),
        avg_p_purchase=("p_purchase", "mean"),
    )
    seg_agg.index = pd.CategoricalIndex(
        seg_agg.index, categories=SEGMENT_ORDER, ordered=True
    )
    seg_agg = seg_agg.sort_index()

    # ---- Budget sliders ---------------------------------------------------
    st.subheader("Campaign Budget per Customer")
    st.caption("Adjust sliders to see how break-even lift and ROI change")

    slider_cols = st.columns(4)
    budgets = {}
    for i, seg in enumerate(SEGMENT_ORDER):
        with slider_cols[i]:
            color = SEGMENT_CONFIG[seg]["color"]
            st.markdown(
                f"<span style='display:inline-block;width:10px;height:10px;"
                f"border-radius:50%;background:{color};margin-right:6px'></span>"
                f"**{seg}**",
                unsafe_allow_html=True,
            )
            budgets[seg] = st.slider(
                f"Budget ({seg})",
                min_value=0,
                max_value=30,
                value=DEFAULT_BUDGETS[seg],
                key=f"budget_{seg}",
                label_visibility="collapsed",
            )
            st.markdown(f"**${budgets[seg]}/customer**")

    st.divider()

    # ---- Break-even lift table --------------------------------------------
    st.subheader("Break-Even Incremental Lift")

    be_data = []
    for seg in SEGMENT_ORDER:
        budget = budgets[seg]
        avg_clv = seg_agg.loc[seg, "avg_clv"]
        be_lift = compute_break_even_lift(budget, avg_clv)
        if be_lift is not None:
            one_in_n = round(1 / be_lift)
        else:
            be_lift = 0
            one_in_n = 0

        be_data.append(
            {
                "Segment": seg,
                "Budget": f"${budget}",
                "Avg CLV": f"${avg_clv:,.0f}",
                "Break-Even Lift": f"{be_lift:.2%}" if budget > 0 else "N/A",
                "Intuition": f"1 in {one_in_n:,}" if budget > 0 else "N/A",
                "Action": SEGMENT_CONFIG[seg]["action"],
            }
        )

    st.dataframe(pd.DataFrame(be_data), width="stretch", hide_index=True)

    st.divider()

    # ---- ROI sensitivity chart --------------------------------------------
    st.subheader("ROI Sensitivity Chart")
    st.caption("Total Net ROI by incremental lift percentage. Dots = break-even point.")

    lift_range = np.arange(0.005, 0.205, 0.005)

    fig_roi = go.Figure()
    for seg in SEGMENT_ORDER:
        budget = budgets[seg]
        avg_clv = seg_agg.loc[seg, "avg_clv"]
        n_cust = seg_agg.loc[seg, "n_customers"]
        color = SEGMENT_CONFIG[seg]["color"]

        net_roi = lift_range * avg_clv * n_cust - budget * n_cust
        fig_roi.add_trace(
            go.Scatter(
                x=lift_range * 100,
                y=net_roi,
                name=seg,
                line=dict(color=color, width=2.5),
                mode="lines",
            )
        )

        # Break-even dot
        if budget > 0 and avg_clv > 0:
            be_x = (budget / avg_clv) * 100
            if 0 < be_x <= 20:
                fig_roi.add_trace(
                    go.Scatter(
                        x=[be_x],
                        y=[0],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=color,
                            line=dict(width=2, color="white"),
                        ),
                        showlegend=False,
                        hovertemplate=f"{seg}: break-even at {be_x:.1f}% lift<extra></extra>",
                    )
                )

    fig_roi.add_hline(y=0, line_dash="dash", line_color="#9CA3AF", line_width=1)
    fig_roi.update_layout(
        xaxis_title="Incremental Lift (percentage points)",
        yaxis_title="Net ROI ($)",
        yaxis_tickprefix="$",
        height=400,
        margin=dict(t=20, b=40, l=60, r=20),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_roi, width="stretch")


# ===========================================================================
# TAB 4: ABOUT
# ===========================================================================
with tab4:
    st.subheader("Methodology")
    st.markdown(
        "This dashboard uses a **two-stage approach** to predict Customer Lifetime Value:\n\n"
        "1. **Stage 1: Purchase Propensity.** A calibrated XGBoost classifier predicts "
        "the probability that each customer will make a purchase in the next 6 months. "
        "The model was selected from a 4-model comparison (Logistic Regression, Random Forest, "
        "XGBoost, LightGBM), tuned with Optuna (50 trials), and calibrated with isotonic "
        "regression to produce accurate probabilities.\n\n"
        "2. **Stage 2: Expected Revenue.** Rather than predicting individual spend amounts "
        "(which yielded negative R²), customers are grouped into spend tiers (Low, Mid, High) "
        "using tercile splits on monetary value. Expected revenue is estimated via a pooled "
        "daily spend rate within each tier, which weights longer-observed customers more heavily "
        "to reduce noise from short-tenure spending bursts.\n\n"
        "3. **CLV Formula:** `P(purchase) × E[revenue | purchase]`, annualized from the "
        "183-day holdout window via 365/183 scaling."
    )

    st.divider()

    st.subheader("Data")
    st.markdown(
        "- **Source:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) "
        "dataset (~4,900 customers, Dec 2009 to Dec 2011)\n"
        "- **Temporal Holdout:** Calibration period ends 2011-06-09; holdout runs "
        "2011-06-09 to 2011-12-09 (183 days)\n"
        "- **Features:** 12 engineered features spanning RFM metrics, purchase behavior, "
        "shopping patterns, and country"
    )

    st.divider()

    st.subheader("Segmentation Rules")
    seg_rules = pd.DataFrame(
        [
            ["High Value", "Top 20% by CLV", "Any", "VIP loyalty: protect margin"],
            [
                "Growing",
                "Middle 40%",
                ">= 20%",
                "Personalized offers, growth potential",
            ],
            ["At-Risk", "Any", "< 20%", "Win-back campaigns, act before dropout"],
            ["Low Value", "Bottom 40%", ">= 20%", "Email-only, minimal budget"],
        ],
        columns=["Segment", "CLV Threshold", "P(purchase)", "Strategy"],
    )
    st.dataframe(seg_rules, width="stretch", hide_index=True)

    st.divider()

    st.subheader("Tech Stack")
    st.markdown(
        "| Component | Libraries |\n"
        "|---|---|\n"
        "| ML Models | XGBoost, Scikit-learn (calibration) |\n"
        "| Hyperparameter Tuning | Optuna (50 trials) |\n"
        "| Explainability | SHAP (TreeExplainer) |\n"
        "| Data Processing | Pandas, NumPy, SciPy |\n"
        "| Visualization | Plotly, Streamlit |\n"
        "| Dashboard | Streamlit |"
    )

    st.divider()

    st.subheader("Links")
    st.markdown(
        "- [GitHub Repository](https://github.com/dsyhub/customer-lifetime-value-prediction)"
    )
