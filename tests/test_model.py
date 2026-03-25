"""Integration tests for model loading and prediction pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.clv_logic import HOLDOUT_DAYS


pytestmark = pytest.mark.slow


def test_model_loads_with_predict_proba(model_and_encoders):
    clf, _ = model_and_encoders
    assert hasattr(clf, "predict_proba"), "Model must support predict_proba"


def test_single_prediction_valid_probability(model_and_encoders, app):
    clf, _ = model_and_encoders
    features = {
        "frequency": 5,
        "recency": 30,
        "T": 365,
        "monetary_value": 500.0,
        "avg_order_value": 500.0,
        "unique_products": 50,
        "avg_basket_size": 20.0,
        "interpurchase_std": 30.0,
        "is_one_time_buyer": 0,
        "cancellation_rate": 0.05,
        "recency_ratio": 30 / 365,
        "country_enc": 0,
    }
    X = pd.DataFrame([features])[app.FEATURE_COLS]
    proba = clf.predict_proba(X)

    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_end_to_end_pipeline(model_and_encoders, app, clv_data):
    """Full pipeline: features -> predict -> CLV -> segment."""
    clf, encoders = model_and_encoders

    # Encode a country
    country_enc = encoders["country"].transform(["United Kingdom"])[0]

    features = {
        "frequency": 5,
        "recency": 30,
        "T": 365,
        "monetary_value": 500.0,
        "avg_order_value": 500.0,
        "unique_products": 50,
        "avg_basket_size": 20.0,
        "interpurchase_std": 30.0,
        "is_one_time_buyer": 0,
        "cancellation_rate": 0.05,
        "recency_ratio": 30 / 365,
        "country_enc": country_enc,
    }
    X = pd.DataFrame([features])[app.FEATURE_COLS]

    # Stage 1: propensity
    p_purchase = clf.predict_proba(X)[0, 1]

    # Stage 2: tier-based revenue (use portfolio thresholds)
    tier_thresholds = clv_data["monetary_value"].quantile([1 / 3, 2 / 3]).values
    if features["monetary_value"] <= tier_thresholds[0]:
        spend_tier = "Low Spend"
    elif features["monetary_value"] <= tier_thresholds[1]:
        spend_tier = "Mid Spend"
    else:
        spend_tier = "High Spend"

    buyers = clv_data[clv_data["actual_holdout_transactions"] > 0]
    expected_rev = buyers.groupby("spend_tier")["actual_holdout_revenue"].mean().to_dict()
    rev = expected_rev.get(spend_tier, 500.0)

    # CLV
    clv_12m = p_purchase * rev * (365 / HOLDOUT_DAYS)
    assert clv_12m >= 0

    # Segment
    top20 = clv_data["clv_12m"].quantile(0.80)
    bottom40 = clv_data["clv_12m"].quantile(0.40)
    segment = app.classify_segment(clv_12m, p_purchase, top20, bottom40)
    assert segment in app.SEGMENT_ORDER
