"""
Unit tests for src/clv_logic.py

Run with:  pytest tests/test_clv_logic.py -v
"""

import pytest
from src.clv_logic import (
    HOLDOUT_DAYS,
    P_PURCHASE_THRESHOLD,
    SEGMENT_ORDER,
    SEGMENT_CONFIG,
    classify_segment,
    compute_clv_12m,
    assign_spend_tier,
    compute_break_even_lift,
    validate_customer_inputs,
)


# ---------------------------------------------------------------------------
# Fixtures - reusable thresholds that mirror a realistic portfolio
# ---------------------------------------------------------------------------
@pytest.fixture
def portfolio_thresholds():
    """Thresholds approximating the UCI Online Retail II dataset."""
    return {
        "top20": 500.0,  # 80th percentile CLV
        "bottom40": 100.0,  # 40th percentile CLV
    }


# ===========================================================================
# classify_segment
# ===========================================================================
class TestClassifySegment:
    """Tests for the 4-tier segmentation logic."""

    def test_high_value_above_top20(self, portfolio_thresholds):
        """CLV above 80th percentile → High Value regardless of p_purchase."""
        result = classify_segment(
            clv_12m=600.0,
            p_purchase=0.10,  # even low p_purchase shouldn't matter
            top20_threshold=portfolio_thresholds["top20"],
            bottom40_threshold=portfolio_thresholds["bottom40"],
        )
        assert result == "High Value"

    def test_at_risk_low_probability(self, portfolio_thresholds):
        """CLV between 40th–80th but p_purchase below threshold → At-Risk."""
        result = classify_segment(
            clv_12m=300.0,
            p_purchase=0.15,  # below 0.20 threshold
            top20_threshold=portfolio_thresholds["top20"],
            bottom40_threshold=portfolio_thresholds["bottom40"],
        )
        assert result == "At-Risk"

    def test_growing_mid_clv_good_probability(self, portfolio_thresholds):
        """CLV between 40th–80th percentile and p_purchase >= 0.20 → Growing."""
        result = classify_segment(
            clv_12m=300.0,
            p_purchase=0.50,
            top20_threshold=portfolio_thresholds["top20"],
            bottom40_threshold=portfolio_thresholds["bottom40"],
        )
        assert result == "Growing"

    def test_low_value_bottom_clv(self, portfolio_thresholds):
        """CLV below 40th percentile and p_purchase >= 0.20 → Low Value."""
        result = classify_segment(
            clv_12m=50.0,
            p_purchase=0.25,
            top20_threshold=portfolio_thresholds["top20"],
            bottom40_threshold=portfolio_thresholds["bottom40"],
        )
        assert result == "Low Value"

    def test_high_value_takes_priority_over_at_risk(self, portfolio_thresholds):
        """High CLV wins even if p_purchase is very low (priority order)."""
        result = classify_segment(
            clv_12m=501.0,
            p_purchase=0.01,
            top20_threshold=portfolio_thresholds["top20"],
            bottom40_threshold=portfolio_thresholds["bottom40"],
        )
        assert result == "High Value"

    def test_at_risk_takes_priority_over_low_value(self, portfolio_thresholds):
        """Low CLV + low p_purchase → At-Risk (not Low Value)."""
        result = classify_segment(
            clv_12m=50.0,
            p_purchase=0.10,
            top20_threshold=portfolio_thresholds["top20"],
            bottom40_threshold=portfolio_thresholds["bottom40"],
        )
        assert result == "At-Risk"

    def test_boundary_exactly_at_top20(self, portfolio_thresholds):
        """CLV exactly equal to threshold is NOT High Value (strict >)."""
        result = classify_segment(
            clv_12m=500.0,
            p_purchase=0.50,
            top20_threshold=portfolio_thresholds["top20"],
            bottom40_threshold=portfolio_thresholds["bottom40"],
        )
        assert result == "Growing"  # equal is not >, so falls through

    def test_boundary_exactly_at_p_purchase_threshold(self, portfolio_thresholds):
        """p_purchase exactly at 0.20 is NOT At-Risk (strict <)."""
        result = classify_segment(
            clv_12m=300.0,
            p_purchase=P_PURCHASE_THRESHOLD,  # 0.20 exactly
            top20_threshold=portfolio_thresholds["top20"],
            bottom40_threshold=portfolio_thresholds["bottom40"],
        )
        assert result == "Growing"  # 0.20 is not < 0.20

    def test_zero_clv_zero_probability(self, portfolio_thresholds):
        """Edge case: completely inactive customer."""
        result = classify_segment(
            clv_12m=0.0,
            p_purchase=0.0,
            top20_threshold=portfolio_thresholds["top20"],
            bottom40_threshold=portfolio_thresholds["bottom40"],
        )
        assert result == "At-Risk"  # p_purchase 0 < 0.20

    def test_result_always_in_valid_segments(self, portfolio_thresholds):
        """Verify output is always one of the four valid segments."""
        test_cases = [
            (1000, 0.99),
            (300, 0.05),
            (200, 0.50),
            (10, 0.80),
            (0, 0),
        ]
        for clv, p in test_cases:
            result = classify_segment(
                clv,
                p,
                portfolio_thresholds["top20"],
                portfolio_thresholds["bottom40"],
            )
            assert (
                result in SEGMENT_ORDER
            ), f"Unexpected segment '{result}' for clv={clv}, p={p}"


# ===========================================================================
# compute_clv_12m
# ===========================================================================
class TestComputeCLV:
    """Tests for the annualized CLV formula."""

    def test_basic_computation(self):
        """Verify P(purchase) × E[revenue] × (365/183)."""
        result = compute_clv_12m(p_purchase=0.5, expected_revenue=1000.0)
        expected = 0.5 * 1000.0 * (365 / HOLDOUT_DAYS)
        assert result == pytest.approx(expected)

    def test_zero_probability_gives_zero_clv(self):
        """If P(purchase) = 0, CLV must be 0 regardless of revenue."""
        assert compute_clv_12m(0.0, 5000.0) == 0.0

    def test_zero_revenue_gives_zero_clv(self):
        """If expected revenue = 0, CLV must be 0."""
        assert compute_clv_12m(0.8, 0.0) == 0.0

    def test_certain_purchase(self):
        """P(purchase) = 1.0 → CLV is just annualized revenue."""
        result = compute_clv_12m(1.0, 851.0)
        expected = 851.0 * (365 / HOLDOUT_DAYS)
        assert result == pytest.approx(expected)

    def test_clv_is_non_negative(self):
        """CLV should never be negative for valid inputs."""
        assert compute_clv_12m(0.3, 402.0) >= 0

    def test_annualization_factor(self):
        """The 365/183 factor should be approximately 1.995."""
        factor = 365 / HOLDOUT_DAYS
        assert factor == pytest.approx(1.9945, rel=1e-3)


# ===========================================================================
# assign_spend_tier
# ===========================================================================
class TestAssignSpendTier:
    """Tests for spend-tier assignment via tercile thresholds."""

    def test_low_spend(self):
        assert assign_spend_tier(100.0, 200.0, 600.0) == "Low Spend"

    def test_mid_spend(self):
        assert assign_spend_tier(400.0, 200.0, 600.0) == "Mid Spend"

    def test_high_spend(self):
        assert assign_spend_tier(800.0, 200.0, 600.0) == "High Spend"

    def test_boundary_at_low_threshold(self):
        """Exactly at the low threshold → Low Spend (<=)."""
        assert assign_spend_tier(200.0, 200.0, 600.0) == "Low Spend"

    def test_boundary_at_high_threshold(self):
        """Exactly at the high threshold → Mid Spend (<=)."""
        assert assign_spend_tier(600.0, 200.0, 600.0) == "Mid Spend"

    def test_just_above_high_threshold(self):
        assert assign_spend_tier(600.01, 200.0, 600.0) == "High Spend"

    def test_zero_monetary_value(self):
        """A customer with $0 avg order value should be Low Spend."""
        assert assign_spend_tier(0.0, 200.0, 600.0) == "Low Spend"


# ===========================================================================
# compute_break_even_lift
# ===========================================================================
class TestBreakEvenLift:
    """Tests for campaign break-even calculations."""

    def test_basic_break_even(self):
        """$10 budget / $500 avg CLV = 2% lift needed."""
        result = compute_break_even_lift(10.0, 500.0)
        assert result == pytest.approx(0.02)

    def test_zero_budget_returns_none(self):
        assert compute_break_even_lift(0.0, 500.0) is None

    def test_zero_clv_returns_none(self):
        assert compute_break_even_lift(10.0, 0.0) is None

    def test_negative_budget_returns_none(self):
        assert compute_break_even_lift(-5.0, 500.0) is None

    def test_high_budget_relative_to_clv(self):
        """Budget exceeding CLV → lift > 100%, still valid calculation."""
        result = compute_break_even_lift(600.0, 500.0)
        assert result == pytest.approx(1.2)


# ===========================================================================
# validate_customer_inputs
# ===========================================================================
class TestValidateCustomerInputs:
    """Tests for manual input validation."""

    def test_valid_inputs_return_no_errors(self):
        errors = validate_customer_inputs(
            recency=60,
            T=365,
            frequency=3,
            cancellation_rate=0.05,
            is_one_time_buyer=0,
        )
        assert errors == []

    def test_recency_exceeds_tenure(self):
        errors = validate_customer_inputs(
            recency=400,
            T=365,
            frequency=3,
            cancellation_rate=0.05,
            is_one_time_buyer=0,
        )
        assert any("Recency" in e and "tenure" in e for e in errors)

    def test_zero_tenure(self):
        errors = validate_customer_inputs(
            recency=0,
            T=0,
            frequency=0,
            cancellation_rate=0.0,
            is_one_time_buyer=1,
        )
        assert any("Tenure" in e for e in errors)

    def test_negative_frequency(self):
        errors = validate_customer_inputs(
            recency=30,
            T=365,
            frequency=-1,
            cancellation_rate=0.05,
            is_one_time_buyer=0,
        )
        assert any("Frequency" in e for e in errors)

    def test_cancellation_rate_out_of_range(self):
        errors = validate_customer_inputs(
            recency=30,
            T=365,
            frequency=3,
            cancellation_rate=1.5,
            is_one_time_buyer=0,
        )
        assert any("Cancellation" in e for e in errors)

    def test_one_time_buyer_contradiction_freq_positive(self):
        """Flag = 1 (one-time) but frequency > 0 (has repeat purchases)."""
        errors = validate_customer_inputs(
            recency=30,
            T=365,
            frequency=3,
            cancellation_rate=0.05,
            is_one_time_buyer=1,
        )
        assert any("contradictory" in e.lower() for e in errors)

    def test_repeat_buyer_contradiction_freq_zero(self):
        """Flag = 0 (repeat) but frequency = 0 (no repeat purchases)."""
        errors = validate_customer_inputs(
            recency=30,
            T=365,
            frequency=0,
            cancellation_rate=0.05,
            is_one_time_buyer=0,
        )
        assert any("one-time buyer flag" in e.lower() for e in errors)

    def test_valid_one_time_buyer(self):
        """Consistent one-time buyer: flag=1, frequency=0."""
        errors = validate_customer_inputs(
            recency=30,
            T=365,
            frequency=0,
            cancellation_rate=0.0,
            is_one_time_buyer=1,
        )
        assert errors == []

    def test_multiple_errors_returned(self):
        """Should catch all problems, not just the first one."""
        errors = validate_customer_inputs(
            recency=500,
            T=100,
            frequency=-1,
            cancellation_rate=2.0,
            is_one_time_buyer=1,
        )
        assert len(errors) >= 3


# ===========================================================================
# Constants sanity checks
# ===========================================================================
class TestConstants:
    """Verify critical constants haven't been accidentally changed."""

    def test_holdout_days(self):
        assert HOLDOUT_DAYS == 183

    def test_p_purchase_threshold(self):
        assert P_PURCHASE_THRESHOLD == 0.20

    def test_all_segments_have_config(self):
        for seg in SEGMENT_ORDER:
            assert seg in SEGMENT_CONFIG
            assert "color" in SEGMENT_CONFIG[seg]
            assert "action" in SEGMENT_CONFIG[seg]
            assert "icon" in SEGMENT_CONFIG[seg]

    def test_segment_order_length(self):
        assert len(SEGMENT_ORDER) == 4
