"""Tests for segment classification logic and constants consistency."""

import pytest


# ---------------------------------------------------------------------------
# classify_segment: covers all 4 segments + priority edge case
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "clv, p_purchase, expected",
    [
        (200, 0.80, "High Value"),      # CLV above top20 threshold
        (80, 0.10, "At-Risk"),          # low p_purchase, not high value
        (80, 0.50, "Growing"),          # middle CLV, active
        (30, 0.50, "Low Value"),        # bottom CLV, active
    ],
    ids=["high_value", "at_risk", "growing", "low_value"],
)
def test_classify_segment_all_tiers(classify_segment, clv, p_purchase, expected):
    # top20_threshold=100, bottom40_threshold=50
    assert classify_segment(clv, p_purchase, 100, 50) == expected


def test_high_value_overrides_at_risk(classify_segment):
    """High CLV takes priority even when p_purchase is below 0.20."""
    result = classify_segment(clv_12m=200, p_purchase=0.05, top20_threshold=100, bottom40_threshold=50)
    assert result == "High Value"


def test_boundary_at_top20_threshold(classify_segment):
    """CLV exactly at top20 threshold uses '>' so it falls through."""
    result = classify_segment(clv_12m=100, p_purchase=0.50, top20_threshold=100, bottom40_threshold=50)
    assert result == "Growing"


def test_boundary_p_purchase_at_020(classify_segment):
    """p_purchase exactly at 0.20 uses '<' so it is NOT At-Risk."""
    result = classify_segment(clv_12m=80, p_purchase=0.20, top20_threshold=100, bottom40_threshold=50)
    assert result == "Growing"


# ---------------------------------------------------------------------------
# Constants consistency
# ---------------------------------------------------------------------------
def test_segment_constants_consistent(app):
    """All entries in SEGMENT_ORDER exist in SEGMENT_CONFIG and DEFAULT_BUDGETS."""
    for seg in app.SEGMENT_ORDER:
        assert seg in app.SEGMENT_CONFIG, f"{seg} missing from SEGMENT_CONFIG"
        assert seg in app.DEFAULT_BUDGETS, f"{seg} missing from DEFAULT_BUDGETS"
