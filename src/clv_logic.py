"""
Pure business logic for CLV scoring and segmentation.

Extracted from app.py so that core calculations can be unit-tested
without importing Streamlit or loading model artifacts.
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HOLDOUT_DAYS = 183
P_PURCHASE_THRESHOLD = 0.20
CLV_TOP20_PCT = 0.80
CLV_BOTTOM40_PCT = 0.40

SEGMENT_ORDER = ["High Value", "Growing", "At-Risk", "Low Value"]

SEGMENT_CONFIG = {
    "High Value": {
        "color": "#2563EB",
        "action": "VIP loyalty, protect margin",
        "icon": "💎",
    },
    "Growing": {
        "color": "#16A34A",
        "action": "Personalized offers, invest in growth",
        "icon": "📈",
    },
    "At-Risk": {
        "color": "#EA580C",
        "action": "Win-back campaign, act fast",
        "icon": "⚠️",
    },
    "Low Value": {
        "color": "#9CA3AF",
        "action": "Email-only, minimal budget",
        "icon": "📧",
    },
}

DEFAULT_BUDGETS = {"High Value": 5, "Growing": 15, "At-Risk": 10, "Low Value": 2}

FEATURE_COLS = [
    "frequency",
    "recency",
    "T",
    "monetary_value",
    "avg_order_value",
    "unique_products",
    "avg_basket_size",
    "interpurchase_std",
    "is_one_time_buyer",
    "cancellation_rate",
    "recency_ratio",
    "country_enc",
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def classify_segment(
    clv_12m: float,
    p_purchase: float,
    top20_threshold: float,
    bottom40_threshold: float,
) -> str:
    """Assign a customer to one of four priority-ordered segments.

    Segment rules (evaluated in order):
        1. High Value  - CLV above the 80th-percentile threshold
        2. At-Risk     - purchase probability below P_PURCHASE_THRESHOLD (0.20)
        3. Growing     - CLV above the 40th-percentile threshold
        4. Low Value   - everyone else

    Parameters
    ----------
    clv_12m : float
        annualized predicted customer lifetime value.
    p_purchase : float
        Predicted probability of purchasing in the holdout window.
    top20_threshold : float
        CLV value at the 80th percentile of the portfolio.
    bottom40_threshold : float
        CLV value at the 40th percentile of the portfolio.

    Returns
    -------
    str
        One of "High Value", "Growing", "At-Risk", "Low Value".
    """
    if clv_12m > top20_threshold:
        return "High Value"
    elif p_purchase < P_PURCHASE_THRESHOLD:
        return "At-Risk"
    elif clv_12m > bottom40_threshold:
        return "Growing"
    else:
        return "Low Value"


def compute_clv_12m(p_purchase: float, expected_revenue: float) -> float:
    """Compute annualized CLV from purchase probability and expected revenue.

    Formula: P(purchase) × E[revenue | purchase] × (365 / HOLDOUT_DAYS)

    Parameters
    ----------
    p_purchase : float
        Predicted probability of purchasing in the holdout window.
    expected_revenue : float
        Expected revenue conditional on a purchase (from spend-tier estimate).

    Returns
    -------
    float
        annualized 12-month CLV estimate.
    """
    return p_purchase * expected_revenue * (365 / HOLDOUT_DAYS)


def assign_spend_tier(
    monetary_value: float,
    tier_low_threshold: float,
    tier_high_threshold: float,
) -> str:
    """Assign a customer to a spend tier based on monetary value terciles.

    Parameters
    ----------
    monetary_value : float
        Average order value for the customer.
    tier_low_threshold : float
        Upper bound for the Low Spend tier (33rd percentile).
    tier_high_threshold : float
        Upper bound for the Mid Spend tier (67th percentile).

    Returns
    -------
    str
        One of "Low Spend", "Mid Spend", "High Spend".
    """
    if monetary_value <= tier_low_threshold:
        return "Low Spend"
    elif monetary_value <= tier_high_threshold:
        return "Mid Spend"
    else:
        return "High Spend"


def compute_break_even_lift(budget: float, avg_clv: float) -> float | None:
    """Compute the break-even incremental lift for a campaign.

    break_even_lift = budget / avg_clv

    Parameters
    ----------
    budget : float
        Campaign spend per customer.
    avg_clv : float
        Average CLV for the target segment.

    Returns
    -------
    float or None
        Break-even lift as a fraction, or None if inputs are non-positive.
    """
    if budget <= 0 or avg_clv <= 0:
        return None
    return budget / avg_clv


def validate_customer_inputs(
    recency: float,
    T: float,
    frequency: int,
    cancellation_rate: float,
    is_one_time_buyer: int,
) -> list[str]:
    """Validate manual customer feature inputs and return a list of errors.

    Parameters
    ----------
    recency : float
        Days since last purchase.
    T : float
        Customer tenure in days.
    frequency : int
        Number of repeat purchases (0 = one-time buyer).
    cancellation_rate : float
        Proportion of invoices that were cancellations.
    is_one_time_buyer : int
        1 if customer made only one purchase, else 0.

    Returns
    -------
    list[str]
        List of human-readable validation error messages (empty if valid).
    """
    errors = []
    if recency > T:
        errors.append(f"Recency ({recency} days) cannot exceed tenure T ({T} days).")
    if T <= 0:
        errors.append("Tenure T must be positive.")
    if frequency < 0:
        errors.append("Frequency cannot be negative.")
    if not (0 <= cancellation_rate <= 1):
        errors.append("Cancellation rate must be between 0 and 1.")
    if is_one_time_buyer == 1 and frequency > 0:
        errors.append(
            "One-time buyer flag is set but frequency > 0. These are contradictory."
        )
    if is_one_time_buyer == 0 and frequency == 0:
        errors.append(
            "Frequency is 0 (no repeat purchases) but one-time buyer flag is not set."
        )
    return errors
