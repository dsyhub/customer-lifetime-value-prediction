/*
  CLV Feature Engineering: BG/NBD + Gamma-Gamma Inputs

  Computes customer-level features for Customer Lifetime Value (CLV) prediction
  using the Buy-Till-You-Die (BG/NBD) and Gamma-Gamma probabilistic models.

  Data Sources:
    - order_items: Transaction history (BG/NBD inputs + holdout labels)
    - users:       Demographics and acquisition channel
    - events:      Behavioral engagement signals

  Parameters:
    @cutoff_date:       End of holdout window (simulated "today")    → 2025-10-01
    @calibration_end:   End of calibration period (model training cutoff) → 2025-04-04
                        (cutoff_date minus 180 days)

  BG/NBD Core Inputs (computed at calibration_end):
    frequency      = total distinct orders - 1   (repeat purchases; 0 = one-time buyer)
    recency        = days from first to last purchase within calibration period
    T              = days from first purchase to calibration_end (customer age)
    monetary_value = avg order value across repeat transactions
                     (fallback to avg_order_value for one-time buyers)

  Holdout Labels (for temporal validation):
    actual_holdout_transactions = orders placed between calibration_end and cutoff_date
    actual_holdout_revenue      = revenue between calibration_end and cutoff_date

  Filter: T > 7 (exclude customers with fewer than 7 days of purchase history)
  Output: data/raw/clv_data.csv
*/


-- =============================================================================
-- CTE 1: ORDER-LEVEL AGGREGATION (calibration period)
-- Collapse order_items rows to one row per (user, order) to get true order count
-- =============================================================================
WITH calibration_order_level AS (
  SELECT
    user_id,
    order_id,
    DATE(MIN(created_at)) AS order_date,
    SUM(sale_price)       AS order_revenue

  FROM `bigquery-public-data.thelook_ecommerce.order_items`
  WHERE DATE(created_at) < @calibration_end    -- Strict point-in-time
  GROUP BY user_id, order_id
),

-- =============================================================================
-- CTE 2: CUSTOMER-LEVEL BG/NBD STATS (calibration period)
-- =============================================================================
calibration_stats AS (
  SELECT
    user_id,

    -- Transaction counts
    COUNT(order_id)       AS total_orders,      -- distinct orders (not items)
    COUNT(order_id) - 1   AS frequency,         -- repeat purchases (0 = one-time buyer)

    -- Revenue
    SUM(order_revenue)    AS total_spend,
    AVG(order_revenue)    AS avg_order_value,

    -- Dates
    MIN(order_date)       AS first_order_date,
    MAX(order_date)       AS last_order_date,

    -- BG/NBD inputs
    DATE_DIFF(@calibration_end, MIN(order_date), DAY) AS T,          -- customer age
    DATE_DIFF(MAX(order_date), MIN(order_date), DAY)  AS recency,    -- first→last purchase
    DATE_DIFF(@calibration_end, MAX(order_date), DAY) AS days_since_last_order

  FROM calibration_order_level
  GROUP BY user_id
),

-- =============================================================================
-- CTE 3: MONETARY VALUE FOR REPEAT TRANSACTIONS
-- Gamma-Gamma requires avg spend on repeat (non-first) transactions
-- =============================================================================
repeat_monetary AS (
  SELECT
    ol.user_id,
    AVG(ol.order_revenue) AS repeat_avg_spend

  FROM calibration_order_level ol
  JOIN calibration_stats cs ON ol.user_id = cs.user_id
  WHERE ol.order_date > cs.first_order_date   -- exclude first purchase
  GROUP BY ol.user_id
),

-- =============================================================================
-- CTE 4: HOLDOUT PERIOD LABELS (calibration_end → cutoff_date)
-- Used for temporal validation in notebook 03
-- =============================================================================
holdout_stats AS (
  SELECT
    user_id,
    COUNT(DISTINCT order_id) AS actual_holdout_transactions,
    SUM(sale_price)          AS actual_holdout_revenue

  FROM `bigquery-public-data.thelook_ecommerce.order_items`
  WHERE DATE(created_at) >= @calibration_end
    AND DATE(created_at) < @cutoff_date
  GROUP BY user_id
),

-- =============================================================================
-- CTE 5: DEMOGRAPHIC FEATURES
-- Source: users table (measured at calibration_end for point-in-time correctness)
-- =============================================================================
user_demographics AS (
  SELECT
    id AS user_id,
    DATE_DIFF(@calibration_end, DATE(created_at), DAY) AS customer_tenure_days,
    age,
    gender,
    traffic_source,
    country

  FROM `bigquery-public-data.thelook_ecommerce.users`
),

-- =============================================================================
-- CTE 6: BEHAVIORAL ENGAGEMENT FEATURES
-- Source: events table (pre-calibration browsing signals)
-- =============================================================================
user_engagement AS (
  SELECT
    user_id,
    COUNT(DISTINCT session_id)                                         AS total_sessions,
    COUNT(*)                                                           AS total_events,
    DATE_DIFF(@calibration_end, MAX(DATE(created_at)), DAY)            AS days_since_last_visit,
    SAFE_DIVIDE(COUNT(*), COUNT(DISTINCT session_id))                  AS avg_events_per_session,
    COUNT(DISTINCT event_type)                                         AS distinct_event_types,
    COUNTIF(event_type = 'cart')                                       AS cart_events,
    COUNTIF(event_type = 'product')                                    AS product_view_events,
    COUNTIF(event_type = 'purchase')                                   AS purchase_events

  FROM `bigquery-public-data.thelook_ecommerce.events`
  WHERE DATE(created_at) < @calibration_end    -- Strict point-in-time
  GROUP BY user_id
)

-- =============================================================================
-- FINAL SELECT: BG/NBD inputs + holdout labels + context features
-- =============================================================================
SELECT
  cs.user_id,

  -- -------------------------------------------------------------------------
  -- BG/NBD CORE INPUTS
  -- -------------------------------------------------------------------------
  cs.frequency,                                           -- 0 = one-time buyer
  cs.recency,                                             -- days first → last purchase
  cs.T,                                                   -- customer age at calibration_end

  -- monetary_value: avg per REPEAT transaction; fallback to AOV for freq=0
  COALESCE(rm.repeat_avg_spend, cs.avg_order_value) AS monetary_value,

  -- -------------------------------------------------------------------------
  -- TRANSACTION CONTEXT (for XGBoost + sanity checks)
  -- -------------------------------------------------------------------------
  cs.total_orders,
  cs.total_spend,
  cs.avg_order_value,
  cs.days_since_last_order,

  -- -------------------------------------------------------------------------
  -- HOLDOUT LABELS (temporal validation)
  -- -------------------------------------------------------------------------
  COALESCE(hs.actual_holdout_transactions, 0) AS actual_holdout_transactions,
  COALESCE(hs.actual_holdout_revenue, 0.0)    AS actual_holdout_revenue,

  -- -------------------------------------------------------------------------
  -- DEMOGRAPHIC FEATURES
  -- -------------------------------------------------------------------------
  ud.customer_tenure_days,
  ud.age,
  ud.gender,
  ud.traffic_source,
  ud.country,

  -- -------------------------------------------------------------------------
  -- ENGAGEMENT FEATURES
  -- -------------------------------------------------------------------------
  COALESCE(ue.total_sessions,          0)   AS total_sessions,
  COALESCE(ue.total_events,            0)   AS total_events,
  COALESCE(ue.days_since_last_visit, 365)   AS days_since_last_visit,
  COALESCE(ue.avg_events_per_session,  0)   AS avg_events_per_session,
  COALESCE(ue.distinct_event_types,    0)   AS distinct_event_types,
  COALESCE(ue.cart_events,             0)   AS cart_events,
  COALESCE(ue.product_view_events,     0)   AS product_view_events,
  COALESCE(ue.purchase_events,         0)   AS purchase_events

FROM calibration_stats cs
LEFT JOIN repeat_monetary   rm ON cs.user_id = rm.user_id
LEFT JOIN holdout_stats      hs ON cs.user_id = hs.user_id
LEFT JOIN user_demographics  ud ON cs.user_id = ud.user_id
LEFT JOIN user_engagement    ue ON cs.user_id = ue.user_id

-- Filter: exclude very new customers (< 7 days of history)
-- and ensure monetary_value is positive (Gamma-Gamma requirement)
WHERE cs.T > 7
  AND COALESCE(rm.repeat_avg_spend, cs.avg_order_value) > 0;
