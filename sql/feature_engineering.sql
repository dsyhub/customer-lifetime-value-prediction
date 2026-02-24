/*
  CLV Feature Engineering: BG/NBD + Gamma-Gamma Inputs
  (formerly: churn prediction feature engineering)

  See sql/clv_features.sql for the canonical version with full documentation.
  This file is retained for backwards compatibility with older notebook references.
*/

-- Redirect to clv_features.sql content below:

WITH calibration_order_level AS (
  SELECT
    user_id,
    order_id,
    DATE(MIN(created_at)) AS order_date,
    SUM(sale_price)       AS order_revenue
  FROM `bigquery-public-data.thelook_ecommerce.order_items`
  WHERE DATE(created_at) < @calibration_end
  GROUP BY user_id, order_id
),

calibration_stats AS (
  SELECT
    user_id,
    COUNT(order_id)       AS total_orders,
    COUNT(order_id) - 1   AS frequency,
    SUM(order_revenue)    AS total_spend,
    AVG(order_revenue)    AS avg_order_value,
    MIN(order_date)       AS first_order_date,
    MAX(order_date)       AS last_order_date,
    DATE_DIFF(@calibration_end, MIN(order_date), DAY) AS T,
    DATE_DIFF(MAX(order_date), MIN(order_date), DAY)  AS recency,
    DATE_DIFF(@calibration_end, MAX(order_date), DAY) AS days_since_last_order
  FROM calibration_order_level
  GROUP BY user_id
),

repeat_monetary AS (
  SELECT
    ol.user_id,
    AVG(ol.order_revenue) AS repeat_avg_spend
  FROM calibration_order_level ol
  JOIN calibration_stats cs ON ol.user_id = cs.user_id
  WHERE ol.order_date > cs.first_order_date
  GROUP BY ol.user_id
),

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

user_engagement AS (
  SELECT
    user_id,
    COUNT(DISTINCT session_id)                              AS total_sessions,
    COUNT(*)                                                AS total_events,
    DATE_DIFF(@calibration_end, MAX(DATE(created_at)), DAY) AS days_since_last_visit,
    SAFE_DIVIDE(COUNT(*), COUNT(DISTINCT session_id))       AS avg_events_per_session,
    COUNT(DISTINCT event_type)                              AS distinct_event_types,
    COUNTIF(event_type = 'cart')                            AS cart_events,
    COUNTIF(event_type = 'product')                         AS product_view_events,
    COUNTIF(event_type = 'purchase')                        AS purchase_events
  FROM `bigquery-public-data.thelook_ecommerce.events`
  WHERE DATE(created_at) < @calibration_end
  GROUP BY user_id
)

SELECT
  cs.user_id,
  cs.frequency,
  cs.recency,
  cs.T,
  COALESCE(rm.repeat_avg_spend, cs.avg_order_value) AS monetary_value,
  cs.total_orders,
  cs.total_spend,
  cs.avg_order_value,
  cs.days_since_last_order,
  COALESCE(hs.actual_holdout_transactions, 0) AS actual_holdout_transactions,
  COALESCE(hs.actual_holdout_revenue, 0.0)    AS actual_holdout_revenue,
  ud.customer_tenure_days,
  ud.age,
  ud.gender,
  ud.traffic_source,
  ud.country,
  COALESCE(ue.total_sessions,          0) AS total_sessions,
  COALESCE(ue.total_events,            0) AS total_events,
  COALESCE(ue.days_since_last_visit, 365) AS days_since_last_visit,
  COALESCE(ue.avg_events_per_session,  0) AS avg_events_per_session,
  COALESCE(ue.distinct_event_types,    0) AS distinct_event_types,
  COALESCE(ue.cart_events,             0) AS cart_events,
  COALESCE(ue.product_view_events,     0) AS product_view_events,
  COALESCE(ue.purchase_events,         0) AS purchase_events

FROM calibration_stats cs
LEFT JOIN repeat_monetary   rm ON cs.user_id = rm.user_id
LEFT JOIN holdout_stats      hs ON cs.user_id = hs.user_id
LEFT JOIN user_demographics  ud ON cs.user_id = ud.user_id
LEFT JOIN user_engagement    ue ON cs.user_id = ue.user_id

WHERE cs.T > 7
  AND COALESCE(rm.repeat_avg_spend, cs.avg_order_value) > 0;
