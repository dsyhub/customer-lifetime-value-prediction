WITH user_behavior AS (
  SELECT
    user_id,
    -- Feature 1: Recency (Days since last order before cutoff)
    DATE_DIFF(@cutoff_date, MAX(DATE(created_at)), DAY) AS days_since_last_order,
    
    -- Feature 2: Frequency (Total orders before cutoff)
    COUNT(order_id) AS total_orders,
    
    -- Feature 3: Monetary (Total spend before cutoff)
    SUM(sale_price) AS total_spend,
    
    -- Feature 4: Average Order Value (AOV)
    CASE WHEN COUNT(order_id) = 0 THEN 0 ELSE SUM(sale_price) / COUNT(order_id) END as avg_order_value,
    
    -- Feature 5: Returns (Risk Signal)
    COUNTIF(status = 'Returned') as returned_orders,
    CASE WHEN COUNT(order_id) = 0 THEN 0 ELSE IEEE_DIVIDE(COUNTIF(status = 'Returned'), COUNT(order_id)) END as return_rate
  
  FROM `bigquery-public-data.thelook_ecommerce.order_items`
  WHERE DATE(created_at) < @cutoff_date  -- strict "past" only
  GROUP BY 1
),

user_demographics AS (
  SELECT
    id AS user_id,
    DATE_DIFF(@cutoff_date, DATE(created_at), DAY) AS customer_tenure_days,
    age,
    gender,
    traffic_source,
    country
  FROM `bigquery-public-data.thelook_ecommerce.users`
),

user_engagement AS (
  SELECT
    user_id,

    -- Session metrics
    COUNT(DISTINCT session_id) AS total_sessions,
    COUNT(*) AS total_events,
    DATE_DIFF(@cutoff_date, MAX(DATE(created_at)), DAY) AS days_since_last_visit,
    SAFE_DIVIDE(COUNT(*), COUNT(DISTINCT session_id)) AS avg_events_per_session,

    -- Event type breakdown (behavioral signals)
    -- department, product, purchase, home, cart
    COUNT(DISTINCT event_type) AS distinct_event_types,
    COUNTIF(event_type = 'cart') AS cart_events,
    COUNTIF(event_type = 'product') AS product_view_events,
    COUNTIF(event_type = 'purchase') AS purchase_events,

    -- Intent ratio
    SAFE_DIVIDE(COUNTIF(event_type = 'cart'), NULLIF(COUNTIF(event_type = 'product'), 0)) AS browse_to_cart_ratio
    
  FROM `bigquery-public-data.thelook_ecommerce.events`
  WHERE DATE(created_at) < @cutoff_date
  GROUP BY 1
),


future_purchases AS (
  SELECT DISTINCT user_id
  FROM `bigquery-public-data.thelook_ecommerce.orders`
  WHERE DATE(created_at) BETWEEN @cutoff_date
    AND DATE_ADD(@cutoff_date, INTERVAL @prediction_window_days DAY)
)

SELECT
  u.*,
  -- Target: 1 if they did NOT buy in the prediction window (Churned), 0 if they did (Retained)
  CASE WHEN f.user_id IS NULL THEN 1 ELSE 0 END AS is_churned
FROM user_behavior u
LEFT JOIN future_purchases f ON u.user_id = f.user_id
-- Filter: Only train on users who were active within the last year
-- This removes "ancient" users who churned 5 years ago, keeping the model focused on recent behavior
WHERE u.days_since_last_order < 365;


-- Additional features for a more comprehensive model