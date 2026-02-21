/* 
  Feature Engineering: Customer Retention Propensity Model

  This query builds a customer-level feature set for predicting repeat purchase
  probability within a 90-days window. Features are computed strictly before the
  cutoff date to prevent data leakage.

  Data Sources:
    - order_items: Transaction history (RFM features)
    - users: Demographics and acquisition channel
    - events: Behavioral engagement signals
    - orders: Target variable (future purchases)

  Parameters:
    @cutoff_date: Point-in_time for train/test split (simulated "today")
    @predication_window_days: Days after cutoff to define retention (default: 90)
 */


-- TODO:
-- 1. Clearly define the "Objective"
-- 2. Determine the target variable & definition of "Churn"
-- 3. EDA
-- 4. Add additional features as needed

-- =============================================================================
-- CTE 1: TRANSACTIONAL FEATURES (RFM + Returns)
-- Source: order_items table
-- =============================================================================
WITH user_behavior AS (
  SELECT
    user_id,
    
    -- Recency: Days since last order (lower = more engaged)
    DATE_DIFF(@cutoff_date, MAX(DATE(created_at)), DAY) AS days_since_last_order,
    
    -- Frequency: Total order count
    COUNT(order_id) AS total_orders,
    
    -- Monetary: Lifetime spend
    SUM(sale_price) AS total_spend,
    
    -- Average Order Value
    SAFE_DIVIDE(SUM(sale_price), COUNT(order_id)) AS avg_order_value,
    
    -- Returns: Potential dissatisfaction signal
    COUNTIF(status = 'Returned') AS returned_orders,
    SAFE_DIVIDE(COUNTIF(status = 'Returned'), COUNT(order_id)) AS return_rate
  
  FROM `bigquery-public-data.thelook_ecommerce.order_items`
  WHERE DATE(created_at) < @cutoff_date  -- Strict point-in-time: no future leakage
  GROUP BY 1
),

-- =============================================================================
-- CTE 2: DEMOGRAPHIC FEATURES
-- Source: users table
-- =============================================================================
user_demographics AS (
  SELECT
    id AS user_id,
    
    -- Tenure: Days since account creation
    DATE_DIFF(@cutoff_date, DATE(created_at), DAY) AS customer_tenure_days,
    
    -- Demographics
    age,
    gender,
    
    -- Acquisition channel (organic vs. paid)
    traffic_source,
    
    -- Geographic market
    country
    
  FROM `bigquery-public-data.thelook_ecommerce.users`
),

-- =============================================================================
-- CTE 3: ENGAGEMENT FEATURES
-- Source: events table (browsing behavior)
-- =============================================================================
user_engagement AS (
  SELECT
    user_id,

    -- Session volume
    COUNT(DISTINCT session_id) AS total_sessions,
    COUNT(*) AS total_events,
    
    -- Engagement recency (more sensitive than order recency)
    DATE_DIFF(@cutoff_date, MAX(DATE(created_at)), DAY) AS days_since_last_visit,
    
    -- Session depth: Average events per session
    SAFE_DIVIDE(COUNT(*), COUNT(DISTINCT session_id)) AS avg_events_per_session,

    -- Behavioral diversity
    COUNT(DISTINCT event_type) AS distinct_event_types,

    -- Funnel events (intent signals) 
    COUNTIF(event_type = 'cart') AS cart_events,
    COUNTIF(event_type = 'product') AS product_view_events,
    COUNTIF(event_type = 'purchase') AS purchase_events,

    -- -- Browse-to-cart ratio: Measures purchase intent
    -- SAFE_DIVIDE(COUNTIF(event_type = 'cart'), NULLIF(COUNTIF(event_type = 'product'), 0)) AS browse_to_cart_ratio
    
  FROM `bigquery-public-data.thelook_ecommerce.events`
  WHERE DATE(created_at) < @cutoff_date   -- Strict point-in-time
  GROUP BY 1
),

-- =============================================================================
-- CTE 4: TARGET VARIABLE
-- Definition: Did user make a purchase within the prediction window?
-- =============================================================================
future_purchases AS (
  SELECT DISTINCT user_id
  FROM `bigquery-public-data.thelook_ecommerce.orders`
  WHERE DATE(created_at) BETWEEN @cutoff_date
    AND DATE_ADD(@cutoff_date, INTERVAL @prediction_window_days DAY)
)

-- =============================================================================
-- FINAL SELECT: Join all features + target
-- =============================================================================
SELECT
  -- Primary key
  ub.user_id,
  
  -- Transactional features (RFM)
  ub.days_since_last_order,
  ub.total_orders,
  ub.total_spend,
  ub.avg_order_value,
  ub.returned_orders,
  ub.return_rate,

  -- Demographic features
  ud.customer_tenure_days,
  ud.age,
  ud.gender,
  ud.traffic_source,
  ud.country,

  -- Engagement features
  COALESCE(ue.total_sessions, 0) AS total_sessions,
  COALESCE(ue.total_events, 0) AS total_events,
  COALESCE(ue.days_since_last_visit, 365) AS days_since_last_visit,  -- Default: no visit
  COALESCE(ue.avg_events_per_session, 0) AS avg_events_per_session,
  COALESCE(ue.distinct_event_types, 0) AS distinct_event_types,
  COALESCE(ue.cart_events, 0) AS cart_events,
  COALESCE(ue.product_view_events, 0) AS product_view_events,
  COALESCE(ue.purchase_events, 0) AS purchase_events,

  -- Target: 1 = made repeat purchase (retained), 0 = no purchase (churned)
  CASE WHEN fp.user_id IS NOT NULL THEN 1 ELSE 0 END AS is_retained

FROM user_behavior ub
LEFT JOIN user_demographics ud ON ub.user_id = ud.user_id
LEFT JOIN user_engagement ue ON ub.user_id = ue.user_id
LEFT JOIN future_purchases fp ON ub.user_id = fp.user_id

-- Filter: Only include users active within the past year
-- Rationale: Users inactive >1 year are already lost; including them adds noise
WHERE ub.days_since_last_order < 365;


