WITH user_behavior AS (
  SELECT
    user_id,
    DATE_DIFF(@cutoff_date, MAX(DATE(created_at)), DAY) AS days_since_last_order,
    COUNT(order_id) AS total_orders,
    SUM(sale_price) AS total_spend,
    CASE WHEN COUNT(order_id) = 0 THEN 0 ELSE SUM(sale_price) / COUNT(order_id) END AS avg_order_value,
    COUNTIF(status = 'Returned') AS returned_orders,
    CASE WHEN COUNT(order_id) = 0 THEN 0 ELSE IEEE_DIVIDE(COUNTIF(status = 'Returned'), COUNT(order_id)) END AS return_rate
  FROM `bigquery-public-data.thelook_ecommerce.order_items`
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
  CASE WHEN f.user_id IS NULL THEN 1 ELSE 0 END AS is_churned
FROM user_behavior u
LEFT JOIN future_purchases f ON u.user_id = f.user_id
WHERE u.days_since_last_order < 365;