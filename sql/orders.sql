-- SELECT DISTINCT user_id
SELECT *
FROM `bigquery-public-data.thelook_ecommerce.orders`
WHERE DATE(created_at) BETWEEN @cutoff_date
  AND DATE_ADD(@cutoff_date, INTERVAL @prediction_window_days DAY)