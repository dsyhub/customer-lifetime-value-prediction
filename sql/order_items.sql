SELECT *
FROM `bigquery-public-data.thelook_ecommerce.order_items`
WHERE DATE(created_at) < @cutoff_date