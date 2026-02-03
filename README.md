# High-Value Customer Churn Prediction

## Obejctive

The project's goal was to identify customers at risk of leaving (chruning) in the next 30 days to target them with retention campaigns. The goal is to move beyond simple accuracy and build a model that optimizes **profitability** by targeting only the users worth saving.

## Executive Summary

In e-commerce, acquiring a new cusotmer is 5x more expensive than retaining an existing one. This project built a maching learning solution to predict customer churn (users who stop buying) and optimize retention campaign budgets.

**Key Result:** The model identified an optimal targeting strategy that generates **$29,070 in projected profit** per campaign by targeting the 81% most at-risk users while avoding waste on the 19% of loyal "safe" customers.

## Technical Approach

- **Data Extraction:** SQL (BigQuery) with "Time Travel" split to prevent data leakage.
- **Handling Imbalance:** The dataset was 95% Churn / 5% Retained. Used **XGBoost with `scale_pos_weight`** and optimized for Recall on the minority class.
- **Business Solution:** Built a profit curve to determine the financial threshold for sending coupons, moving beyond simple "Accuracy" metrics.

## Key Insights

1. **The "Zone of Death":** Customer retention probability drops to near zero after **100 days** of inactivity. Immediate intervention is required between Day 30-60.
2. **Money Doesn't Buy Loyalty:** High spenders ("Whales") churn at the exact same rate as low spenders. VIP status is not a safety net.
3. **Strategic Targeting:** Spamming the entire user base wastes budget. The model saves money by filtering out the top 19% of highly loyal users who do not need a discount to stay.

## How to Run

1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Run the App:** `streamlit run src/app.py`
