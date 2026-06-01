-- ====================================================================
-- ML Galaxy Portfolio - Advanced SQL Feature Engineering (2026 Edition)
-- Targets snowflake, BigQuery, and Redshift data warehouses.
-- Demonstrates rolling stats, customer cohort profiles, and outlier alerts.
-- ====================================================================

-- 1. Rolling Transaction Metrics & Volatilities
WITH base_transactions AS (
    SELECT 
        user_id,
        transaction_timestamp,
        amount,
        -- Extract time features
        EXTRACT(HOUR FROM transaction_timestamp) AS txn_hour,
        EXTRACT(DAYOFWEEK FROM transaction_timestamp) AS txn_dayofweek
    FROM production.user_transactions
    WHERE transaction_timestamp >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
),

rolling_features AS (
    SELECT 
        user_id,
        transaction_timestamp,
        amount,
        txn_hour,
        txn_dayofweek,
        -- 30-Day moving average transaction scale
        AVG(amount) OVER (
            PARTITION BY user_id 
            ORDER BY transaction_timestamp 
            RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW
        ) AS mvg_avg_30d,
        -- 30-Day moving standard deviation for risk analysis
        STDDEV(amount) OVER (
            PARTITION BY user_id 
            ORDER BY transaction_timestamp 
            RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW
        ) AS mvg_std_30d,
        -- Frequency metrics
        COUNT(amount) OVER (
            PARTITION BY user_id 
            ORDER BY transaction_timestamp 
            RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW
        ) AS txn_count_30d
    FROM base_transactions
)

-- 2. RFM (Recency, Frequency, Monetary) Profiles & Anomaly Scoring
SELECT 
    user_id,
    transaction_timestamp,
    amount,
    mvg_avg_30d,
    mvg_std_30d,
    txn_count_30d,
    -- Calculate immediate Z-Score anomaly flag
    CASE 
        WHEN mvg_std_30d > 0 THEN (amount - mvg_avg_30d) / mvg_std_30d 
        ELSE 0 
    END AS amount_z_score_flag,
    -- Alert if transaction is 3 standard deviations out of bound
    CASE 
        WHEN ABS(CASE WHEN mvg_std_30d > 0 THEN (amount - mvg_avg_30d) / mvg_std_30d ELSE 0 END) >= 3.0 THEN 1 
        ELSE 0 
    END AS anomaly_flag
FROM rolling_features
ORDER BY user_id, transaction_timestamp DESC;
