
-- "Naive" means it ignores adstock, saturation and seasonality, which is why the model is needed.
SELECT channel,ROUND(SUM(spend), 0)AS total_spend,
ROUND(100.0 * SUM(spend) / (SELECT SUM(spend) FROM ad_spend_clean), 1) AS spend_share_pct,
ROUND(AVG(spend), 0) AS avg_weekly_spend
FROM ad_spend_clean
GROUP BY channel
ORDER BY total_spend DESC;
