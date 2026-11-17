
-- This table is the input to the model.
DROP TABLE IF EXISTS model_input;
CREATE TABLE model_input AS
SELECT
    s.week_start,
    s.revenue,
    s.promo_flag,
    SUM(CASE WHEN a.channel = 'amazon_ads' THEN a.spend END) AS amazon_ads,
    SUM(CASE WHEN a.channel = 'google_ads' THEN a.spend END) AS google_ads,
    SUM(CASE WHEN a.channel = 'meta_ads'   THEN a.spend END) AS meta_ads
FROM raw_sales s
JOIN ad_spend_clean a ON a.week_start = s.week_start
GROUP BY s.week_start, s.revenue, s.promo_flag
ORDER BY s.week_start;
