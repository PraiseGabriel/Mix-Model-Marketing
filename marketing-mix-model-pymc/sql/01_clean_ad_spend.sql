DROP TABLE IF EXISTS ad_spend_clean;
CREATE TABLE ad_spend_clean AS
SELECT week_start,LOWER(REPLACE(TRIM(channel), ' ', '_')) AS channel,
MAX(spend)AS spend   -- duplicates are identical, so MAX keeps one value
FROM raw_ad_spend
GROUP BY week_start, LOWER(REPLACE(TRIM(channel), ' ', '_'));
