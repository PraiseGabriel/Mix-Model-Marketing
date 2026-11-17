
SELECT channel, COUNT(*) AS weeks, COUNT(DISTINCT week_start) AS distinct_weeks
FROM ad_spend_clean
GROUP BY channel;

SELECT week_start, channel, COUNT(*) AS n
FROM ad_spend_clean
GROUP BY week_start, channel
HAVING COUNT(*) > 1;


SELECT COUNT(*) AS bad_spend_rows
FROM ad_spend_clean
WHERE spend IS NULL OR spend < 0;


SELECT s.week_start, COUNT(a.channel) AS channels_with_spend
FROM raw_sales s
LEFT JOIN ad_spend_clean a ON a.week_start = s.week_start
GROUP BY s.week_start
HAVING COUNT(a.channel) <> 3;
