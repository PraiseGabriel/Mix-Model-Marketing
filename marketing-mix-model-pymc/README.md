Marketing Mix Model in PyMC

I put this project together to show how I approach marketing mix modeling (MMM). I've done this kind of work on real e-commerce campaigns, using Amazon Ads and Google Ads data, but I'm not allowed to share that work or its data. So I rebuilt the method on simulated data instead.

Everything in this repo is simulated. There is no client or employer data in it. Simulating also has an upside: because I set the true effect of every channel myself, I can check whether the model actually finds them. You can't do that with real data.

What's in here

I wanted it to cover the whole workflow, not just the model:

Simulate the data (src/generate_data.py). Two years of weekly data for three channels (Amazon Ads, Google Ads, Meta Ads), with known adstock decay, saturation and channel effects, plus trend, seasonality and promotions. I made the raw spend feed messy on purpose, with inconsistent channel names and duplicate rows, because real feeds are.
Clean it with SQL (sql/ and src/run_sql.py). The raw files go into SQLite, get cleaned, pass a set of data quality checks, and end up as one model-ready table.
Fit the model (src/mmm.py). A Bayesian MMM in PyMC with adstock, Hill saturation, seasonality, trend and promotions. I held out the last 13 weeks to test it on data it never saw.
Check it against the truth. The script compares the estimates with the real values, reports ROAS and marginal ROAS for each channel, and tests a budget shift.
Results

The sampler converged well: 4 chains, 1,000 draws each, max R-hat 1.005, minimum effective sample size 1,486, and no divergences.

How well it predicts (13 weeks the model never saw)

	Error (MAPE)
This model, hold-out	3.1%
Naive benchmark (average of the last 13 training weeks)	16.6%

On the training weeks, R-squared was 0.97.

Fit and hold-out

Did it find the true values?

Channel	True decay	Est. decay	True ROAS	Est. ROAS	True marginal ROAS	Est. marginal ROAS
amazon_ads	0.50	0.53	1.28	1.21	1.18	1.34
google_ads	0.30	0.34	1.50	2.72	1.57	1.77
meta_ads	0.70	0.81	1.16	0.85	0.94	0.97
The adstock decay and marginal ROAS for all three channels landed inside the 90% credible intervals (full intervals are in results/channel_results.csv).
Average ROAS landed inside the interval for Amazon and Meta, but not for Google. The model overestimated it: 2.72 against a true value of 1.50 (interval 1.63 to 4.69). I'm leaving that in on purpose, because it's the most useful result in the project.

ROAS recovery Response curves

A budget test. Moving about $5.7k a week from Meta to Google, the channels with the lowest and highest marginal ROAS, is estimated to add about $4.2k in weekly revenue (90% interval $1.1k to $7.3k). The model puts a 98.5% probability on the change being positive. That holds only as long as the model's assumptions do.

What I took from it
Average ROAS is the hardest number to get right in an MMM. It depends on how much revenue the model gives to media versus the baseline, and with only 104 weeks of data that split is uncertain. The Google result shows it.
Marginal ROAS is the number budget decisions actually need, and it was recovered much more reliably.
A good hold-out score doesn't prove the channel attribution is right. It took checking against the known truth, and looking at how wide the intervals were, to see that.
Limits
The model has the same structure as the process that generated the data, so real data will be harder. In real work I'd also check for misspecification and correlated channels, and I'd look for outside evidence such as lift tests.
It uses geometric adstock with an 8-week window, an additive baseline, and two years of weekly data.
ROAS here means revenue per $1 of ad spend, not profit.
How to run it
bash
pip install -r requirements.txt
python src/generate_data.py   # simulate the data and the ground truth
python src/run_sql.py         # clean with SQL, run checks, export data/model_input.csv
python src/mmm.py             # fit the model (about 3 to 4 minutes on one CPU core), write results/
Files
sql/01_clean_ad_spend.sql: fixes channel labels and removes duplicates
sql/02_data_quality_checks.sql: row counts, duplicate keys, missing spend, week coverage
sql/03_model_input.sql: builds the weekly table joined to revenue and promotions
sql/04_channel_summary.sql: a simple spend summary for context
results/: figures, metrics.json, channel_results.csv, parameter_summary.csv

Built by Praise Oton. github.com/PraiseGabriel