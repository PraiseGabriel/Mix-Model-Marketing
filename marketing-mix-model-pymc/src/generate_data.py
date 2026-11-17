"""
Im Generating this Data because Im not allow to share the orginal data of the company I did this for

Outputs (in data/):
  raw_ad_spend.csv        weekly spend by channel, deliberately messy (duplicates, inconsistent labels)
  raw_sales.csv           weekly revenue and promo flag
  ground_truth.json       true parameters and true ROAS per channel
  true_contributions.csv  true weekly revenue contribution of each channel (for validation only)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_WEEKS = 104
L_MAX = 8  # adstock carry-over window in weeks
DATA = Path(__file__).resolve().parents[1] / "data"

CHANNELS = {
    # mean weekly spend, decay, half-saturation K (share of max spend), slope S, max effect beta ($ per week)
    "amazon_ads": dict(mean=25_000, decay=0.50, K=0.50, S=1.5, beta=90_000),
    "google_ads": dict(mean=18_000, decay=0.30, K=0.40, S=2.0, beta=60_000),
    "meta_ads":   dict(mean=12_000, decay=0.70, K=0.60, S=1.2, beta=45_000),
}


def geometric_adstock(x, decay, l_max=L_MAX):
    """Weighted sum of the current and previous l_max-1 weeks. Weights decay^k, normalised to sum to 1."""
    w = decay ** np.arange(l_max)
    w = w / w.sum()
    out = np.zeros_like(x, dtype=float)
    for k in range(l_max):
        out[k:] += w[k] * x[: len(x) - k]
    return out


def hill(x, K, S):
    return x**S / (x**S + K**S)


def main():
    rng = np.random.default_rng(SEED)
    weeks = pd.date_range("2023-01-02", periods=N_WEEKS, freq="W-MON")
    t = np.arange(N_WEEKS)
    yearly = 2 * np.pi * t / 52.0

    # spend: seasonal, with on/off "flights" so channels move independently enough to be identified
    spend = {}
    for name, p in CHANNELS.items():
        season = 1 + 0.25 * np.sin(yearly + rng.uniform(0, 2 * np.pi))
        flight = np.ones(N_WEEKS)
        for _ in range(6):
            start = rng.integers(0, N_WEEKS - 6)
            flight[start : start + rng.integers(3, 7)] *= rng.choice([0.35, 1.7])
        noise = rng.lognormal(0, 0.15, N_WEEKS)
        spend[name] = np.clip(p["mean"] * season * flight * noise, 0, None).round(2)

    # non-media baseline: level, trend, annual seasonality, promotions
    promo = np.zeros(N_WEEKS, dtype=int)
    for yr in (0, 1):
        for w in (19 + 52 * yr, 33 + 52 * yr, 46 + 52 * yr, 48 + 52 * yr):
            if w < N_WEEKS:
                promo[w] = 1
    baseline = (
        120_000
        + 150 * t
        + 15_000 * np.sin(yearly - 1.0) + 6_000 * np.cos(2 * yearly)
        + 12_000 * promo
    )

    # media effect = beta * Hill(adstock(spend) / max raw spend)
    contrib = {}
    for name, p in CHANNELS.items():
        x = geometric_adstock(spend[name], p["decay"]) / spend[name].max()
        contrib[name] = p["beta"] * hill(np.clip(x, 1e-9, None), p["K"], p["S"])

    revenue = baseline + sum(contrib.values()) + rng.normal(0, 6_000, N_WEEKS)

    # raw tables, messy on purpose: the SQL layer cleans them
    labels = {
        "amazon_ads": ["amazon_ads", "Amazon Ads", "amazon_ads "],
        "google_ads": ["google_ads", "Google Ads", " GOOGLE_ADS"],
        "meta_ads":   ["meta_ads", "Meta Ads", "meta_ads"],
    }
    rows = []
    for name in CHANNELS:
        for i, w in enumerate(weeks):
            rows.append((w.date(), rng.choice(labels[name], p=[0.7, 0.2, 0.1]), spend[name][i]))
    ad_spend = pd.DataFrame(rows, columns=["week_start", "channel", "spend"])
    dups = ad_spend.sample(9, random_state=SEED)  # exact duplicate rows, like a re-run feed
    ad_spend = pd.concat([ad_spend, dups]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    ad_spend.to_csv(DATA / "raw_ad_spend.csv", index=False)

    pd.DataFrame({"week_start": weeks.date, "revenue": revenue.round(2), "promo_flag": promo}).to_csv(
        DATA / "raw_sales.csv", index=False
    )
    pd.DataFrame({"week_start": weeks.date, **{k: v.round(2) for k, v in contrib.items()}}).to_csv(
        DATA / "true_contributions.csv", index=False
    )

    total_media = sum(c.sum() for c in contrib.values())
    truth = {
        name: {
            **{k: v for k, v in p.items() if k != "mean"},
            "total_spend": float(spend[name].sum()),
            "true_roas": float(contrib[name].sum() / spend[name].sum()),
            "true_share_of_media_revenue": float(contrib[name].sum() / total_media),
        }
        for name, p in CHANNELS.items()
    }
    (DATA / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print("raw spend rows:", len(ad_spend), "| weeks:", N_WEEKS)
    for k, v in truth.items():
        print(f"{k:11s} true ROAS = {v['true_roas']:.2f}")


if __name__ == "__main__":
    main()
