"""
For Context i'll state thw formula behind each logic Bayesian marketing mix model in PyMC.

Model, per week t and channel c:
    adstock_c(t) = sum_k w_c[k] * spend_c(t-k) / max_spend_c      geometric weights w_c[k] ~ alpha_c^k, normalised
    sat_c(t)     = adstock^s_c / (adstock^s_c + K_c^s_c)           Hill saturation (diminishing returns)
    revenue(t)   = intercept + trend + annual seasonality + promo + sum_c beta_c * sat_c(t) + noise


"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CHANNELS = ["amazon_ads", "google_ads", "meta_ads"]
L_MAX = 8
HOLDOUT = 13
SEED = 7


def lag_matrix(x, l_max):
    """(T, C) -> (T, L, C) where [t, k, c] = x[t-k, c] (zero before the start)."""
    T, C = x.shape
    out = np.zeros((T, l_max, C))
    for k in range(l_max):
        out[k:, k, :] = x[: T - k, :]
    return out


def fourier(t, period, order):
    cols = []
    for j in range(1, order + 1):
        cols += [np.sin(2 * np.pi * j * t / period), np.cos(2 * np.pi * j * t / period)]
    return np.column_stack(cols)


def hill_np(x, K, S):
    return x**S / (x**S + K**S)


def main():
    df = pd.read_csv(ROOT / "data" / "model_input.csv", parse_dates=["week_start"])
    truth = json.loads((ROOT / "data" / "ground_truth.json").read_text())
    T = len(df)
    n_train = T - HOLDOUT
    train_idx = np.arange(n_train)

    spend = df[CHANNELS].to_numpy()
    spend_max = spend.max(axis=0)
    lags = lag_matrix(spend / spend_max, L_MAX)
    y_scale = df["revenue"].iloc[:n_train].mean()
    y = df["revenue"].to_numpy() / y_scale
    t = np.arange(T)
    season = fourier(t, 52, 2)
    promo = df["promo_flag"].to_numpy().astype(float)
    t_scaled = t / T

    coords = {"week": np.arange(T), "channel": CHANNELS, "fourier": np.arange(season.shape[1])}
    with pm.Model(coords=coords) as model:
        alpha = pm.Beta("alpha", 2, 2, dims="channel")                 # adstock decay
        K = pm.Beta("K", 2, 2, dims="channel")                         # half-saturation point
        S = pm.Gamma("S", 3, 1.5, dims="channel")                      # saturation slope
        beta = pm.HalfNormal("beta", 1.0, dims="channel")              # max effect, in units of mean revenue

        k = np.arange(L_MAX)[:, None]
        w = alpha[None, :] ** k
        w = w / w.sum(axis=0, keepdims=True)
        adstock = pt.sum(lags * w[None, :, :], axis=1)                 # (T, C)
        x = pt.maximum(adstock, 1e-6)
        sat = x**S / (x**S + K**S)
        contrib = pm.Deterministic("channel_contrib", sat * beta, dims=("week", "channel"))

        intercept = pm.Normal("intercept", 1.0, 0.5)
        trend = pm.Normal("trend", 0, 0.3)
        season_coef = pm.Normal("season_coef", 0, 0.2, dims="fourier")
        promo_coef = pm.Normal("promo_coef", 0, 0.2)
        mu = pm.Deterministic(
            "mu",
            intercept + trend * t_scaled + pt.dot(season, season_coef) + promo_coef * promo + contrib.sum(axis=1),
            dims="week",
        )
        sigma = pm.HalfNormal("sigma", 0.1)
        pm.Normal("obs", mu[train_idx], sigma, observed=y[train_idx])

        idata = pm.sample(draws=1000, tune=1500, chains=4, cores=1, target_accept=0.97, random_seed=SEED,
                          progressbar=False)

    post = idata.posterior
    stack = lambda name: post[name].stack(sample=("chain", "draw")).values

    # ---- sampler diagnostics
    import arviz as az
    summ = az.summary(idata, var_names=["alpha", "K", "S", "beta", "sigma"])
    rhat_max = float(summ["r_hat"].max())
    ess_min = float(summ["ess_bulk"].min())
    divergences = int(idata.sample_stats["diverging"].sum())

    # ---- fit and out-of-sample accuracy
    mu_draws = stack("mu") * y_scale                                   # (T, S)
    mu_mean = mu_draws.mean(axis=1)
    actual = df["revenue"].to_numpy()
    r2 = 1 - ((actual[:n_train] - mu_mean[:n_train]) ** 2).sum() / ((actual[:n_train] - actual[:n_train].mean()) ** 2).sum()
    mape_train = np.mean(np.abs(actual[:n_train] - mu_mean[:n_train]) / actual[:n_train]) * 100
    mape_test = np.mean(np.abs(actual[n_train:] - mu_mean[n_train:]) / actual[n_train:]) * 100
    naive = actual[:n_train][-13:].mean()                              # naive benchmark: mean of last 13 training weeks
    mape_naive = np.mean(np.abs(actual[n_train:] - naive) / actual[n_train:]) * 100

    # ---- channel ROAS: model vs truth
    contrib_draws = stack("channel_contrib") * y_scale                 # (T, C, S)
    total_spend = spend.sum(axis=0)
    roas_draws = contrib_draws.sum(axis=0) / total_spend[:, None]      # (C, S)
    alpha_d, K_d, S_d, beta_d = stack("alpha"), stack("K"), stack("S"), stack("beta") * y_scale

    # marginal ROAS at average weekly spend (steady state: adstock equals spend), +10% spend
    avg = spend.mean(axis=0)
    x0 = (avg / spend_max)[:, None]
    f = lambda xx: beta_d * hill_np(xx, K_d, S_d)
    mroas_draws = (f(1.1 * x0) - f(x0)) / (0.1 * avg[:, None])
    t_cfg = {c: truth[c] for c in CHANNELS}
    t_beta = np.array([t_cfg[c]["beta"] for c in CHANNELS])[:, None]
    t_K = np.array([t_cfg[c]["K"] for c in CHANNELS])[:, None]
    t_S = np.array([t_cfg[c]["S"] for c in CHANNELS])[:, None]
    mroas_true = ((t_beta * hill_np(1.1 * x0, t_K, t_S)) - (t_beta * hill_np(x0, t_K, t_S))) / (0.1 * avg[:, None])

    rows = []
    for i, c in enumerate(CHANNELS):
        lo, hi = np.percentile(roas_draws[i], [5, 95])
        mlo, mhi = np.percentile(mroas_draws[i], [5, 95])
        alo, ahi = np.percentile(alpha_d[i], [5, 95])
        rows.append(dict(
            channel=c,
            true_roas=truth[c]["true_roas"], est_roas=roas_draws[i].mean(), roas_5pct=lo, roas_95pct=hi,
            true_marginal_roas=float(mroas_true[i, 0]), est_marginal_roas=mroas_draws[i].mean(),
            mroas_5pct=mlo, mroas_95pct=mhi,
            true_decay=truth[c]["decay"], est_decay=alpha_d[i].mean(), decay_5pct=alo, decay_95pct=ahi,
            roas_in_90pct_interval=bool(lo <= truth[c]["true_roas"] <= hi),
        ))
    ch = pd.DataFrame(rows)
    ch.to_csv(RES / "channel_results.csv", index=False)

    # ---- budget reallocation: move 10% of total weekly budget from lowest to highest marginal-ROAS channel
    order = np.argsort(mroas_draws.mean(axis=1))
    src, dst = order[0], order[-1]
    shift = 0.10 * avg.sum()
    shift = min(shift, 0.5 * avg[src])                                 # never move more than half of a channel's budget
    def resp(i, spend_w):                                              # weekly revenue from channel i at steady spend
        return beta_d[i] * hill_np(spend_w / spend_max[i], K_d[i], S_d[i])
    gain = (resp(dst, avg[dst] + shift) - resp(dst, avg[dst])) - (resp(src, avg[src]) - resp(src, avg[src] - shift))
    realloc = dict(
        from_channel=CHANNELS[src], to_channel=CHANNELS[dst], weekly_shift=float(shift),
        weekly_revenue_gain_median=float(np.median(gain)),
        gain_5pct=float(np.percentile(gain, 5)), gain_95pct=float(np.percentile(gain, 95)),
        prob_gain_positive=float((gain > 0).mean()),
    )

    metrics = dict(
        n_weeks=T, train_weeks=n_train, holdout_weeks=HOLDOUT,
        r2_train=float(r2), mape_train_pct=float(mape_train), mape_holdout_pct=float(mape_test),
        mape_holdout_naive_benchmark_pct=float(mape_naive),
        rhat_max=rhat_max, ess_bulk_min=ess_min, divergences=divergences,
        reallocation=realloc,
    )
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2))
    summ.to_csv(RES / "parameter_summary.csv")

    # ---- figures
    weeks = df["week_start"]
    lo, hi = np.percentile(mu_draws, [5, 95], axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(weeks, actual, color="black", lw=1.2, label="Actual revenue")
    ax.plot(weeks, mu_mean, color="gray", lw=1.5, label="Model (posterior mean)")
    ax.fill_between(weeks, lo, hi, color="gray", alpha=0.25, label="90% interval")
    ax.axvline(weeks.iloc[n_train], color="black", ls="--", lw=1)
    ax.text(weeks.iloc[n_train], ax.get_ylim()[1], " held-out", va="top", fontsize=9)
    ax.set_title(f"Fit and hold-out test (hold-out MAPE {mape_test:.1f}%)")
    ax.set_ylabel("Weekly revenue"); ax.legend(loc="upper left"); fig.tight_layout()
    fig.savefig(RES / "fit_and_holdout.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    pos = np.arange(len(CHANNELS)); wd = 0.35
    err = np.vstack([ch["est_roas"] - ch["roas_5pct"], ch["roas_95pct"] - ch["est_roas"]])
    ax.bar(pos - wd / 2, ch["true_roas"], wd, color="white", edgecolor="black", hatch="//", label="True (simulation)")
    ax.bar(pos + wd / 2, ch["est_roas"], wd, yerr=err, color="gray", edgecolor="black", capsize=4, label="Model (90% interval)")
    ax.set_xticks(pos); ax.set_xticklabels(CHANNELS); ax.set_ylabel("ROAS (revenue per $1 of spend)")
    ax.set_title("Channel ROAS: model vs. truth"); ax.legend(); fig.tight_layout()
    fig.savefig(RES / "roas_recovery.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    grid = np.linspace(0, 1, 100)
    for i, c in enumerate(CHANNELS):
        curves = beta_d[i][None, :] * hill_np(grid[:, None], K_d[i][None, :], S_d[i][None, :])
        clo, chi = np.percentile(curves, [5, 95], axis=1)
        axes[i].fill_between(grid * spend_max[i], clo, chi, color="gray", alpha=0.3, label="Model 90% interval")
        axes[i].plot(grid * spend_max[i], curves.mean(axis=1), color="gray", label="Model mean")
        axes[i].plot(grid * spend_max[i], t_cfg[c]["beta"] * hill_np(grid, t_cfg[c]["K"], t_cfg[c]["S"]),
                     color="black", ls="--", label="True")
        axes[i].axvline(avg[i], color="black", lw=0.8, ls=":")
        axes[i].set_title(c); axes[i].set_xlabel("Weekly spend")
    axes[0].set_ylabel("Weekly revenue from channel"); axes[0].legend(fontsize=8)
    fig.suptitle("Saturation (response) curves; dotted line = average spend"); fig.tight_layout()
    fig.savefig(RES / "response_curves.png", dpi=150); plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(ch.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
