"""Feature engineering: 41 features computed from the 71 intraday returns."""
import re

import numpy as np
import pandas as pd

META = {"ID", "date", "eqt_code", "target", "end_of_day_return"}


def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with missing returns imputed and the engineered features appended."""
    df_feat = df.copy()
    ret_cols = [c for c in df.columns if c not in META]
    returns = df_feat[ret_cols].apply(pd.to_numeric, errors="coerce")
    row_means = returns.mean(axis=1)
    for col in returns.columns:  # a missing return takes the mean of its row
        returns[col] = returns[col].fillna(row_means)
    returns = returns.fillna(0)
    df_feat[ret_cols] = returns
    arr = returns.values.astype(np.float32)
    n, mid = len(ret_cols), len(ret_cols) // 2

    # A) Time-series statistics
    df_feat["return_mean"] = arr.mean(axis=1)
    df_feat["return_std"] = arr.std(axis=1)
    df_feat["return_max"] = arr.max(axis=1)
    df_feat["return_min"] = arr.min(axis=1)
    df_feat["mean_last_5"] = arr[:, -5:].mean(axis=1)
    df_feat["mean_last_15"] = arr[:, -15:].mean(axis=1)
    df_feat["std_last_15"] = arr[:, -15:].std(axis=1)

    for span in [5, 10, 20, 40]:  # exponentially weighted means, recent returns weigh more
        alpha = 2 / (span + 1)
        w = alpha * (1 - alpha) ** np.arange(n - 1, -1, -1)
        w /= w.sum()
        df_feat[f"ewma_{span}"] = (arr @ w).astype(np.float32)

    df_feat["ewma_ratio_5_20"] = df_feat["ewma_5"] / (df_feat["ewma_20"].abs() + 1e-8)
    df_feat["ewma_ratio_5_40"] = df_feat["ewma_5"] / (df_feat["ewma_40"].abs() + 1e-8)
    df_feat["ewma_ratio_10_40"] = df_feat["ewma_10"] / (df_feat["ewma_40"].abs() + 1e-8)

    mu = arr.mean(axis=1, keepdims=True)
    sig = arr.std(axis=1, keepdims=True) + 1e-8
    norm = (arr - mu) / sig
    df_feat["skewness"] = (norm**3).mean(axis=1)
    df_feat["excess_kurtosis"] = (norm**4).mean(axis=1) - 3

    x0, x1 = arr[:, :-1], arr[:, 1:]
    cov = ((x0 - x0.mean(axis=1, keepdims=True)) * (x1 - x1.mean(axis=1, keepdims=True))).mean(axis=1)
    df_feat["autocorr_lag1"] = cov / (x0.std(axis=1) * x1.std(axis=1) + 1e-8)

    # B) Volatility
    df_feat["realized_variance"] = (arr**2).sum(axis=1)
    df_feat["vol_morning"] = arr[:, :mid].std(axis=1)
    df_feat["vol_afternoon"] = arr[:, mid:].std(axis=1)
    df_feat["vol_ratio"] = df_feat["vol_morning"] / (df_feat["vol_afternoon"] + 1e-8)

    # C) Shape of the cumulative price path
    prices = np.cumsum(arr, axis=1)
    df_feat["price_end"] = prices[:, -1]
    df_feat["price_max"] = prices.max(axis=1)
    df_feat["price_min"] = prices.min(axis=1)
    df_feat["drawdown"] = df_feat["price_end"] - df_feat["price_max"]
    df_feat["zero_crossings"] = ((arr[:, 1:] * arr[:, :-1]) < 0).sum(axis=1)

    # D) Cross-section: each stock compared with the other stocks of the same day
    df_feat["market_return_daily"] = df_feat.groupby("date")["price_end"].transform("mean")
    df_feat["market_std_daily"] = df_feat.groupby("date")["price_end"].transform("std")
    df_feat["z_score_daily"] = (df_feat["price_end"] - df_feat["market_return_daily"]) / (
        df_feat["market_std_daily"] + 1e-8
    )
    for feat in ["mean_last_15", "return_std", "ewma_5"]:
        df_feat[f"z_{feat}"] = df_feat.groupby("date")[feat].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    for feat in ["price_end", "mean_last_15", "ewma_5"]:
        df_feat[f"rank_{feat}"] = df_feat.groupby("date")[feat].rank(pct=True)

    last_col = ret_cols[-1]
    grp = df_feat.groupby("date")[last_col]
    df_feat["z_score_last_bin"] = (df_feat[last_col] - grp.transform("mean")) / (grp.transform("std") + 1e-8)

    # E) Market regime of the day
    df_feat["market_dispersion"] = df_feat.groupby("date")["return_std"].transform("mean")
    df_feat["market_up_ratio"] = df_feat.groupby("date")["price_end"].transform(lambda x: (x > 0).mean())
    df_feat["alpha_vs_market"] = df_feat["price_end"] - df_feat["market_return_daily"]
    df_feat["rank_alpha"] = df_feat.groupby("date")["alpha_vs_market"].rank(pct=True)

    # Intraday beta of each stock against the average stock of the day
    _, inv = np.unique(df_feat["date"].values, return_inverse=True)
    beta_vals = np.zeros(len(df_feat), dtype=np.float32)
    for i in range(inv.max() + 1):
        mask = inv == i
        grp_ = arr[mask]
        mkt = grp_.mean(axis=0)
        mkt_d = mkt - mkt.mean()
        var_m = float((mkt_d**2).mean()) + 1e-8
        stk_d = grp_ - grp_.mean(axis=1, keepdims=True)
        beta_vals[mask] = (stk_d * mkt_d).mean(axis=1) / var_m
    df_feat["intraday_beta"] = beta_vals

    df_feat["eqt_code"] = df_feat["eqt_code"].astype("category").cat.codes
    df_feat.columns = [re.sub(r"[\[\]{} ,:]+", "_", str(c)) for c in df_feat.columns]
    return df_feat
