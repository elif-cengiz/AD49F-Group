"""
Walk-forward backtest and regime monitoring (Member C).

Rule: on every rebalancing day only data from BEFORE that day is used.
"""
import numpy as np
import pandas as pd

from src.config import AR_WINDOW, COST_BPS, LOOKBACK, TRAIN_END
from src.clustering import choose_k, hier_labels

EW = "Equal weight"
CB = "Cluster-balanced"
STRATS = [EW, CB]


def cluster_weights_from_labels(window: pd.DataFrame, labels) -> pd.Series:
    """Equal weight across clusters, inverse volatility within each cluster."""
    labels = pd.Series(np.asarray(labels), index=window.columns)
    vol = window.std()
    w = pd.Series(0.0, index=window.columns)
    clusters = sorted(labels.unique())
    for c in clusters:
        m = labels[labels == c].index
        iv = 1 / vol[m]
        w[m] = iv / iv.sum() / len(clusters)
    return w


def cluster_risk_weights(window: pd.DataFrame, k: int) -> pd.Series:
    return cluster_weights_from_labels(window, hier_labels(window, k)[0])


def rebalance_dates(rets: pd.DataFrame) -> pd.DatetimeIndex:
    test_idx = rets.index[rets.index > pd.Timestamp(TRAIN_END)]
    firsts = pd.Series(test_idx, index=test_idx).groupby(test_idx.to_period("M")).first()
    return pd.DatetimeIndex(firsts.values)


def run_backtest(rets: pd.DataFrame, k: int, fixed_labels=None):
    """Monthly rebalancing. Returns (daily returns, start-of-period weights, period returns).

    If fixed_labels is given (leakage demo only), the clusters are kept fixed.
    """
    simple = np.exp(rets) - 1
    dates = rebalance_dates(rets)
    N = rets.shape[1]
    w_ew = pd.Series(1 / N, index=rets.columns)
    daily = {s: [] for s in STRATS}
    weights = {s: {} for s in STRATS}
    period_rets = {}
    prev = {s: None for s in STRATS}
    for i, d in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else None
        period = simple.loc[(simple.index >= d) & ((simple.index < end) if end is not None else True)]
        hist = rets.loc[rets.index < d].tail(LOOKBACK)          # day d EXCLUDED: past data only
        if fixed_labels is not None:
            w_cl = cluster_weights_from_labels(hist, fixed_labels)
        else:
            w_cl = cluster_risk_weights(hist, k)
        growth = (1 + period).cumprod()
        period_rets[d] = growth.iloc[-1] - 1
        for name, w in zip(STRATS, [w_ew, w_cl]):
            cost = 0.0 if prev[name] is None else (w - prev[name]).abs().sum() * COST_BPS / 1e4
            val = (growth * w).sum(axis=1)                       # weights drift with prices during the month
            pr = val.pct_change()
            pr.iloc[0] = val.iloc[0] - 1
            pr.iloc[0] -= cost
            daily[name].append(pr)
            weights[name][d] = w
            end_w = growth.iloc[-1] * w
            prev[name] = end_w / end_w.sum()
    daily = pd.DataFrame({s: pd.concat(v) for s, v in daily.items()})
    weights = {s: pd.DataFrame(v).T for s, v in weights.items()}
    return daily, weights, pd.DataFrame(period_rets).T


def metrics(r: pd.Series) -> pd.Series:
    ann = 252
    eq = (1 + r).cumprod()
    return pd.Series({
        "Ann. return": eq.iloc[-1] ** (ann / len(r)) - 1,
        "Ann. vol": r.std() * np.sqrt(ann),
        "Return/vol": (r.mean() * ann) / (r.std() * np.sqrt(ann)),
        "Max drawdown": (eq / eq.cummax() - 1).min(),
    })


def attribution(weights: dict, period_rets: pd.DataFrame) -> pd.DataFrame:
    """Approximate contribution: for each period, start-of-period weight x the stock's period return, summed.
    Shows which stocks explain the gap between the cluster-balanced and equal-weight portfolios."""
    contrib = {s: (weights[s] * period_rets).sum() for s in STRATS}
    out = pd.DataFrame(contrib)
    out["Difference"] = out[CB] - out[EW]
    out["Avg. weight (cluster)"] = weights[CB].mean()
    return out.sort_values("Difference", ascending=False)


def absorption_ratio(rets: pd.DataFrame, window: int = AR_WINDOW):
    """Kritzman et al. (2011): share of the first N/5 covariance eigenvalues (rolling window)."""
    k = max(1, rets.shape[1] // 5)
    ar = {}
    for i in range(window, len(rets) + 1):
        v = np.linalg.eigvalsh(np.cov(rets.iloc[i - window:i].values, rowvar=False))[::-1]
        ar[rets.index[i - 1]] = v[:k].sum() / v.sum()
    return pd.Series(ar), k


def _fmt(table: pd.DataFrame) -> str:
    out = table.copy().astype(object)
    for c in table:
        out[c] = table[c].map(lambda x: f"{x:.2f}" if c == "Return/vol" else f"{x:.1%}")
    return out.to_string()


def act3(rets, bench, fx, k=None):
    import matplotlib.pyplot as plt
    from src.data import split
    from src.plotting import BLUE, GREY, RED, savefig

    print("\n=== ACT 3 — Does the decision change? Walk-forward test ===")
    train, _ = split(rets)
    k = k or choose_k(train)
    bt, weights, period_rets = run_backtest(rets, k)
    usd = bt.sub(np.exp(fx.reindex(bt.index).fillna(0)) - 1, axis=0)
    table = pd.DataFrame({f"{s} (TRY)": metrics(bt[s]) for s in STRATS}
                         | {f"{s} (USD, approx.)": metrics(usd[s]) for s in STRATS}).T
    print(f"  Test: {bt.index[0].date()} – {bt.index[-1].date()}, k = {k}, cost {COST_BPS} bp one-way")
    print(_fmt(table))

    print("\n  Sensitivity — cluster-balanced return/vol for different k "
          f"(equal weight: {metrics(bt[EW])['Return/vol']:.2f}):")
    sens = {kk: metrics(run_backtest(rets, kk)[0][CB])["Return/vol"] for kk in range(2, 9)}
    print("   " + "  ".join(f"k={kk}: {v:.2f}" for kk, v in sens.items()))

    att = attribution(weights, period_rets)
    print("\n  Stocks that explain most of the gap (approx. contribution, cluster − equal weight):")
    print(att.head(5).map(lambda x: f"{x:+.1%}").to_string())

    full_labels = hier_labels(rets, k)[0]                        # test data included: LEAKAGE DEMO ONLY
    leak = run_backtest(rets, k, fixed_labels=full_labels)[0]
    print("\n  LEAKAGE DEMO — if the clusters are found on all data, test period included:")
    print(f"   Cluster-balanced return/vol: honest = {metrics(bt[CB])['Return/vol']:.2f}"
          f" | leaky = {metrics(leak[CB])['Return/vol']:.2f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot((1 + bt[EW]).cumprod(), color=GREY, label=EW)
    ax.plot((1 + bt[CB]).cumprod(), color=BLUE, label=CB)
    ax.set_title("Cumulative return in the test period (TRY, after costs)")
    ax.legend(frameon=False)
    savefig("3a_backtest.png")

    ar, kk = absorption_ratio(rets)
    xu = np.exp(bench.fillna(0).cumsum()).reindex(ar.index)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(ar, color=RED); ax1.set_ylabel(f"Absorption ratio (first {kk} PCs)", color=RED)
    ax2 = ax1.twinx(); ax2.plot(xu, color=GREY, lw=1); ax2.set_ylabel("XU100 (1 = start)")
    ax1.axvline(pd.Timestamp(TRAIN_END), ls="--", color="grey")
    ax1.set_title(f"When does the market turn into a single bet? ({AR_WINDOW}-day window)")
    savefig("3b_absorption_ratio.png")
    return bt, table, att
