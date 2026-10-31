"""
Data download, cache and cleaning (Member C).

Flow: yfinance -> data/*.csv (cache) -> BIST calendar -> log returns -> checks.
After the first run the data is always read from the cache; no internet is needed in class.
"""
import os

import numpy as np
import pandas as pd

from src.config import (BENCH, DATA_DIR, END_DATE, FX, MAX_MISSING, PRICE_LIMIT,
                        START_DATE, TICKERS, TRAIN_END)

PRICES_CSV = os.path.join(DATA_DIR, "bist_prices.csv")
RAW_CSV = os.path.join(DATA_DIR, "bist_prices_unadjusted.csv")
SYN_RAW_CSV = os.path.join(DATA_DIR, "synthetic_unadjusted.csv")


def extract_close(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Return a (dates x tickers) table of closes, whatever column layout yfinance returned.

    Depending on the version and `group_by`, columns are (Price, Ticker), (Ticker, Price) or flat for a
    single ticker; handling all three avoids KeyErrors.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            return raw["Close"]
        if "Close" in raw.columns.get_level_values(1):
            return raw.xs("Close", axis=1, level=1)
        raise KeyError(f"No 'Close' column found: {list(raw.columns)[:5]}...")
    if "Close" not in raw.columns:
        raise KeyError(f"No 'Close' column found: {list(raw.columns)}")
    return raw[["Close"]].rename(columns={"Close": tickers[0]})


def outside_price_limit(logret, tol: float = 0.01):
    """True where a daily log return is outside the BIST price limit (plus a small rounding tolerance).

    The limit is ±10% in SIMPLE returns, which is asymmetric in log returns:
    +10% -> log(1.10) = +0.095, but -10% -> log(0.90) = -0.105.
    A symmetric check such as |r| > log(1.11) wrongly flags genuine limit-down days.
    """
    lo, hi = np.log(1 - PRICE_LIMIT - tol), np.log(1 + PRICE_LIMIT + tol)
    return (logret < lo) | (logret > hi)


def _download_close(syms: list, auto_adjust: bool) -> pd.DataFrame:
    """Download closes; retry tickers that came back empty one by one.

    With threads, yfinance can fail on single tickers (e.g. "database is locked" from its timezone cache).
    """
    import yfinance as yf

    raw = yf.download(syms, start=START_DATE, end=END_DATE, auto_adjust=auto_adjust,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data. Check your internet connection.")
    close = extract_close(raw, syms).reindex(columns=syms)
    for s in [s for s in syms if close[s].isna().all()]:
        one = yf.download(s, start=START_DATE, end=END_DATE, auto_adjust=auto_adjust,
                          progress=False, threads=False)
        if one is not None and not one.empty:
            close[s] = extract_close(one, [s]).iloc[:, 0].reindex(close.index)
            print(f"  Retried {s}: OK")
    return close


def download_prices() -> pd.DataFrame:
    """Download adjusted prices (and unadjusted ones for the check); read the cache if it exists."""
    if os.path.exists(PRICES_CSV):
        return pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    prices = _download_close(list(TICKERS) + [BENCH, FX], auto_adjust=True)
    prices.to_csv(PRICES_CSV)
    _download_close(list(TICKERS), auto_adjust=False).to_csv(RAW_CSV)
    return prices


def synthetic_prices(seed: int = 6) -> pd.DataFrame:
    """BIST-like synthetic data for offline rehearsal: market + FX + sector factors, two stress periods,
    the ±10% price limit, a public holiday and one unadjusted bonus-issue day."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START_DATE, END_DATE, inclusive="left")
    T = len(dates)
    mkt_vol = np.full(T, 0.014)
    for a, b in [("2024-07-25", "2024-08-20"), ("2025-03-19", "2025-04-15")]:
        mkt_vol[(dates >= a) & (dates <= b)] = 0.028
    mkt = rng.standard_t(4, T) / np.sqrt(2) * mkt_vol + 0.0012
    fx = rng.normal(0.0009, 0.004, T)
    sectors = sorted(set(TICKERS.values()))
    sec_f = {s: rng.normal(0, 0.009, T) for s in sectors}
    fx_beta = {"Banking": -1.2, "Holding": -0.4, "Retail": -0.6, "Telecom": -0.5,
               "Automotive": 1.0, "Steel": 1.1, "Transport": 1.3, "Defense": 0.8,
               "Construction": 0.9, "Energy": 0.7, "Chemicals": 0.9, "Glass": 0.8}
    rets = {}
    for t, s in TICKERS.items():
        beta = rng.uniform(0.8, 1.3) * (1.25 if s == "Banking" else 1.0)
        idio = rng.standard_t(5, T) / np.sqrt(5 / 3) * rng.uniform(0.010, 0.018)
        rets[t] = beta * mkt + fx_beta[s] * fx * 1.5 + sec_f[s] + idio
    r = pd.DataFrame(rets, index=dates).clip(-0.095, 0.095)
    r[BENCH] = r.mean(axis=1) * 0.9
    r[FX] = fx
    holiday = (dates >= "2026-05-27") & (dates <= "2026-05-29")   # holiday days that Yahoo carries forward
    r.loc[holiday, list(TICKERS) + [BENCH]] = 0.0
    prices = 100 * np.exp(r.cumsum())
    raw = prices[list(TICKERS)].copy()
    raw.loc[raw.index >= "2025-02-10", "FROTO.IS"] /= 2.0          # unadjusted bonus issue
    os.makedirs(DATA_DIR, exist_ok=True)
    raw.to_csv(SYN_RAW_CSV)
    return prices


def load_data(synthetic: bool = False):
    """Return (stock log returns, XU100 log return, USD/TRY log return)."""
    prices = (synthetic_prices() if synthetic else download_prices()).sort_index()
    prices = prices[~prices.index.duplicated()]

    stocks = prices[[t for t in TICKERS if t in prices.columns]].dropna(how="all")
    missing = stocks.isna().mean()
    keep = missing[missing < MAX_MISSING].index.tolist()
    dropped = sorted(set(TICKERS) - set(keep))
    if dropped:
        print(f"  Warning: more than {MAX_MISSING:.0%} missing data, dropped stocks: {dropped}")
    stocks = stocks[keep].dropna()

    rets = np.log(stocks).diff().dropna()
    # Public holidays: Yahoo carries the price forward, so EVERY stock has exactly 0 return. Drop these rows.
    all_zero = (rets == 0).all(axis=1)
    if all_zero.any():
        print(f"  Dropped {int(all_zero.sum())} days where every stock returned 0 "
              f"(holiday): {[d.date().isoformat() for d in rets.index[all_zero]]}")
        rets = rets.loc[~all_zero]

    # Align to the BIST calendar FIRST, then take differences; otherwise the first day silently becomes NaN.
    bench = np.log(prices[BENCH].reindex(stocks.index).ffill()).diff().reindex(rets.index)
    fx = np.log(prices[FX].reindex(stocks.index).ffill()).diff().reindex(rets.index)
    return rets, bench, fx


def split(rets: pd.DataFrame):
    """Train/test split. The test period starts on the first trading day after TRAIN_END."""
    train = rets.loc[:TRAIN_END]
    test = rets.loc[rets.index > pd.Timestamp(TRAIN_END)]
    return train, test


def act0(rets: pd.DataFrame, synthetic: bool = False):
    """Act 0: data summary and the bonus-issue trap."""
    import matplotlib.pyplot as plt
    from src.plotting import BLUE, RED, savefig

    print("\n=== ACT 0 — Data and the honesty contract ===")
    train, test = split(rets)
    print(f"  {rets.shape[1]} stocks, {len(rets)} trading days ({rets.index[0].date()} – {rets.index[-1].date()})")
    print(f"  Train: {len(train)} days  |  Test: {len(test)} days  |  Split: {TRAIN_END}")

    raw_path = SYN_RAW_CSV if synthetic else RAW_CSV
    if not os.path.exists(raw_path):
        print("  (No unadjusted price file; bonus-issue check skipped.)")
    else:
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        st = np.log(raw).diff().stack().dropna()
        bad = st[outside_price_limit(st)]
        print(f"  Days beyond the ±{PRICE_LIMIT:.0%} price limit in UNADJUSTED prices: {len(bad)} (fake returns)")
        if len(bad):
            print(bad.sort_values().head(5).round(3).to_string())
            d, t = bad.abs().idxmax()
            win = raw.loc[d - pd.Timedelta(days=60): d + pd.Timedelta(days=60), t]
            adj = np.exp(rets[t].cumsum()).loc[win.index.min(): win.index.max()]
            fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
            ax[0].plot(win, color=RED); ax[0].axvline(d, ls="--", color="grey")
            ax[0].set_title(f"{t.replace('.IS', '')}: unadjusted price")
            ax[1].plot(adj, color=BLUE); ax[1].set_title(f"{t.replace('.IS', '')}: adjusted (auto_adjust)")
            for a in ax:
                a.tick_params(axis="x", rotation=30)
            savefig("0_bonus_issue_trap.png")
    over = int(outside_price_limit(rets).sum().sum())
    print(f"  Days beyond the limit in ADJUSTED returns: {over} (should be 0)")
