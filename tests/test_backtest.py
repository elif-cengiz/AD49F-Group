"""Backtest tests (Member C)."""
import numpy as np
import pandas as pd

from helpers import data, train_test
from src.backtest import EW, absorption_ratio, attribution, cluster_risk_weights, run_backtest
from src.clustering import choose_k
from src.config import LOOKBACK, TRAIN_END


def _k():
    return choose_k(train_test()[0])


def test_weights_are_valid():
    train, _ = train_test()
    w = cluster_risk_weights(train.tail(LOOKBACK), _k())
    assert abs(w.sum() - 1) < 1e-10 and (w >= 0).all()


def test_backtest_has_no_lookahead():
    """Changing returns AFTER a date must not change results BEFORE that date."""
    rets, _, _ = data()
    base = run_backtest(rets, _k())[0]
    cut = base.index[len(base) // 2]
    future = rets.copy()
    mask = future.index >= cut
    future.loc[mask] = np.random.default_rng(1).normal(0, 0.03, future.loc[mask].shape)
    alt = run_backtest(future, _k())[0]
    before = base.index < cut
    assert np.allclose(base.loc[before].values, alt.loc[before].values)


def test_backtest_only_covers_test_period():
    rets, _, _ = data()
    bt = run_backtest(rets, _k())[0]
    assert bt.index.min() > pd.Timestamp(TRAIN_END)
    assert not bt.isna().any().any()
    assert bt.index.is_unique


def test_equal_weight_matches_manual_calculation():
    """The first month's equal-weight return (no cost yet) must match a manual calculation."""
    rets, _, _ = data()
    bt, weights, period_rets = run_backtest(rets, _k())
    d0 = period_rets.index[0]
    manual = period_rets.loc[d0].mean()                        # equal weight, drifting within the month
    model = (1 + bt[EW].loc[(bt.index >= d0) & (bt.index < period_rets.index[1])]).prod() - 1
    assert abs(model - manual) < 1e-10                          # no cost at the first rebalance


def test_attribution_adds_up():
    """Stock contributions must add up to the total of the period returns."""
    rets, _, _ = data()
    _, weights, period_rets = run_backtest(rets, _k())
    att = attribution(weights, period_rets)
    total = (weights[EW] * period_rets).sum(axis=1).sum()
    assert abs(att[EW].sum() - total) < 1e-10


def test_absorption_ratio_bounds():
    rets, _, _ = data()
    ar, _ = absorption_ratio(rets)
    assert ((ar > 0) & (ar < 1)).all()
