"""Data tests (Member C)."""
import numpy as np
import pandas as pd

from helpers import data, train_test
from src.config import TRAIN_END
from src.data import outside_price_limit


def test_dates_sorted_and_unique():
    rets, _, _ = data()
    assert rets.index.is_monotonic_increasing and rets.index.is_unique


def test_no_missing_after_cleaning():
    rets, _, _ = data()
    assert not rets.isna().any().any()


def test_no_holiday_rows():
    """No day where every stock returned exactly 0 (a holiday carried forward by Yahoo) may remain."""
    rets, _, _ = data()
    assert not (rets == 0).all(axis=1).any()


def test_adjusted_returns_within_price_limit():
    """BIST daily price limit is ±10%: no adjusted return may exceed it."""
    rets, _, _ = data()
    assert not outside_price_limit(rets).any().any()


def test_limit_down_day_is_not_flagged():
    """A genuine -10% day is log(0.9) = -0.105 in log returns; it must NOT be flagged.
    A bonus issue (price halves, log(0.5) = -0.69) must be flagged."""
    assert not outside_price_limit(np.log(0.90)) and not outside_price_limit(np.log(1.10))
    assert outside_price_limit(np.log(0.50))


def test_benchmark_and_fx_have_no_nan():
    """A misalignment silently creates a NaN on the first day and breaks correlations."""
    _, bench, fx = data()
    assert not bench.isna().any() and not fx.isna().any()


def test_train_test_do_not_overlap():
    train, test = train_test()
    assert train.index.max() < test.index.min()
    assert train.index.max() <= pd.Timestamp(TRAIN_END) < test.index.min()
