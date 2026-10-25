"""PCA tests (Member A)."""
import numpy as np
import pandas as pd

from helpers import data, train_test
from src.config import TRAIN_END
from src.pca import effective_bets, fit_pca, loadings_as_correlation, marchenko_pastur_upper


def test_eigenvalues_sum_to_n_and_vectors_orthonormal():
    train, _ = train_test()
    vals, vecs, _, _ = fit_pca(train)
    N = train.shape[1]
    assert abs(vals.sum() - N) < 1e-8                          # trace of a correlation matrix = N
    assert np.allclose(vecs.T.values @ vecs.values, np.eye(N), atol=1e-8)
    assert (np.diff(vals) <= 1e-12).all()                      # sorted largest first


def test_loadings_are_correlations():
    """Eigenvector x sqrt(eigenvalue) must equal the stock's correlation with the component score."""
    train, _ = train_test()
    vals, vecs, mu, sd = fit_pca(train)
    L = loadings_as_correlation(vals, vecs)
    pc1 = ((train - mu) / sd).values @ vecs["PC1"].values
    direct = np.array([np.corrcoef(train[c], pc1)[0, 1] for c in train.columns])
    assert np.allclose(L["PC1"].values, direct, atol=1e-6)


def test_pc1_is_market_factor():
    train, _ = train_test()
    _, bench, _ = data()
    vals, vecs, mu, sd = fit_pca(train)
    pc1 = ((train - mu) / sd).values @ vecs["PC1"].values
    assert np.corrcoef(pc1, bench.reindex(train.index).values)[0, 1] > 0.8
    assert (vecs["PC1"] > 0).all()                             # sign fixing works


def test_shuffled_data_has_no_factors():
    """Shuffle each column independently in time: correlation disappears, the tails stay the same.
    No component should exceed the noise threshold (at most 1, by chance)."""
    train, _ = train_test()
    rng = np.random.default_rng(0)
    shuffled = train.apply(lambda c: rng.permutation(c.values))
    vals, _, _, _ = fit_pca(shuffled)
    assert (vals > marchenko_pastur_upper(train.shape[1], len(train))).sum() <= 1


def test_fit_ignores_test_period():
    """Corrupting the test period must not change the PCA fitted on training data (no leakage)."""
    rets, _, _ = data()
    train, _ = train_test()
    corrupted = rets.copy()
    corrupted.loc[corrupted.index > pd.Timestamp(TRAIN_END)] *= -5
    v1, l1, _, _ = fit_pca(train)
    v2, l2, _, _ = fit_pca(corrupted.loc[:TRAIN_END])
    assert np.allclose(v1, v2) and np.allclose(l1.values, l2.values)


def test_effective_bets_bounds():
    N = 20
    assert abs(effective_bets(np.ones(N)) - N) < 1e-9          # fully independent: N bets
    assert effective_bets([N] + [1e-12] * (N - 1)) < 1.01      # a single factor: 1 bet


def test_robust_methods_agree_on_pc1():
    train, _ = train_test()
    base = fit_pca(train)[1]["PC1"]
    for m in ["spearman", "ledoit_wolf"]:
        assert np.corrcoef(base, fit_pca(train, m)[1]["PC1"])[0, 1] > 0.95
