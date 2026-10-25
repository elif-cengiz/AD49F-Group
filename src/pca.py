"""
PCA: how many independent risks are there? (Member A)

Everything is fitted on the training data only; the test data is never used here.
"""
import numpy as np
import pandas as pd

from src.config import TICKERS


def fit_pca(train: pd.DataFrame, method: str = "pearson"):
    """Eigendecomposition of the correlation matrix.

    method: "pearson" (default), "spearman" (rank correlation, robust to outliers)
            or "ledoit_wolf" (correlation from a shrunk covariance matrix).
    Returns: eigenvalues (largest first), eigenvectors (DataFrame), training mean, training std.
    """
    mu, sd = train.mean(), train.std()
    z = (train - mu) / sd
    if method == "pearson":
        corr = np.corrcoef(z.values, rowvar=False)
    elif method == "spearman":
        corr = train.rank().corr().values
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(z.values).covariance_
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
    else:
        raise ValueError(method)
    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # Sign ambiguity: the sign of an eigenvector is arbitrary; fix it so that the weights sum to a positive number.
    vecs = vecs * np.sign(vecs.sum(axis=0) + 1e-12)
    vecs = pd.DataFrame(vecs, index=train.columns, columns=[f"PC{i + 1}" for i in range(len(vals))])
    return vals, vecs, mu, sd


def loadings_as_correlation(vals, vecs: pd.DataFrame) -> pd.DataFrame:
    """Eigenvector x sqrt(eigenvalue) = correlation of each stock with the component (an easier-to-read loading)."""
    return vecs * np.sqrt(vals)


def marchenko_pastur_upper(N: int, T: int) -> float:
    """Upper bound for the largest eigenvalue of a correlation matrix built from pure independent noise."""
    return (1 + np.sqrt(N / T)) ** 2


def effective_bets(vals) -> float:
    """Effective number of independent bets from eigenvalue entropy: between 1 (one factor) and N (all independent)."""
    p = np.asarray(vals) / np.sum(vals)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def scores(data: pd.DataFrame, vecs: pd.DataFrame, mu, sd, k: int) -> pd.DataFrame:
    z = (data - mu) / sd
    return pd.DataFrame(z.values @ vecs.values[:, :k], index=data.index, columns=vecs.columns[:k])


def act1(rets, bench, fx):
    import matplotlib.pyplot as plt
    from src.data import split
    from src.plotting import BLUE, LIGHT, RED, savefig

    print("\n=== ACT 1 — How many bets? PCA ===")
    train, _ = split(rets)
    T, N = train.shape
    vals, vecs, mu, sd = fit_pca(train)
    share = vals / vals.sum()
    lam_plus = marchenko_pastur_upper(N, T)
    n_signal = int((vals > lam_plus).sum())
    n_70 = int(np.searchsorted(np.cumsum(share), 0.70) + 1)
    print(f"  PC1 variance share: {100 * share[0]:.1f}% | first 3 PCs: {100 * share[:3].sum():.1f}%")
    print(f"  70% rule: {n_70} components  |  Marchenko–Pastur (λ+ = {lam_plus:.2f}): {n_signal} components")
    print(f"  Effective number of independent bets: {effective_bets(vals):.1f} / {N}")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(1, N + 1), vals, color=[BLUE if v > lam_plus else LIGHT for v in vals])
    ax.axhline(lam_plus, ls="--", color=RED, label=f"Noise threshold λ+ = {lam_plus:.2f}")
    ax.axvline(n_70 + 0.5, ls=":", color="grey", label=f"70% rule: {n_70} components")
    ax.set_xticks(range(1, N + 1)); ax.set_xlabel("Component"); ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Only {n_signal} of {N} components stand out from noise")
    ax.legend(frameon=False)
    savefig("1a_scree_mp.png")

    k = max(3, n_signal)
    L = loadings_as_correlation(vals, vecs).iloc[:, :k]
    fig, ax = plt.subplots(figsize=(5.8, 7.2))
    im = ax.imshow(L.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(N))
    ax.set_yticklabels([f"{t.replace('.IS', '')} ({TICKERS.get(t, '?')})" for t in L.index])
    ax.set_xticks(range(k)); ax.set_xticklabels(L.columns)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Stock–component correlation")
    ax.set_title("Factor loadings (training period)")
    savefig("1b_loadings_heatmap.png")

    sc = scores(train, vecs, mu, sd, k)
    obs = pd.DataFrame({"XU100": bench.reindex(train.index), "USD/TRY": fx.reindex(train.index)})
    print("  Correlation of PC scores with XU100 and USD/TRY returns (training):")
    print(pd.concat([sc, obs], axis=1).corr().loc[sc.columns, obs.columns].round(2).to_string())

    print("  Robustness — first 3 eigenvalues and effective bets:")
    for m in ["pearson", "spearman", "ledoit_wolf"]:
        v = fit_pca(train, m)[0]
        print(f"   {m:<12} {np.round(v[:3], 2)}  effective bets {effective_bets(v):.1f}")

    # Leakage check: how much would the loadings change if the same PCA saw all data (test included)?
    _, vecs_full, _, _ = fit_pca(rets)
    for pc in ["PC1", "PC2"]:
        diff = (vecs[pc] - vecs_full[pc]).abs().max()
        print(f"  {pc}: largest difference between training and full-sample loadings = {diff:.2f}")
    return vals, vecs
