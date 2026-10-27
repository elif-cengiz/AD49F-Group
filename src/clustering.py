"""
Clustering: which stocks are playing the same bet? (Member B)
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.config import K_OVERRIDE, K_RANGE, TICKERS


def corr_distance(r: pd.DataFrame) -> np.ndarray:
    """Mantegna correlation distance: d = sqrt(0.5 * (1 - rho)), from 0 (identical) to 1 (opposite)."""
    c = r.corr().values
    d = np.sqrt(np.clip(0.5 * (1 - c), 0, None))
    np.fill_diagonal(d, 0)
    return d


def hier_labels(r: pd.DataFrame, k: int, method: str = "average"):
    d = corr_distance(r)
    Z = linkage(squareform(d, checks=False), method=method)
    return fcluster(Z, k, criterion="maxclust"), Z, d


def silhouette_scores(train: pd.DataFrame, method: str = "average") -> dict:
    d = corr_distance(train)
    Z = linkage(squareform(d, checks=False), method=method)
    out = {}
    for k in K_RANGE:
        lab = fcluster(Z, k, criterion="maxclust")
        if len(set(lab)) > 1:
            out[k] = float(silhouette_score(d, lab, metric="precomputed"))
    return out


def choose_k(train: pd.DataFrame) -> int:
    """k is chosen on training data only. If K_OVERRIDE was justified in advance, it is used instead."""
    if K_OVERRIDE is not None:
        return int(K_OVERRIDE)
    sc = silhouette_scores(train)
    return max(sc, key=sc.get)


def stability(train: pd.DataFrame, test: pd.DataFrame, k: int) -> float:
    """How similar are the clusters when the same method is run on the train and test periods separately? (ARI)"""
    return float(adjusted_rand_score(hier_labels(train, k)[0], hier_labels(test, k)[0]))


def sector_agreement(train: pd.DataFrame, k: int) -> float:
    sectors = pd.factorize(pd.Series([TICKERS.get(t, t) for t in train.columns]))[0]
    return float(adjusted_rand_score(hier_labels(train, k)[0], sectors))


def act2(rets, vecs=None):
    import matplotlib.pyplot as plt
    from src.data import split
    from src.plotting import savefig

    print("\n=== ACT 2 — Which bets? Clustering ===")
    train, test = split(rets)
    sc = silhouette_scores(train)
    print("  Silhouette (training):", {k: round(v, 3) for k, v in sc.items()})
    best = sorted(sc.items(), key=lambda kv: -kv[1])[:2]
    if len(best) == 2 and best[0][1] - best[1][1] < 0.01:
        print(f"  Warning: k={best[0][0]} and k={best[1][0]} scored almost the same; justify the choice economically.")
    k = choose_k(train)
    labels, Z, _ = hier_labels(train, k)
    groups = pd.Series(labels, index=train.columns)
    print(f"  Chosen k = {k}{' (K_OVERRIDE)' if K_OVERRIDE else ''}")
    print(f"  Stability (train–test ARI): {stability(train, test, k):.2f}   (1 = identical, 0 = random)")
    print(f"  Agreement with sector labels (ARI): {sector_agreement(train, k):.2f}")
    for c in sorted(groups.unique()):
        m = groups[groups == c].index
        print(f"   Cluster {c}: " + ", ".join(f"{t.replace('.IS', '')} ({TICKERS.get(t, '?')})" for t in m))
    singles = [t.replace(".IS", "") for c in groups.unique() if (groups == c).sum() == 1 for t in groups[groups == c].index]
    if singles:
        print(f"  Warning: single-stock clusters {singles}. Each gets a large weight in the cluster-balanced portfolio.")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    dendrogram(Z, labels=[t.replace(".IS", "") for t in train.columns], ax=ax,
               color_threshold=Z[-(k - 1), 2] if k > 1 else 0, leaf_rotation=90)
    ax.set_ylabel("Correlation distance")
    ax.set_title(f"The stocks split into {k} risk groups (training period)")
    savefig("2a_dendrogram.png")

    if vecs is not None:
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for t in train.columns:
            x, y = vecs.loc[t, "PC1"], vecs.loc[t, "PC2"]
            ax.scatter(x, y, color=cmap((groups[t] - 1) % 10), s=60)
            ax.annotate(t.replace(".IS", ""), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("PC1 weight (market)"); ax.set_ylabel("PC2 weight")
        ax.set_title("Stock map: colour = cluster, position = PCA")
        savefig("2b_pca_map.png")
    return k, groups
