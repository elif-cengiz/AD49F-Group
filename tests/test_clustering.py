"""Clustering tests (Member B)."""
import numpy as np
import pandas as pd

from helpers import data, train_test
from src.clustering import choose_k, corr_distance, hier_labels
from src.config import TRAIN_END


def test_distance_matrix_is_valid():
    train, _ = train_test()
    d = corr_distance(train)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0)
    assert (d >= 0).all() and (d <= 1).all()


def test_k_selection_uses_only_training_data():
    rets, _, _ = data()
    corrupted = rets.copy()
    corrupted.loc[corrupted.index > pd.Timestamp(TRAIN_END)] = 0.0
    assert choose_k(rets.loc[:TRAIN_END]) == choose_k(corrupted.loc[:TRAIN_END])


def test_labels_cover_all_stocks_with_k_clusters():
    train, _ = train_test()
    k = choose_k(train)
    labels, _, _ = hier_labels(train, k)
    assert len(labels) == train.shape[1] and len(set(labels)) == k


def test_identical_stocks_cluster_together():
    """Two stocks with the same return series must always land in the same cluster."""
    train, _ = train_test()
    t = train.copy()
    t["COPY"] = t.iloc[:, 0] + 1e-9
    labels, _, _ = hier_labels(t, 3)
    assert labels[0] == labels[-1]
