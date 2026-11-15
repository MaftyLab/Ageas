#!/usr/bin/env python3
"""Label-free profile of a dataset.

The profile describes the terminal cells, with their labels, and the
progenitor cells, from their features alone: how far the progenitors sit
from the mature cells, how both cluster, and what the input looks like.
The panel is designed from it (:func:`~ageas.design_panel`), so it never
takes a progenitor label and works where no progenitor fate is known.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .shift import _flat, shared_space

#: Share of progenitors from which a cluster counts as theirs.
OWN_CLUSTER = 0.8


def _input(terminal_x, query_x, terminal_labels) -> dict:
    """Size of the data and what its values look like (terminal cells' input)."""
    x = _flat(terminal_x)
    labels = np.asarray(terminal_labels)
    nonzero = x[x != 0]
    return {
        'n_terminal': int(len(x)),
        'n_query': int(len(query_x)),
        'n_features': int(x.shape[1]),
        'n_classes': int(len(np.unique(labels))),
        'min_class_share': float(np.bincount(labels).min() / len(labels)),
        'sparsity': float(np.mean(x == 0)),
        'nonnegative': bool(x.min() >= 0),
        'integer_share': float(np.mean(nonzero == np.round(nonzero))) if nonzero.size else 1.0,
    }


def _distances(terminal_z, query_z, n_neighbors: int) -> dict:
    """How far the progenitors sit from the mature cells, in the shared space."""
    terminal_nn = NearestNeighbors(n_neighbors=2).fit(terminal_z)
    within = terminal_nn.kneighbors(terminal_z)[0][:, 1]  # nearest other terminal cell
    across = terminal_nn.kneighbors(query_z, n_neighbors=1)[0][:, 0]

    z = np.vstack([terminal_z, query_z])
    is_query = np.r_[np.zeros(len(terminal_z), dtype=bool), np.ones(len(query_z), dtype=bool)]
    k = min(n_neighbors, len(z) - 1)
    index = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z)[1][:, 1:]
    share = is_query[index].mean(axis=1)
    return {
        'nn_distance_ratio': float(np.median(across) / np.median(within)),
        'coverage': float(np.mean(across <= np.quantile(within, 0.95))),
        'domain_auc': float(roc_auc_score(is_query, share)),
    }


def _kmeans(z, max_clusters: int, seed: int) -> np.ndarray:
    """k-means clusters of ``z``, with the number of clusters (2 to
    ``max_clusters``) that has the best silhouette on up to 2,000 cells."""
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(z), size=min(len(z), 2000), replace=False)
    best = None
    for k in range(2, min(max_clusters, len(z) - 1) + 1):
        assign = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(z)
        score = silhouette_score(z[sample], assign[sample])
        if best is None or score > best[0]:
            best = (score, assign)
    return best[1]


def _clusters(terminal_z, query_z, terminal_labels, max_clusters: int, seed: int) -> dict:
    """How the mature cells cluster among themselves, and the progenitors among all cells."""
    labels = np.asarray(terminal_labels)
    terminal_assign = _kmeans(terminal_z, max_clusters, seed)
    purity = sum(
        np.bincount(labels[terminal_assign == c]).max() for c in np.unique(terminal_assign)
    ) / len(labels)

    assign = _kmeans(np.vstack([terminal_z, query_z]), max_clusters, seed)
    is_query = np.r_[np.zeros(len(terminal_z), dtype=bool), np.ones(len(query_z), dtype=bool)]
    query_share = {c: is_query[assign == c].mean() for c in np.unique(assign)}
    own = np.array([query_share[c] >= OWN_CLUSTER for c in assign[is_query]])
    return {
        'n_clusters': int(len(query_share)),
        'query_own_clusters': float(own.mean()),
        'terminal_purity': float(purity),
    }


def _linear() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def _linear_view(terminal_z, query_z, terminal_labels) -> dict:
    """What a linear model of the mature classes makes of the progenitors."""
    labels = np.asarray(terminal_labels)
    probs = np.clip(_linear().fit(terminal_z, labels).predict_proba(query_z), 1e-12, 1.0)
    entropy = -(probs * np.log(probs)).sum(axis=1) / np.log(probs.shape[1])
    return {'query_ambiguity': float(entropy.mean())}


def _boundary(terminal_z, terminal_labels, seed: int) -> dict:
    """How well a line separates the mature classes, and what a flexible model adds.

    Five-fold accuracy on the terminal cells of a logistic regression and of
    a 15-nearest-neighbour classifier, which can follow any boundary shape.
    """
    labels = np.asarray(terminal_labels)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    linear = cross_val_score(_linear(), terminal_z, labels, cv=folds).mean()
    flexible = cross_val_score(KNeighborsClassifier(n_neighbors=15), terminal_z, labels, cv=folds).mean()
    return {
        'terminal_linear_accuracy': float(linear),
        'terminal_nonlinear_gain': float(flexible - linear),
    }


def profile_dataset(terminal_x, terminal_labels, query_x, n_components: int = 50,
                    n_neighbors: int = 50, max_clusters: int = 8, seed: int = 0) -> dict:
    """Label-free profile of one dataset.

    :param terminal_x: Terminal cells' model input, ``(n_cells, ..., n_features)``.
    :param terminal_labels: Integer class of every terminal cell.
    :param query_x: Progenitor cells' model input, same feature order. Their
        labels are never needed.
    :param n_components: Principal components of the shared space.
    :param n_neighbors: Neighbours that judge whether a cell sits among
        progenitors or terminal cells.
    :param max_clusters: Most clusters tried when clustering all cells.
    :param seed: Seed for the components and the clustering.
    :returns: Mapping of measure name to value:

        * ``n_terminal``, ``n_query``, ``n_features``, ``n_classes`` and
          ``min_class_share`` (the rarest terminal class's share);
        * ``sparsity``: share of zero entries in the terminal cells' input;
        * ``nonnegative``: whether no input value is negative;
        * ``integer_share``: share of non-zero entries that are whole
          numbers (1 for raw counts);
        * ``nn_distance_ratio``: median distance from a progenitor to its
          nearest terminal cell, over the median distance between nearest
          terminal cells (about 1 when progenitors sit among mature cells);
        * ``coverage``: share of progenitors no farther from the nearest
          terminal cell than 95% of terminal cells are from theirs;
        * ``domain_auc``: how well the share of progenitors among a cell's
          neighbours tells progenitors from terminal cells (0.5: not at all);
        * ``n_clusters``: clusters found among all cells;
        * ``query_own_clusters``: share of progenitors in clusters, among
          all cells, made at least 80% of progenitors;
        * ``terminal_purity``: clustering the terminal cells alone, share of
          them in their cluster's majority class (1: the mature classes
          cluster apart);
        * ``query_ambiguity``: mean entropy, scaled to 0–1, of a linear
          model of the mature classes on the progenitors (1: it cannot tell
          which class any progenitor resembles);
        * ``terminal_linear_accuracy``: five-fold accuracy of that linear
          model on the terminal cells;
        * ``terminal_nonlinear_gain``: how much a nearest-neighbour
          classifier, which can follow any boundary, adds to that accuracy
          (near 0 when a line separates the classes as well as anything).
    """
    terminal_z, query_z = shared_space(terminal_x, query_x, n_components, seed)
    return {
        **_input(terminal_x, query_x, terminal_labels),
        **_distances(terminal_z, query_z, n_neighbors),
        **_clusters(terminal_z, query_z, terminal_labels, max_clusters, seed),
        **_linear_view(terminal_z, query_z, terminal_labels),
        **_boundary(terminal_z, terminal_labels, seed),
    }
