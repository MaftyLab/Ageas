#!/usr/bin/env python3
"""Shift-aware validation weights.

Progenitor cells can sit far from the terminal cells a model is trained and
validated on. Weighting each terminal cell by how much it resembles the
progenitors makes validation lean on the cells most like the ones the
ensemble will be asked about (importance weighting under covariate shift).
Only the progenitors' features are used, never their labels.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def _flat(x) -> np.ndarray:
    """``(n_cells, ..., n_features)`` data as an ``(n_cells, n)`` float array."""
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(len(x), -1)


def shared_space(terminal_x, query_x, n_components: int = 50, seed: int = 0) -> tuple:
    """Terminal and progenitor cells in one principal-component space.

    :returns: ``(terminal_z, query_z)``.
    """
    x = np.vstack([_flat(terminal_x), _flat(query_x)])
    n_components = max(1, min(n_components, x.shape[1], len(x) - 1))
    z = PCA(n_components=n_components, svd_solver='randomized', random_state=seed).fit_transform(x)
    n_terminal = len(terminal_x)
    return z[:n_terminal], z[n_terminal:]


def _cap(weights: np.ndarray, max_ratio: float) -> np.ndarray:
    """Weights rescaled to mean 1 with none above ``max_ratio``."""
    weights = weights / weights.mean()
    for _ in range(100):
        if weights.max() <= max_ratio:
            break
        weights = np.minimum(weights, max_ratio)
        weights = weights / weights.mean()
    return np.minimum(weights, max_ratio)


def shift_weights(
    terminal_x,
    query_x,
    n_components: int = 50,
    n_neighbors: int = 50,
    max_ratio: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """Weight of every terminal cell for shift-aware validation.

    In the cells' shared principal-component space (:func:`shared_space`),
    the share of progenitors among a terminal cell's nearest neighbours
    (itself left out) gives its odds of being a progenitor, which estimate
    how much denser the progenitors are than the terminal cells around it.
    A neighbourhood is a local judgement, so cells beyond the progenitors
    are not over-weighted the way a linear classifier's extrapolation would
    over-weight them. The share is smoothed by one pseudo-neighbour drawn
    at the overall progenitor share, so no cell's weight is exactly zero.
    The odds are scaled to mean 1, so a weighted loss stays on the scale of
    the ordinary one, and capped at ``max_ratio`` so that a few cells cannot
    carry the whole validation.

    :param terminal_x: Terminal cells' model input, ``(n_cells, ..., n_features)``.
    :param query_x: Progenitor cells' model input, same feature order.
    :param n_components: Principal components of the shared space.
    :param n_neighbors: Neighbours that judge each terminal cell.
    :param max_ratio: Largest weight, as a multiple of the mean weight.
    :param seed: Seed for the components.
    :returns: One non-negative weight per terminal cell, mean 1.
    """
    terminal_z, query_z = shared_space(terminal_x, query_x, n_components, seed)
    z = np.vstack([terminal_z, query_z])
    is_query = np.r_[np.zeros(len(terminal_z), dtype=bool), np.ones(len(query_z), dtype=bool)]
    k = min(n_neighbors, len(z) - 1)
    _, index = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(terminal_z)
    neighbours = index[:, 1:]  # the first neighbour is the cell itself
    prior = is_query.mean()
    share = (is_query[neighbours].sum(axis=1) + prior) / (k + 1)
    return _cap(share / (1.0 - share), max_ratio)
