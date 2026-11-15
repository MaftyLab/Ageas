#!/usr/bin/env python3
"""The graph stage: walks through unlabeled cells carry the terminal labels to the progenitors.

When a dataset holds unlabeled cells between its progenitors and its terminal
states (a continuous trajectory, as in a time course), a random walk over a
neighbour graph of every cell, stopped at the first terminal cell it meets,
tells which terminal state each progenitor leads to: its absorption
probabilities, as CellRank and Palantir compute fate probabilities, without a
pseudotime direction. The graph follows Palantir's defaults: PCA (LSI for
TF-IDF input), multiscale diffusion components, an adaptive Gaussian kernel
over 30 neighbours.

The stage then ranks the progenitors by a weighted average of the walk's
ranks and the committee vote's, and calls the second class for the share of
cells the walk gives it on average. It runs only with two terminal classes
and at least ``min_context_per_query`` unlabeled cells per progenitor;
otherwise the vote passes through unchanged. Progenitor features are used;
progenitor labels never are.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, splu
from scipy.stats import rankdata
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors

#: The two embeddings: PCA of z-scored features (log-normalised RNA) or LSI (TF-IDF ATAC).
KINDS = ('pca', 'lsi')


def embed(features, kind: str = 'pca', n_components: int = 50, seed: int = 0) -> np.ndarray:
    """Every cell in one low-dimensional space.

    :param features: ``(n_cells, n_features)``, dense or sparse.
    :param kind: ``'pca'``: z-scored features (clipped at 10 SD), then PCA.
        ``'lsi'``: truncated SVD of the features as given, first component
        dropped (it follows sequencing depth), then z-scored, as Signac's LSI.
    :param n_components: Components kept (fewer if the data has fewer).
    """
    assert kind in KINDS, f'kind must be one of {KINDS}'
    n_cells, n_features = features.shape
    if kind == 'pca':
        x = features.toarray() if sp.issparse(features) else np.asarray(features)
        x = np.asarray(x, dtype=np.float64)
        sd = x.std(axis=0)
        x = np.clip((x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0), -10, 10)
        return PCA(min(n_components, n_features, n_cells - 1), random_state=seed).fit_transform(x)
    n = min(n_components, n_features - 2, n_cells - 2)
    lsi = TruncatedSVD(n + 1, random_state=seed).fit_transform(sp.csr_matrix(features, dtype=np.float64))[:, 1:]
    sd = lsi.std(axis=0)
    return (lsi - lsi.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def affinity(z: np.ndarray, neighbors: int) -> sp.csr_matrix:
    """Symmetric kNN graph, adaptive Gaussian kernel.

    Each cell's bandwidth is the distance to its ``neighbors // 3``-th
    neighbour, floored at 1% of the median neighbour distance so that
    duplicate cells keep their links to the rest.
    """
    k = min(neighbors, len(z) - 1)
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z)
    dist, idx = dist[:, 1:], idx[:, 1:]
    positive = dist[dist > 0]
    floor = 0.01 * float(np.median(positive)) if positive.size else 1.0
    sigma = np.maximum(dist[:, max(k // 3, 1) - 1], floor)
    rows = np.repeat(np.arange(len(z)), k)
    weights = np.exp(-(dist.ravel() ** 2) / (sigma[rows] * sigma[idx.ravel()]))
    a = sp.csr_matrix((weights, (rows, idx.ravel())), shape=(len(z), len(z)))
    a = a.maximum(a.T).tocsr()
    a.eliminate_zeros()
    return a


def diffusion_space(z: np.ndarray, neighbors: int = 30, n_components: int = 10) -> np.ndarray:
    """Palantir's multiscale space: diffusion components scaled by ``lambda / (1 - lambda)``, the first dropped."""
    a = affinity(z, neighbors)
    d = np.maximum(np.asarray(a.sum(axis=1)).ravel(), 1e-12)
    inv_sqrt = sp.diags(1.0 / np.sqrt(d))
    k = min(n_components + 1, len(z) - 2)
    values, vectors = eigsh(inv_sqrt @ a @ inv_sqrt, k=k, which='LA')
    order = np.argsort(-values)
    values, vectors = values[order][1:], (inv_sqrt @ vectors[:, order])[:, 1:]
    values = np.clip(values, 0.0, 1.0 - 1e-6)
    return vectors * (values / (1.0 - values))


def absorption(a: sp.csr_matrix, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Chance that a walk from each cell first reaches a terminal cell of each class.

    :param a: Symmetric affinity over every cell.
    :param labels: Class of each terminal cell; ``-1`` for every other cell.
    :returns: ``(n_cells, n_classes)``; terminal cells keep their own class,
        and a cell whose walks never reach a terminal cell (its part of the
        graph holds none) gets every class alike.
    """
    terminal = labels >= 0
    _, part = connected_components(a, directed=False)
    reachable = np.isin(part, np.unique(part[terminal]))
    free = ~terminal & reachable
    p = (sp.diags(1.0 / np.maximum(np.asarray(a.sum(axis=1)).ravel(), 1e-300)) @ a).tocsr()
    onehot = np.eye(n_classes)[labels[terminal]]
    out = np.full((len(labels), n_classes), 1.0 / n_classes)
    out[terminal] = onehot
    if free.any():
        system = (sp.identity(int(free.sum())) - p[free][:, free]).tocsc()
        hits = np.asarray(splu(system).solve(np.asarray(p[free][:, terminal] @ onehot)))
        reach = hits.sum(axis=1, keepdims=True)
        out[free] = np.where(reach > 1e-12, hits / np.maximum(reach, 1e-300), 1.0 / n_classes)
    return out


def graph_fate(features, labels, kind: str = 'pca', neighbors: int = 30, diffusion_components: int = 10,
               n_components: int = 50, seed: int = 0) -> np.ndarray:
    """Absorption probabilities of every cell over the diffusion-space neighbour graph.

    :param features: Every cell's features (terminal, query and unlabeled).
    :param labels: Class index of each terminal cell; ``-1`` for every other cell.
    """
    labels = np.asarray(labels, dtype=int)
    z = diffusion_space(embed(features, kind, n_components, seed), neighbors, diffusion_components)
    return absorption(affinity(z, neighbors), labels, int(labels.max()) + 1)


def fuse(vote: np.ndarray, walk: np.ndarray, weight: float = 0.6) -> np.ndarray:
    """Two-class probabilities from the vote and the walk.

    Cells are ranked by ``(1 - weight) * rank(vote) + weight * rank(walk)``
    (ranks of the second class's probability, scaled to (0, 1]), and the
    walk's mean probability of the second class sets how many are called it.
    The returned probability is a cell's place in that ranking, rescaled so
    that exactly those cells lie above 0.5 (0.01-0.99): a ranking with an
    estimated share, not a per-cell calibration.
    """
    n = len(vote)
    score = (1 - weight) * rankdata(vote[:, 1]) / n + weight * rankdata(walk[:, 1]) / n
    called = int(round(float(walk[:, 1].mean()) * n))
    ordered = np.sort(score)[::-1]
    low, high = ordered[-1], ordered[0]
    if called == 0:
        cut = high + 1e-9
    elif called == n:
        cut = low - 1e-9
    else:
        cut = (ordered[called - 1] + ordered[called]) / 2
    p = np.where(score > cut,
                 0.5 + 0.49 * (score - cut) / max(high - cut, 1e-12),
                 0.01 + 0.49 * (score - low) / max(cut - low, 1e-12))
    p = np.where(score > cut, np.maximum(p, 0.5 + 1e-9), np.minimum(p, 0.5 - 1e-9))
    return np.column_stack([1 - p, p])


def graph_stage(vote: np.ndarray, features, labels, query_index, settings: dict, kind: str = 'pca',
                seed: int = 0) -> dict:
    """Fuse the walk into the vote when the data allows it; otherwise pass the vote through.

    :param vote: The committee vote for the query cells, ``(n_query, n_classes)``.
    :param features: Every cell's features: terminal, query and unlabeled.
    :param labels: Class index of each terminal cell; ``-1`` for query and unlabeled cells.
    :param query_index: Position of each query cell (in ``vote`` order) among ``features``.
    :param settings: ``min_context_per_query``, ``neighbors``, ``diffusion_components``,
        ``n_components`` and ``weight`` (protocol ``graph``).
    :param kind: Embedding, ``'pca'`` or ``'lsi'`` (see :func:`embed`).
    :returns: ``probs`` (the stage's output), ``active``, ``reason``,
        ``n_context``, ``n_query``, ``walk`` (the query cells' absorption
        probabilities, or ``None``) and ``share`` (the walk's mean probability
        of the second class, or ``None``).
    """
    labels = np.asarray(labels, dtype=int)
    query_index = np.asarray(query_index, dtype=int)
    is_query = np.zeros(len(labels), dtype=bool)
    is_query[query_index] = True
    n_context = int(((labels < 0) & ~is_query).sum())
    out = {'probs': vote, 'active': False, 'n_context': n_context, 'n_query': int(len(query_index)),
           'walk': None, 'share': None, 'weight': settings['weight']}
    if vote.shape[1] != 2 or len(np.unique(labels[labels >= 0])) != 2:
        return dict(out, reason='the stage needs exactly two terminal classes')
    if n_context < settings['min_context_per_query'] * len(query_index):
        return dict(out, reason=f"{n_context} unlabeled cells for {len(query_index)} query cells: under "
                                f"{settings['min_context_per_query']} per query cell")
    walk = graph_fate(features, labels, kind, settings['neighbors'], settings['diffusion_components'],
                      settings['n_components'], seed)[query_index]
    return dict(out, probs=fuse(vote, walk, settings['weight']), active=True, walk=walk,
                share=float(walk[:, 1].mean()), reason=f'{n_context} unlabeled cells for {len(query_index)} query cells')
