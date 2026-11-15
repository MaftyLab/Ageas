"""The graph stage: walks through the unlabeled cells carry the terminal labels to the progenitors.

When a dataset holds unlabeled cells between its progenitors and its terminal
states, a random walk over a neighbour graph of all cells, stopped at the
first terminal cell, tells which terminal state each progenitor leads to. The
stage ranks the progenitors by a weighted average of the walk's ranks and the
vote's, and calls the second class for the share of cells the walk gives it
on average. Without such cells the vote passes through unchanged.
"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ageas.tool.graph import fuse, graph_fate, graph_stage

SETTINGS = {'min_context_per_query': 1.0, 'neighbors': 15, 'diffusion_components': 5,
            'n_components': 10, 'weight': 0.6}


def _fork(seed=0, n_features=20):
    """A trunk that forks into branches 0 and 1.

    Progress ``p`` runs 0-1 along the trunk and 1-2 along a branch, where the
    branches drift apart. Every feature mixes progress and drift, plus noise.
    Terminal cells sit at the branch ends; the query cells sit just past the
    fork (70% on branch 0); unlabeled context cells fill the trunk and the
    branches in between. Returns features, labels (class for terminal cells,
    -1 otherwise), the query positions and the query cells' true branch.
    """
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(2, n_features))
    parts = []

    def cells(n, low, high, branch, label):
        p = rng.uniform(low, high, n)
        drift = np.where(branch == 1, 1.0, -1.0) * np.maximum(p - 1.0, 0.0)
        x = np.column_stack([p, drift]) @ loadings + rng.normal(0.0, 0.05, size=(n, n_features))
        parts.append((x, np.full(n, label), np.full(n, branch)))

    cells(300, 0.3, 1.0, 0, -1)                     # trunk context
    for branch in (0, 1):
        cells(200, 1.2, 1.8, branch, -1)            # branch context
        cells(150, 1.8, 2.0, branch, branch)        # terminal cells
    first_query = sum(len(x) for x, _, _ in parts)
    cells(140, 1.0, 1.15, 0, -1)                    # query, branch 0
    cells(60, 1.0, 1.15, 1, -1)                     # query, branch 1
    x = np.vstack([p[0] for p in parts])
    labels = np.concatenate([p[1] for p in parts]).astype(int)
    branch = np.concatenate([p[2] for p in parts])
    query = np.arange(first_query, len(x))
    return x, labels, query, branch[query]


def test_walks_through_the_unlabeled_cells_send_progenitors_to_their_own_branch():
    x, labels, query, truth = _fork()

    walk = graph_fate(x, labels, neighbors=15, diffusion_components=5, n_components=10, seed=0)

    assert walk.shape == (len(x), 2)
    np.testing.assert_allclose(walk.sum(axis=1), 1.0, atol=1e-8)
    assert ((walk >= 0) & (walk <= 1)).all()
    np.testing.assert_allclose(walk[labels == 1, 1], 1.0)  # terminal cells keep their own class
    assert roc_auc_score(truth, walk[query, 1]) > 0.85


def test_lsi_embedding_for_sparse_count_like_input():
    x, labels, query, truth = _fork(seed=1)
    counts = np.clip(np.round(np.exp(x / 2)), 0, None)  # non-negative, count-like

    walk = graph_fate(counts, labels, kind='lsi', neighbors=15, diffusion_components=5, n_components=10, seed=0)

    assert walk.shape == (len(x), 2)
    np.testing.assert_allclose(walk.sum(axis=1), 1.0, atol=1e-8)


def test_fusion_ranks_by_the_weighted_average_and_calls_the_walks_share():
    vote = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.4, 0.3, 0.9])
    walk = np.array([0.7, 0.1, 0.2, 0.3, 0.9, 0.5, 0.6, 0.4])
    as_probs = lambda p: np.column_stack([1 - p, p])  # noqa: E731

    by_walk = fuse(as_probs(vote), as_probs(walk), weight=1.0)[:, 1]
    by_vote = fuse(as_probs(vote), as_probs(walk), weight=0.0)[:, 1]
    blended = fuse(as_probs(vote), as_probs(walk), weight=0.6)

    assert list(np.argsort(by_walk)) == list(np.argsort(walk))
    assert list(np.argsort(by_vote)) == list(np.argsort(vote))
    np.testing.assert_allclose(blended.sum(axis=1), 1.0)
    share = round(walk.mean() * len(walk))  # 0.4625 x 8 -> 4 cells called class 1
    for probs in (by_walk, by_vote, blended[:, 1]):
        assert (probs > 0.5).sum() == share
        assert ((probs >= 0) & (probs <= 1)).all()


def test_without_unlabeled_cells_between_progenitors_and_terminal_states_the_vote_passes_through():
    x, labels, query, _ = _fork()
    context = np.setdiff1d(np.flatnonzero(labels < 0), query)
    keep = np.setdiff1d(np.arange(len(x)), context)          # terminal and query cells only
    query_kept = np.searchsorted(keep, query)
    vote = np.column_stack([np.full(len(query), 0.6), np.full(len(query), 0.4)])

    stage = graph_stage(vote, x[keep], labels[keep], query_kept, SETTINGS, seed=0)

    assert stage['active'] is False
    assert stage['n_context'] == 0
    np.testing.assert_array_equal(stage['probs'], vote)


def test_with_enough_unlabeled_cells_the_stage_fuses_the_walk_into_the_vote():
    x, labels, query, truth = _fork()
    vote = np.column_stack([np.full(len(query), 0.5), np.full(len(query), 0.5)])
    vote[:, 1] += np.random.default_rng(0).normal(0, 0.01, len(query))  # an uninformative vote
    vote[:, 0] = 1 - vote[:, 1]

    stage = graph_stage(vote, x, labels, query, SETTINGS, seed=0)

    assert stage['active'] is True
    assert stage['n_context'] == 700
    assert stage['share'] == pytest.approx(stage['walk'][:, 1].mean())
    assert roc_auc_score(truth, stage['probs'][:, 1]) > roc_auc_score(truth, vote[:, 1]) + 0.2
    np.testing.assert_allclose(stage['probs'].sum(axis=1), 1.0)


def test_the_stage_needs_as_many_unlabeled_cells_as_progenitors():
    x, labels, query, _ = _fork()
    vote = np.column_stack([np.full(len(query), 0.5), np.full(len(query), 0.5)])

    stage = graph_stage(vote, x, labels, query, dict(SETTINGS, min_context_per_query=4.0), seed=0)

    assert stage['active'] is False  # 700 context cells for 200 progenitors: under 4 per progenitor
    np.testing.assert_array_equal(stage['probs'], vote)


def test_more_than_two_terminal_classes_leave_the_vote_alone():
    x, labels, query, _ = _fork()
    labels = labels.copy()
    labels[np.flatnonzero(labels == 1)[:50]] = 2
    vote = np.full((len(query), 3), 1 / 3)

    stage = graph_stage(vote, x, labels, query, SETTINGS, seed=0)

    assert stage['active'] is False
    np.testing.assert_array_equal(stage['probs'], vote)
