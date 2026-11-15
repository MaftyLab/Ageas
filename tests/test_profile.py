"""Label-free profile of a dataset.

The profile describes the terminal cells (with their labels) and the
progenitor cells (features only): how far the progenitors sit from the
mature cells, how both cluster, and what the input looks like. The panel is
designed from it, so it must work where no progenitor fate is known.
"""
import numpy as np

from ageas import profile_dataset


def _cells(rng, n, centre, n_features=20):
    x = rng.normal(0.0, 1.0, size=(n, n_features))
    x[:, :len(centre)] += centre
    return x


def _two_classes(rng, n=150):
    """Mature cells of two classes, at -3 and +3 along the first feature."""
    x = np.vstack([_cells(rng, n, [-3.0]), _cells(rng, n, [3.0])])
    return x, np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]


def test_progenitors_far_from_the_mature_cells_score_as_far():
    rng = np.random.default_rng(0)
    terminal, labels = _two_classes(rng)
    among = np.vstack([_cells(rng, 50, [-3.0]), _cells(rng, 50, [3.0])])
    away = _cells(rng, 100, [0.0, 8.0])

    near = profile_dataset(terminal, labels, among, seed=0)
    far = profile_dataset(terminal, labels, away, seed=0)

    # Nearest-neighbour distances are dominated by the 18 noise features, so
    # even 8 units away only nearly doubles the ratio.
    assert far['nn_distance_ratio'] > 1.5 * near['nn_distance_ratio']
    assert far['coverage'] < 0.2 < 0.8 < near['coverage']
    assert far['domain_auc'] > 0.95 and near['domain_auc'] < 0.7


def test_progenitors_in_a_cluster_of_their_own_are_told_apart_from_mixed_ones():
    rng = np.random.default_rng(1)
    terminal, labels = _two_classes(rng)
    mixed = np.vstack([_cells(rng, 50, [-3.0]), _cells(rng, 50, [3.0])])
    apart = _cells(rng, 100, [0.0, 8.0])

    among = profile_dataset(terminal, labels, mixed, seed=0)
    own = profile_dataset(terminal, labels, apart, seed=0)

    assert own['query_own_clusters'] > 0.8
    assert among['query_own_clusters'] < 0.2
    # The two mature classes form clean clusters of their own either way.
    assert among['terminal_purity'] > 0.9 and own['terminal_purity'] > 0.9


def test_progenitors_between_the_mature_classes_look_ambiguous():
    # A model of the mature classes is unsure about cells halfway between
    # them, and sure about cells that sit inside one class.
    rng = np.random.default_rng(2)
    terminal, labels = _two_classes(rng)
    between = _cells(rng, 100, [0.0])
    inside = np.vstack([_cells(rng, 50, [-3.0]), _cells(rng, 50, [3.0])])

    torn = profile_dataset(terminal, labels, between, seed=0)
    settled = profile_dataset(terminal, labels, inside, seed=0)

    assert torn['query_ambiguity'] > 0.5 > 0.2 > settled['query_ambiguity']


def test_sparse_counts_are_told_apart_from_continuous_values():
    rng = np.random.default_rng(3)
    labels = np.r_[np.zeros(100, dtype=int), np.ones(200, dtype=int)]
    rates = np.where(labels[:, None] == 0, 0.1, 0.3) * np.ones((1, 40))
    counts = rng.poisson(rates)
    progenitor_counts = rng.poisson(0.2, size=(60, 40))
    continuous = rng.normal(0.0, 1.0, size=(300, 40))

    sparse = profile_dataset(counts, labels, progenitor_counts, seed=0)
    dense = profile_dataset(continuous, labels, rng.normal(0.0, 1.0, size=(60, 40)), seed=0)

    assert sparse['sparsity'] > 0.7 and sparse['nonnegative'] and sparse['integer_share'] == 1.0
    assert dense['sparsity'] < 0.01 and not dense['nonnegative'] and dense['integer_share'] < 0.01
    assert (sparse['n_terminal'], sparse['n_query'], sparse['n_features'], sparse['n_classes']) == (300, 60, 40, 2)
    assert sparse['min_class_share'] == 1 / 3


def test_a_boundary_no_line_can_draw_is_flagged_as_needing_a_flexible_model():
    # Exclusive-or: the class depends on whether the first two features
    # share a sign, which no linear model can express.
    rng = np.random.default_rng(4)
    xor = rng.normal(0.0, 1.0, size=(400, 10))
    xor[:, :2] = rng.choice([-2.0, 2.0], size=(400, 2)) + rng.normal(0.0, 0.3, size=(400, 2))
    xor_labels = (np.sign(xor[:, 0]) == np.sign(xor[:, 1])).astype(int)
    terminal, labels = _two_classes(rng)
    progenitors = _cells(rng, 50, [0.0])

    curved = profile_dataset(xor, xor_labels, rng.normal(0.0, 1.0, size=(50, 10)), seed=0)
    straight = profile_dataset(terminal, labels, progenitors, seed=0)

    assert curved['terminal_linear_accuracy'] < 0.7
    assert curved['terminal_nonlinear_gain'] > 0.2
    assert straight['terminal_linear_accuracy'] > 0.95
    assert straight['terminal_nonlinear_gain'] < 0.03
