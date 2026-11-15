"""Shift-aware validation weights.

Terminal cells are weighted by how much they resemble the progenitor cells,
judged from features alone, so that validation leans on the terminal cells
most like the cells the ensemble will be asked about.
"""
import numpy as np
import pytest

from ageas import shift_weights


def _blob(rng, n, centre, n_features=20):
    x = rng.normal(0.0, 1.0, size=(n, n_features))
    x[:, 0] += centre
    return x


def test_terminal_cells_resembling_the_progenitors_weigh_more():
    # Terminal cells sit in two groups, at 0 and at 4 along the first feature;
    # the progenitors sit by the second group.
    rng = np.random.default_rng(0)
    near = _blob(rng, 200, 4.0)
    far = _blob(rng, 200, 0.0)
    progenitors = _blob(rng, 100, 4.0)

    weights = shift_weights(np.vstack([far, near]), progenitors, seed=0)

    assert weights.shape == (400,)
    assert weights[200:].mean() > 5 * weights[:200].mean()


def test_weights_average_one_so_ordinary_and_shift_aware_losses_share_a_scale():
    rng = np.random.default_rng(1)
    terminal = np.vstack([_blob(rng, 150, 0.0), _blob(rng, 150, 3.0)])
    progenitors = _blob(rng, 80, 2.0)

    weights = shift_weights(terminal, progenitors, seed=0)

    assert np.all(weights >= 0)
    assert weights.mean() == pytest.approx(1.0)


def test_weights_stay_close_to_one_when_progenitors_look_like_terminal_cells():
    rng = np.random.default_rng(2)
    weights = shift_weights(_blob(rng, 300, 0.0), _blob(rng, 150, 0.0), seed=0)

    assert np.mean((weights > 0.5) & (weights < 2.0)) > 0.9


def test_no_terminal_cell_carries_the_validation_alone():
    # Progenitors far beyond every terminal cell: the odds explode for the
    # few nearest terminal cells, so the weights are capped.
    rng = np.random.default_rng(3)
    terminal = _blob(rng, 300, 0.0)
    progenitors = _blob(rng, 100, 8.0)

    weights = shift_weights(terminal, progenitors, max_ratio=10.0, seed=0)

    assert weights.max() <= 10.0 + 1e-9
    assert weights.mean() == pytest.approx(1.0)
