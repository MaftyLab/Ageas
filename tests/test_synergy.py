"""No-harm integration: which units are worth combining.

Selection runs on out-of-fold probabilities. A unit joins only if it lowers
the ensemble's OOF log-loss by at least ``min_relative_gain`` of the current
loss, and every unit that stays must raise it by at least that much if
removed.
"""
import numpy as np
import pytest

from ageas import no_harm_selection


def _probs(p_true, labels):
    """Two-class probability rows giving ``p_true`` to each cell's own label."""
    p_true = np.asarray(p_true, dtype=float)
    rows = np.arange(len(labels))
    probs = np.empty((len(labels), 2))
    probs[rows, labels] = p_true
    probs[rows, 1 - labels] = 1 - p_true
    return probs


def test_complementary_units_are_both_kept():
    # A is sure about cells 0 and 2, B about cells 1 and 3, and each is a
    # coin flip elsewhere. Together every cell gets 0.7: log-loss 0.357
    # against 0.399 for either alone.
    labels = np.array([0, 0, 1, 1])
    oof = {
        "A": _probs([0.9, 0.5, 0.9, 0.5], labels),
        "B": _probs([0.5, 0.9, 0.5, 0.9], labels),
    }
    assert no_harm_selection(oof, labels, min_relative_gain=0.01) == pytest.approx({"A": 0.5, "B": 0.5})


def test_gain_is_measured_relative_to_the_current_loss():
    # Both units are confident, so mixing them gains only 0.00085 nats:
    # below an absolute threshold of 0.001, but 2% of the loss of 0.042.
    labels = np.array([0, 0, 1, 1])
    oof = {
        "A": _probs([0.999, 0.92, 0.999, 0.92], labels),
        "B": _probs([0.92, 0.999, 0.92, 0.999], labels),
    }
    assert no_harm_selection(oof, labels, min_relative_gain=0.01) == pytest.approx({"A": 0.5, "B": 0.5})


def test_accurate_but_overconfident_unit_is_left_out():
    # C is right on 9 of 10 cells (accuracy 0.9, enough for the R1 gates)
    # but puts 0.001 on the true label of the tenth. Mixed in at any weight
    # it raises A's log-loss, so it is not integrated.
    labels = np.array([0] * 5 + [1] * 5)
    oof = {
        "A": _probs([0.95] * 10, labels),
        "C": _probs([0.999] * 9 + [0.001], labels),
    }
    assert no_harm_selection(oof, labels, min_relative_gain=0.01) == pytest.approx({"A": 1.0})


def test_exact_duplicate_gets_no_weight_of_its_own():
    labels = np.array([0, 0, 1, 1])
    a = _probs([0.9, 0.5, 0.9, 0.5], labels)
    oof = {"A": a, "B": _probs([0.5, 0.9, 0.5, 0.9], labels), "A_copy": a.copy()}
    weights = no_harm_selection(oof, labels, min_relative_gain=0.01)
    assert len({"A", "A_copy"} & set(weights)) == 1
    assert sorted(weights.values()) == pytest.approx([0.5, 0.5])


def test_early_pick_made_redundant_by_later_units_is_dropped():
    # X (0.8 on both cells) is the best single unit, and forward selection
    # adds Y (sure about cell a) and then Z (sure about cell b) on top of it.
    # Once Y and Z are both in, X only dilutes them: removing it lowers the
    # OOF log-loss from 0.209 to 0.202, so the leave-one-out pass drops it.
    labels = np.array([0, 1])
    oof = {
        "X": _probs([0.8, 0.8], labels),
        "Y": _probs([0.999, 0.635], labels),
        "Z": _probs([0.635, 0.999], labels),
    }
    assert no_harm_selection(oof, labels, min_relative_gain=0.01) == pytest.approx({"Y": 0.5, "Z": 0.5})


def test_every_kept_unit_is_needed_by_the_others():
    # Eight units on 200 cells, each informed about a random half of the
    # cells (with its own sharpness) and noisy elsewhere.
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    oof = {}
    for i in range(8):
        informed = rng.random(200) < 0.5
        signal = np.where(informed, (2 * labels - 1) * rng.uniform(1.0, 4.0), 0.0)
        p1 = 1 / (1 + np.exp(-(signal + rng.normal(0, 0.5, size=200))))
        oof[f"u{i}"] = np.column_stack([1 - p1, p1])

    weights = no_harm_selection(oof, labels, min_relative_gain=0.01)

    def ensemble_loss(w):
        mix = sum(oof[u] * v for u, v in w.items()) / sum(w.values())
        return -np.mean(np.log(mix[np.arange(len(labels)), labels]))

    assert len(weights) > 1
    assert sum(weights.values()) == pytest.approx(1.0)
    full = ensemble_loss(weights)
    for unit_id in weights:
        rest = {u: v for u, v in weights.items() if u != unit_id}
        assert ensemble_loss(rest) - full >= 0.01 * full


def test_cell_weights_decide_which_units_are_worth_combining():
    # A is sure about the first half of the cells, B about the second half.
    # Counting every cell alike, A is the better unit and B still adds to
    # it; weighting the second half (the cells most like the progenitors)
    # makes B the better unit, and A then only dilutes it.
    labels = np.array([0, 1] * 50)
    first = np.arange(100) < 50
    oof = {
        "A": _probs(np.where(first, 0.95, 0.6), labels),
        "B": _probs(np.where(first, 0.6, 0.9), labels),
    }
    ordinary = no_harm_selection(oof, labels, min_relative_gain=0.01)
    shifted = no_harm_selection(oof, labels, min_relative_gain=0.01, weights=np.where(first, 0.1, 1.9))

    assert ordinary == pytest.approx({"A": 0.5, "B": 0.5})
    assert shifted == pytest.approx({"B": 1.0})
