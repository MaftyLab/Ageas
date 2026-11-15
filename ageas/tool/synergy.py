#!/usr/bin/env python3
"""No-harm integration of experts.

Units are combined only when they do better together. Selection runs on
out-of-fold (OOF) probabilities with greedy forward selection (Caruana et
al., 2004): repeats are allowed and act as weights. A leave-one-out pass
then drops every unit whose removal does not make the ensemble worse, so no
integrated unit hurts the others.
"""
import numpy as np

_EPS = 1e-12


def log_loss(probs: np.ndarray, labels: np.ndarray, weights: np.ndarray = None) -> float:
    """Mean negative log-probability that ``probs`` gives each cell's own label.

    :param weights: Optional weight of each cell (e.g. shift-aware validation
        weights); the mean is then weighted.
    """
    labels = np.asarray(labels, dtype=int)
    p_true = np.asarray(probs, dtype=float)[np.arange(len(labels)), labels]
    return float(-np.average(np.log(np.clip(p_true, _EPS, 1.0)), weights=weights))


def _mixture(oof_probs: dict, counts: dict) -> np.ndarray:
    """Count-weighted mean of the members' probabilities."""
    total = sum(counts.values())
    return sum(oof_probs[unit_id] * (n / total) for unit_id, n in counts.items())


def no_harm_selection(
    oof_probs: dict,
    labels,
    min_relative_gain: float = 0.01,
    max_steps: int = 50,
    weights: np.ndarray = None,
) -> dict:
    """Weights of the units that improve the ensemble's OOF log-loss.

    Starts from the best single unit and repeatedly adds whichever unit
    (repeats allowed) lowers the OOF log-loss of the count-weighted mean
    the most, until the best addition gains less than ``min_relative_gain``
    of the current loss. Then, while some unit's removal (all of its copies)
    would not raise the log-loss by at least that fraction, the unit whose
    removal hurts least is dropped. Every unit that stays therefore improves
    the ensemble by at least ``min_relative_gain`` of its loss. Gains are
    relative because terminal-cell losses can be tiny, which would make any
    fixed threshold either meaningless or impossible to clear.

    :param oof_probs: Mapping ``unit_id -> (n_cells, n_classes)`` OOF
        probabilities, rows aligned with ``labels``.
    :param labels: Integer class label of every cell.
    :param min_relative_gain: Smallest log-loss improvement a unit must
        bring, both to join and to stay, as a fraction of the ensemble's
        current loss.
    :param max_steps: Largest number of additions, repeats included.
    :param weights: Optional weight of each cell in the log-loss, e.g. the
        shift-aware validation weights; ``None`` counts every cell alike.
    :returns: Mapping ``unit_id -> weight`` over the selected units; the
        weights sum to one.
    """
    labels = np.asarray(labels, dtype=int)

    def loss(counts):
        return log_loss(_mixture(oof_probs, counts), labels, weights)

    first = min(oof_probs, key=lambda unit_id: log_loss(oof_probs[unit_id], labels, weights))
    counts = {first: 1}
    current = loss(counts)
    for _ in range(max_steps - 1):
        trials = {
            unit_id: loss({**counts, unit_id: counts.get(unit_id, 0) + 1})
            for unit_id in oof_probs
        }
        pick = min(trials, key=trials.get)
        if current - trials[pick] < min_relative_gain * current:
            break
        counts[pick] = counts.get(pick, 0) + 1
        current = trials[pick]

    while len(counts) > 1:
        without = {
            unit_id: loss({k: n for k, n in counts.items() if k != unit_id})
            for unit_id in counts
        }
        weakest = min(without, key=without.get)
        if without[weakest] - current >= min_relative_gain * current:
            break
        del counts[weakest]
        current = without[weakest]

    total = sum(counts.values())
    return {unit_id: n / total for unit_id, n in counts.items()}
