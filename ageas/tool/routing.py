#!/usr/bin/env python3
"""Per-unit feature routing.

Every unit first trains on all features. Its own attribution table then
ranks the features it relies on, and the pruning ladder decides how far the
unit can be narrowed to them without losing performance.
"""
import numpy as np
import pandas as pd


def key_features(attribution: pd.DataFrame, mass: float, floor: int = 0) -> list:
    """Features a unit relies on at one pruning level.

    For each ``Class_{i}_Scores`` column the features are ranked by score,
    and the smallest top-ranked prefix whose positive scores add up to
    ``mass`` of the column's positive total is kept, topped up to ``floor``
    features. Only positively scored features count: a feature that does
    not push towards a class is never kept on that class's behalf.

    :param attribution: Per-feature, per-class score table; ``_Std``
        columns are ignored.
    :param mass: Fraction of each class's positive attribution to keep, in
        ``(0, 1]``.
    :param floor: Minimum number of positively scored features per class.
    :returns: The union over classes, in the table's feature order.
    """
    keep = np.zeros(len(attribution), dtype=bool)
    for col in attribution.columns:
        if not str(col).endswith('_Scores'):
            continue
        scores = attribution[col].to_numpy(dtype=float)
        order = np.argsort(-scores, kind='stable')
        positive = order[scores[order] > 0]
        if len(positive) == 0:
            continue
        cumulative = np.cumsum(scores[positive])
        n_keep = np.searchsorted(cumulative, mass * cumulative[-1] * (1 - 1e-12)) + 1
        keep[positive[:max(n_keep, floor)]] = True
    return attribution.index[keep].tolist()


def search_ladder(levels: list, full_score: float, score_at, relative_delta: float, choice: str = 'sparsest'):
    """Pruning level of a unit, searched down the ladder.

    Levels are tried from weakest to strongest, and the search stops at the
    first level whose score exceeds ``full_score * (1 + relative_delta)``.
    The tolerance is relative because terminal cells are often nearly
    separable: with an all-feature loss of 0.005, any absolute margin worth
    having elsewhere lets through versions several times worse. Scores are
    losses (lower is better), so a pruned version that beats the all-feature
    one always passes.

    :param levels: Mass levels ordered from weakest to strongest pruning,
        e.g. ``[0.95, 0.8, 0.6, 0.4]``.
    :param full_score: Validation loss of the all-feature version.
    :param score_at: Callable mapping a level to the validation loss of the
        version pruned to that level.
    :param relative_delta: Largest loss increase accepted for a narrower
        expert, as a fraction of ``full_score``.
    :param choice: ``'sparsest'`` keeps the strongest passing level, trading
        up to ``relative_delta`` of loss for fewer features. ``'best'`` keeps
        the version with the lowest loss among those tried, the all-feature
        one included (ties go to the weaker pruning): the unit works on the
        features it works best on, and the tolerance only bounds the search.
    :returns: The chosen level, or ``None`` when the unit keeps all features.
    """
    assert choice in ('sparsest', 'best'), "choice must be 'sparsest' or 'best'."
    chosen, best = None, full_score
    for level in levels:
        score = score_at(level)
        if score > full_score * (1 + relative_delta):
            break
        if choice == 'sparsest' or score < best:
            chosen, best = level, score
    return chosen
