"""Pruning ladder: which features a unit keeps at each level.

A unit's attribution table ranks the features it relies on. At mass level
``m`` each class keeps the smallest top-ranked prefix holding ``m`` of its
positive attribution, and the unit works on the union over classes.
"""
import pandas as pd

from ageas import key_features, search_ladder

LADDER = [0.95, 0.8, 0.6, 0.4]


def _attribution():
    # Class 0 leans on a and b, class 1 on e and f; each class's positive
    # scores add up to 1. The Std column must not influence the selection.
    return pd.DataFrame(
        {
            "Class_0_Scores": [0.5, 0.3, 0.1, 0.1, -0.2, 0.0],
            "Class_0_Std": [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            "Class_1_Scores": [-0.5, -0.3, 0.0, 0.0, 0.6, 0.4],
        },
        index=list("abcdef"),
    )


def test_each_class_keeps_the_smallest_prefix_reaching_the_mass():
    table = _attribution()
    assert key_features(table, mass=0.8) == ["a", "b", "e", "f"]
    assert key_features(table, mass=0.5) == ["a", "e"]


def test_floor_tops_up_but_never_adds_features_a_class_does_not_rely_on():
    table = _attribution()
    # Class 0 tops up to three features (c beats its tie with d by order).
    # Class 1 has only two positive features, so the floor cannot add the
    # zero-scored c or d on its behalf.
    assert key_features(table, mass=0.5, floor=3) == ["a", "b", "c", "e", "f"]


def _scorer(losses):
    """Stand-in for training at a level: returns its validation loss and logs the call."""
    tried = []

    def score_at(level):
        tried.append(level)
        return losses[level]

    return score_at, tried


def test_ladder_keeps_the_last_level_before_the_first_failure():
    # With relative_delta 0.2 the limit is 0.12. 0.6 exceeds it, so the
    # search stops there and never trains 0.4, even though 0.4 would pass.
    score_at, tried = _scorer({0.95: 0.10, 0.8: 0.11, 0.6: 0.13, 0.4: 0.10})
    assert search_ladder(LADDER, full_score=0.10, score_at=score_at, relative_delta=0.2) == 0.8
    assert tried == [0.95, 0.8, 0.6]


def test_ladder_keeps_all_features_when_the_first_level_fails():
    score_at, tried = _scorer({0.95: 0.30})
    assert search_ladder(LADDER, full_score=0.10, score_at=score_at, relative_delta=0.2) is None
    assert tried == [0.95]


def test_ladder_accepts_losses_within_the_tolerance_or_better():
    # 0.75 sits exactly on 0.5 * (1 + 0.5); every other level beats the
    # all-feature loss, so the strongest level wins.
    score_at, _ = _scorer({0.95: 0.75, 0.8: 0.40, 0.6: 0.5, 0.4: 0.25})
    assert search_ladder(LADDER, full_score=0.5, score_at=score_at, relative_delta=0.5) == 0.4


def test_best_choice_keeps_the_best_validated_version_the_search_reached():
    # Limit 0.12: 0.4 fails and ends the search. Among the versions tried,
    # 0.8 has the lowest loss, where the sparsest choice would take 0.6.
    losses = {0.95: 0.09, 0.8: 0.08, 0.6: 0.11, 0.4: 0.13}
    score_at, tried = _scorer(losses)
    assert search_ladder(LADDER, full_score=0.10, score_at=score_at, relative_delta=0.2, choice='best') == 0.8
    assert tried == [0.95, 0.8, 0.6, 0.4]
    score_at, _ = _scorer(losses)
    assert search_ladder(LADDER, full_score=0.10, score_at=score_at, relative_delta=0.2) == 0.6


def test_best_choice_keeps_all_features_when_every_narrower_version_is_worse():
    # Zheng68K: every pruned version validated worse than the all-feature
    # one, yet within the tolerance, so the sparsest choice narrowed anyway.
    score_at, _ = _scorer({0.95: 0.105, 0.8: 0.11, 0.6: 0.13})
    assert search_ladder(LADDER, full_score=0.10, score_at=score_at, relative_delta=0.2, choice='best') is None


def test_tolerance_scales_with_the_all_feature_loss():
    # Near-separable terminal cells (LARRY): an absolute margin of 0.02 let a
    # version four times worse than the all-feature one through. Relative
    # to a loss of 0.005, it is far outside a 10% tolerance.
    score_at, _ = _scorer({0.95: 0.02})
    assert search_ladder(LADDER, full_score=0.005, score_at=score_at, relative_delta=0.1) is None
