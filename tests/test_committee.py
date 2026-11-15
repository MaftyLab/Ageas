"""The committee: competent, distinct units that vote with equal weight.

Units are kept when they are about as accurate as the best on held-out
terminal cells; on the query cells, near-identical units count once and a
unit at odds with the rest of the committee is dropped. The members'
class probabilities are averaged, so the call is confident where they agree
and hedged where they split.
"""
import numpy as np

from ageas.tool.committee import competent_units, select_committee, vote


def _two_class(p1) -> np.ndarray:
    """``(n, 2)`` probabilities from each cell's class-1 probability."""
    p1 = np.asarray(p1, dtype=float)
    return np.column_stack([1.0 - p1, p1])


def _units() -> dict:
    """100 query cells: class 1 on cells 0-69 at 0.9, class 0 on 70-99 at 0.9.

    ``twin`` is ``best`` give or take 0.01; ``other`` and ``third`` each turn
    a different block of 20 cells the other way.
    """
    best = np.where(np.arange(100) < 70, 0.9, 0.1)
    twin = np.where(best > 0.5, 0.89, 0.11)
    other, third = best.copy(), best.copy()
    other[:20] = 0.2
    third[70:90] = 0.8
    return {name: _two_class(p) for name, p in
            [('best', best), ('twin', twin), ('other', other), ('third', third)]}


def test_near_identical_units_count_once_keeping_the_better_ranked():
    units = _units()

    committee = select_committee(units, ranked=['best', 'twin', 'other', 'third'])

    assert committee == ['best', 'other', 'third']


def test_a_unit_at_odds_with_the_rest_of_the_committee_is_dropped():
    # ``broken`` calls class 0 on every cell. The rest's consensus is class 1
    # on cells 0-69, so it sides with them on only 30 of 100 cells. ``other``
    # and ``third`` each dissent on one block of 20 cells and stay.
    units = _units()
    units['broken'] = _two_class(np.full(100, 0.05))

    committee = select_committee(units, ranked=['best', 'other', 'third', 'broken'])

    assert committee == ['best', 'other', 'third']


def test_competent_units_are_within_two_points_of_the_best_and_ranked_by_loss():
    # 50 terminal cells of each class, out of fold.
    labels = np.repeat([0, 1], 50)
    right = np.where(labels == 1, 0.9, 0.1)
    oof = {
        'sharp': _two_class(np.where(labels == 1, 0.99, 0.01)),  # balanced accuracy 1.00, lowest loss
        'good': _two_class(right),                                # 1.00
        'close': _two_class(np.r_[right[:99], 0.1]),              # one class-1 cell wrong: 0.99
        'poor': _two_class(np.r_[right[:90], np.full(10, 0.1)]),  # ten class-1 cells wrong: 0.90
    }

    ranked = competent_units(oof, labels, tolerance=0.02)

    assert ranked == ['sharp', 'good', 'close']


def test_vote_is_confident_where_members_agree_and_hedges_where_they_split():
    # Cell 0: all three say class 0 at 0.9. Cell 1: two say class 0, one class 1.
    members = {
        'a': np.array([[0.9, 0.1], [0.9, 0.1]]),
        'b': np.array([[0.9, 0.1], [0.9, 0.1]]),
        'c': np.array([[0.9, 0.1], [0.1, 0.9]]),
    }

    result = vote(members)

    np.testing.assert_allclose(result['probs'], [[0.9, 0.1], [0.6333, 0.3667]], atol=1e-4)
    np.testing.assert_allclose(result['agreement'], [1.0, 2 / 3])
    # Disagreement in bits: entropy of the average minus the average entropy,
    # worked by hand: H(0.633, 0.367) - H(0.9, 0.1) = 0.9481 - 0.4690.
    np.testing.assert_allclose(result['disagreement'], [0.0, 0.4791], atol=1e-3)


def test_each_model_type_can_get_one_voice():
    # Three ResNet variants say class 0 at 0.9; one GRU says class 1 at 0.8.
    # Counting units, the ResNets outvote it 3 to 1: (2.7 + 0.2) / 4 = 0.725.
    # Counting model types, each type has one voice: (0.9 + 0.2) / 2 = 0.55.
    members = {f'resnet_{k}': np.array([[0.9, 0.1]]) for k in 'abc'}
    members['rnn_gru'] = np.array([[0.2, 0.8]])
    groups = {u: u.split('_')[0] for u in members}

    np.testing.assert_allclose(vote(members)['probs'], [[0.725, 0.275]])
    result = vote(members, groups=groups)
    np.testing.assert_allclose(result['probs'], [[0.55, 0.45]])
    np.testing.assert_allclose(result['agreement'], [0.75])  # still counted over members


def test_the_no_harm_mixture_can_count_as_much_as_the_whole_committee():
    # Members at 0.9, 0.7 and 0.2 for class 0 average 0.6, a narrow call for
    # class 0. The mixture says 0.2. Half committee, half mixture: (0.6 + 0.2) / 2
    # = 0.4, so the call flips to class 1, which one member of three backs.
    # As one more voter the mixture would only reach (0.9 + 0.7 + 0.2 + 0.2) / 4 = 0.5.
    members = {'a': np.array([[0.9, 0.1]]), 'b': np.array([[0.7, 0.3]]), 'c': np.array([[0.2, 0.8]])}

    result = vote(members, mixture=np.array([[0.2, 0.8]]))

    np.testing.assert_allclose(result['probs'], [[0.4, 0.6]])
    np.testing.assert_allclose(result['agreement'], [1 / 3])


def test_the_mixture_joins_the_committee_in_log_odds_under_the_logit_rule():
    # The committee of 0.9 and 0.6 for class 0 sits at log-odds 1.301 (0.786).
    # The mixture at 0.3 sits at ln(3/7) = -0.847. Their average, 0.227, is 0.557.
    members = {'sure': np.array([[0.9, 0.1]]), 'unsure': np.array([[0.6, 0.4]])}

    result = vote(members, rule='logit', mixture=np.array([[0.3, 0.7]]))

    np.testing.assert_allclose(result['probs'], [[0.557, 0.443]], atol=1e-3)


def test_log_odds_vote_lets_the_surer_member_count_for_more():
    # Members at 0.9 and 0.6 for class 0. In log-odds, ln 9 = 2.197 and
    # ln 1.5 = 0.405 average 1.301, which is 0.786; a plain average gives 0.75.
    members = {'sure': np.array([[0.9, 0.1]]), 'unsure': np.array([[0.6, 0.4]])}

    result = vote(members, rule='logit')

    np.testing.assert_allclose(result['probs'], [[0.786, 0.214]], atol=1e-3)
    np.testing.assert_allclose(result['agreement'], [1.0])
