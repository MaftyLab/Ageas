"""Checking the panel strategy against validation.

Each panel rule says which of its units it expects to validate well and
which poorly. After the run, the units' out-of-fold losses on the terminal
cells tell whether each expectation held, under shift-aware and under
ordinary validation.
"""
import numpy as np
import pytest

from ageas import UnitSpec, check_strategy


def _probs(p_true, labels):
    p_true = np.asarray(p_true, dtype=float)
    rows = np.arange(len(labels))
    probs = np.empty((len(labels), 2))
    probs[rows, labels] = p_true
    probs[rows, 1 - labels] = 1 - p_true
    return probs


def _unit(name, favoured_by=(), disfavoured_by=()):
    return UnitSpec(name, 'logreg', {}, favoured_by=list(favoured_by), disfavoured_by=list(disfavoured_by))


def test_each_rule_is_borne_out_or_contradicted_by_its_units_losses():
    labels = np.array([0, 1] * 20)
    panel = [
        _unit('good', favoured_by=['right_rule'], disfavoured_by=['wrong_rule']),
        _unit('poor', favoured_by=['wrong_rule'], disfavoured_by=['right_rule']),
    ]
    oof = {'logreg_good': _probs([0.9] * 40, labels), 'logreg_poor': _probs([0.6] * 40, labels)}

    report = check_strategy(panel, oof, labels)

    rules = {r['rule']: r for r in report['rules']}
    assert rules['right_rule']['ordinary']['borne_out'] is True
    assert rules['wrong_rule']['ordinary']['borne_out'] is False
    assert rules['right_rule']['ordinary']['favoured_loss'] == pytest.approx(-np.log(0.9))
    assert rules['right_rule']['ordinary']['disfavoured_loss'] == pytest.approx(-np.log(0.6))


def test_shift_aware_and_ordinary_validation_can_judge_a_rule_differently():
    # 'sure_first' is better on the first half of the cells, 'sure_second'
    # on the second half, which is the half most like the progenitors.
    labels = np.array([0, 1] * 50)
    first = np.arange(100) < 50
    panel = [_unit('sure_first', favoured_by=['rule']), _unit('sure_second', disfavoured_by=['rule'])]
    oof = {
        'logreg_sure_first': _probs(np.where(first, 0.95, 0.6), labels),
        'logreg_sure_second': _probs(np.where(first, 0.6, 0.9), labels),
    }

    report = check_strategy(panel, oof, labels, cell_weights=np.where(first, 0.1, 1.9))

    (rule,) = report['rules']
    assert rule['ordinary']['borne_out'] is True
    assert rule['shift']['borne_out'] is False


def test_overall_agreement_ranks_units_by_how_strongly_the_rules_favour_them():
    labels = np.array([0, 1] * 20)
    panel = [
        _unit('best', favoured_by=['a', 'b']),
        _unit('middle', favoured_by=['a'], disfavoured_by=['b']),
        _unit('worst', disfavoured_by=['a', 'b']),
    ]
    oof = {
        'logreg_best': _probs([0.95] * 40, labels),
        'logreg_middle': _probs([0.8] * 40, labels),
        'logreg_worst': _probs([0.6] * 40, labels),
    }

    report = check_strategy(panel, oof, labels)

    assert report['agreement']['ordinary'] == pytest.approx(1.0)
