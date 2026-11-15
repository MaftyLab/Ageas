#!/usr/bin/env python3
"""Checking the panel strategy against validation.

Each panel rule (:mod:`ageas.panel`) names the units it expects to
validate well and those it expects to validate poorly. After a run, the
units' out-of-fold log-losses on the terminal cells say whether each
expectation held, under ordinary validation and, given the cell weights,
under shift-aware validation. Only terminal cells are scored.
"""
import numpy as np
from scipy.stats import spearmanr

from .synergy import log_loss


def check_strategy(panel: list, oof_probs: dict, labels, cell_weights=None) -> dict:
    """Whether validation bears out what each panel rule expected.

    :param panel: The run's :class:`~ageas.UnitSpec` list.
    :param oof_probs: ``unit_id -> (n_cells, n_classes)`` out-of-fold
        probabilities on the terminal cells (``deck.oof['probs']``).
    :param labels: Integer class of every terminal cell.
    :param cell_weights: Shift-aware weight of every terminal cell, or
        ``None`` to report ordinary validation only.
    :returns: ``{'units': [...], 'rules': [...], 'agreement': {...}}``.
        Each unit row has its ``unit_id``, ``type``, ``preference`` (rules
        favouring it minus rules disfavouring it) and log-loss under every
        validation; each rule row has ``n_favoured``, ``n_disfavoured`` and,
        per validation, the mean ``favoured_loss`` and ``disfavoured_loss``
        and ``borne_out`` (``None`` when the rule has no units on one side).
        ``agreement`` maps each validation to the Spearman correlation
        between the units' preference and their validation (minus their
        log-loss), ``None`` when the preferences or losses do not vary.
    """
    views = {'ordinary': None} if cell_weights is None else {'ordinary': None, 'shift': cell_weights}
    units = []
    for spec in panel:
        if spec.unit_id not in oof_probs:
            continue
        row = {'unit_id': spec.unit_id, 'type': spec.type, 'spec': spec,
               'preference': len(spec.favoured_by) - len(spec.disfavoured_by)}
        for view, weights in views.items():
            row[view] = log_loss(oof_probs[spec.unit_id], labels, weights)
        units.append(row)

    rules = []
    for rule in sorted({r for u in units for r in u['spec'].favoured_by + u['spec'].disfavoured_by}):
        favoured = [u for u in units if rule in u['spec'].favoured_by]
        disfavoured = [u for u in units if rule in u['spec'].disfavoured_by]
        entry = {'rule': rule, 'n_favoured': len(favoured), 'n_disfavoured': len(disfavoured)}
        for view in views:
            f = float(np.mean([u[view] for u in favoured])) if favoured else None
            d = float(np.mean([u[view] for u in disfavoured])) if disfavoured else None
            entry[view] = {
                'favoured_loss': f,
                'disfavoured_loss': d,
                'borne_out': None if f is None or d is None else bool(f < d),
            }
        rules.append(entry)

    agreement = {}
    preference = [u['preference'] for u in units]
    for view in views:
        losses = [u[view] for u in units]
        varied = len(set(preference)) > 1 and len(set(losses)) > 1
        agreement[view] = float(spearmanr(preference, [-v for v in losses])[0]) if varied else None
    for u in units:
        del u['spec']
    return {'units': units, 'rules': rules, 'agreement': agreement}
