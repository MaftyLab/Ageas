#!/usr/bin/env python3
"""Committee integration: competent, distinct units vote with equal weight.

The no-harm integration keeps the few units that lower the out-of-fold loss
on terminal cells. Those cells are nearly separable, so the units it keeps
agree almost everywhere and the ensemble is as sure on the progenitors as on
the terminal cells. The committee instead keeps every unit that is about as
accurate as the best, counts near-identical units once, drops a unit at
odds with the rest, and averages the members' class probabilities. Where
the members agree the call stays confident; where they split it hedges.
The no-harm mixture can join the vote with as much weight as the whole
committee. Only the query cells' features (through the members'
predictions) are used, never their labels.
"""
import numpy as np

from .synergy import log_loss

_EPS = 1e-12


def balanced_accuracy(probs: np.ndarray, labels: np.ndarray, weights=None) -> float:
    """Mean per-class recall of the calls in ``probs``, each cell counted with its weight."""
    labels = np.asarray(labels)
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    hits = np.asarray(probs).argmax(axis=1) == labels
    recalls = [np.average(hits[labels == c], weights=weights[labels == c]) for c in np.unique(labels)]
    return float(np.mean(recalls))


def competent_units(oof_probs: dict, labels: np.ndarray, weights=None, tolerance: float = 0.02) -> list:
    """Units whose out-of-fold balanced accuracy is within ``tolerance`` of the best unit's.

    :param oof_probs: ``unit_id -> (n_cells, n_classes)`` out-of-fold
        probabilities on the terminal cells.
    :param labels: The terminal cells' labels.
    :param weights: Optional weight of each terminal cell (shift-aware
        validation), used for both the accuracy and the loss.
    :param tolerance: Largest shortfall from the best balanced accuracy.
    :returns: The competent unit IDs, ranked by out-of-fold log-loss (best first).
    """
    accuracy = {u: balanced_accuracy(p, labels, weights) for u, p in oof_probs.items()}
    best = max(accuracy.values())
    kept = [u for u in oof_probs if accuracy[u] >= best - tolerance]
    return sorted(kept, key=lambda u: log_loss(oof_probs[u], labels, weights))


def entropy_bits(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits of each row (last axis) of ``probs``."""
    probs = np.asarray(probs, dtype=float)
    return -np.sum(probs * np.log2(probs + _EPS), axis=-1)


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits between matching rows of ``p`` and ``q`` (0 to 1)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return np.clip(entropy_bits((p + q) / 2) - (entropy_bits(p) + entropy_bits(q)) / 2, 0.0, None)


def select_committee(
    query_probs: dict, ranked: list, max_jsd: float = 0.01, min_majority: float = 0.5,
) -> list:
    """The committee drawn from ``ranked`` units, judged on their query probabilities.

    1. Walking ``ranked`` best first, a unit joins unless its class
       probabilities differ from an existing member's by less than
       ``max_jsd`` bits on average over the query cells: such a pair is one
       view counted twice.
    2. With three or more members, a member is dropped when its call matches
       the call of the other members' average on fewer than
       ``min_majority`` of the query cells: it has collapsed or flipped
       where the rest have not. Every member is judged against the same
       committee, in one pass.

    :param query_probs: ``unit_id -> (n_cells, n_classes)`` probabilities on
        the query cells.
    :param ranked: Candidate units, best first.
    :param max_jsd: Mean Jensen-Shannon divergence (bits) below which two
        units count as one.
    :param min_majority: Smallest share of query cells on which a member
        must side with the rest of the committee.
    :returns: Member unit IDs, best first.
    """
    members = []
    for unit_id in ranked:
        if all(js_divergence_bits(query_probs[unit_id], query_probs[m]).mean() >= max_jsd for m in members):
            members.append(unit_id)
    if len(members) < 3:
        return members

    def sides_with_rest(unit_id) -> float:
        rest = np.mean([query_probs[m] for m in members if m != unit_id], axis=0)
        return float((np.asarray(query_probs[unit_id]).argmax(axis=1) == rest.argmax(axis=1)).mean())

    return [m for m in members if sides_with_rest(m) >= min_majority]


def _combine(stack: np.ndarray, rule: str) -> np.ndarray:
    """Average a ``(n_members, n_cells, n_classes)`` stack by ``rule``."""
    if rule == 'logit':
        log_mean = np.log(np.clip(stack, _EPS, None)).mean(axis=0)
        probs = np.exp(log_mean - log_mean.max(axis=1, keepdims=True))
        return probs / probs.sum(axis=1, keepdims=True)
    return stack.mean(axis=0)


def vote(member_probs: dict, rule: str = 'mean', groups: dict = None, mixture: np.ndarray = None) -> dict:
    """Equal-weight soft vote of the members' class probabilities.

    :param member_probs: ``unit_id -> (n_cells, n_classes)`` probabilities.
    :param rule: ``'mean'`` averages the probabilities; ``'logit'`` averages
        their logarithms and renormalises (the normalised geometric mean; with
        two classes, the average log-odds), so a surer member counts for more.
    :param groups: Optional ``unit_id -> group`` (e.g. the model type). The
        members of each group vote among themselves first, then every group
        has one voice, so a type with many variants does not outvote the rest.
    :param mixture: Optional ``(n_cells, n_classes)`` probabilities of the
        no-harm mixture. The committee's average and the mixture are then
        averaged by ``rule`` with equal weight, so the mixture counts as much
        as the whole committee.
    :returns: ``probs`` (the committee's average, with the mixture when one
        is given), ``agreement`` (share of members whose own call is that
        call) and ``disagreement`` (entropy of the members' plain average
        minus their average entropy, in bits: 0 when every member gives the
        same distribution).
    """
    assert rule in ('mean', 'logit'), "rule must be 'mean' or 'logit'."
    stack = np.stack([np.asarray(p, dtype=float) for p in member_probs.values()])
    if groups is None:
        probs = _combine(stack, rule)
    else:
        by_group = {}
        for unit_id, probs in member_probs.items():
            by_group.setdefault(groups[unit_id], []).append(np.asarray(probs, dtype=float))
        probs = _combine(np.stack([_combine(np.stack(m), rule) for m in by_group.values()]), rule)
    if mixture is not None:
        probs = _combine(np.stack([probs, np.asarray(mixture, dtype=float)]), rule)
    call = probs.argmax(axis=1)
    agreement = (stack.argmax(axis=2) == call).mean(axis=0)
    disagreement = np.clip(entropy_bits(stack.mean(axis=0)) - entropy_bits(stack).mean(axis=0), 0.0, None)
    return {'probs': probs, 'agreement': agreement, 'disagreement': disagreement}
