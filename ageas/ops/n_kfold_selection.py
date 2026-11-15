#!/usr/bin/env python3
"""N-iteration k-fold model selection.

Runs successive rounds of k-fold cross-validation over a hangar's units,
keeping only the units that survive the configured retention/cutoff thresholds
between rounds. After all rounds the surviving models are retrained on the
full dataset (the "last mission") to produce the final deck used for
downstream prediction or factor extraction.

Each unit's out-of-fold (OOF) metrics from the last selection round are
stored under ``unit.report[operation_name]['final']['oof']`` (keys prefixed
with ``oof.``) and are used both for the final retention gate and, by
default, for weighting explanations in :meth:`~ageas.Deck.debrief`.
"""
import logging
import time
from warnings import warn

import numpy as np

from ageas.deck import Deck
from ageas.hangar import Hangar
from ageas.tool import kfold_random_split

_logger = logging.getLogger(__name__)

#: Final retention gates supported by :func:`main`.
FINAL_GATES = ('oof', 'test', 'insample', 'none')


def _better(a: float, b: float, monitor_type: str) -> bool:
    """Return ``True`` if score ``a`` passes threshold ``b``."""
    return a > b if monitor_type == 'max' else a < b


def _oof_summary(fold_records: list) -> dict:
    """Average per-fold test metrics into an ``oof.*`` dictionary."""
    values: dict = {}
    for rec in fold_records:
        if rec is None:
            continue
        for name, val in rec.items():
            values.setdefault(name, []).append(float(val))
    return {
        'oof.' + name.split('.', 1)[-1]: float(np.mean(vals))
        for name, vals in values.items()
    }


def main(
    hangar: Hangar,
    operation_name: str = 'trail',
    accelerator: str = 'cpu',
    cuda_devices: list = None,
    query_dataset=None,
    test_dataset=None,
    n_dataloader_workers: int = 1,
    using_model_types: list = None,
    using_model_list: list = None,
    kfold_selection_list: list = None,
    valid_fraction: float = 0.1,
    stratified_kfold_test: bool = False,
    stratified_kfold_valid: bool = False,
    oversample_method: str = None,
    oversample_by: str = 'median',
    monitor_type: str = 'max',
    monitor_metric: str = 'test.accuracy',
    retention_point: float = 0.9,
    cutoff_point: float = 0.7,
    selection_ratio: float = 0.5,
    relative_retention: float = None,
    final_gate: str = 'oof',
    skip_final: bool = False,
    seed: int = 42,
    verbose: bool = None,
) -> 'Deck':
    """N-iteration k-fold selection of top-performing models.

    Each iteration performs a fresh k-fold cross-validation over the squad,
    ranks the units by their mean out-of-fold ``monitor_metric`` and keeps
    the top ``selection_ratio`` plus any unit beyond ``retention_point``,
    while discarding everything not beyond ``cutoff_point``. If no unit
    passes, the best-ranked unit is kept so the ensemble is never empty.
    After all iterations a final pass (the "last mission") retrains the
    survivors on the full dataset and applies a final retention filter
    chosen by ``final_gate``.

    :param hangar: Source hangar from which the operating squad is generated.
    :param operation_name: Name under which the per-unit reports are stored.
    :param accelerator: Accelerator hint, ``'cpu'`` or ``'cuda'``.
    :param cuda_devices: Optional GPU device indices. ``None`` uses all
        visible devices.
    :param query_dataset: Dataset that is split by k-fold for training and
        validation.
    :param test_dataset: Optional held-out test set used in the last mission
        (and by ``final_gate='test'``).
    :param n_dataloader_workers: Number of dataloader workers for the deck.
    :param using_model_types: Whitelist of model types to include from the
        hangar.
    :param using_model_list: Whitelist of explicit unit IDs to include from
        the hangar.
    :param kfold_selection_list: Fold counts per selection round. ``[5]`` runs
        a single 5-fold round; ``[2, 3, 4]`` would run three rounds. Defaults
        to ``[5]``.
    :param valid_fraction: Fraction of each training fold reserved for
        validation.
    :param stratified_kfold_test: If ``True``, the test fold is
        class-stratified.
    :param stratified_kfold_valid: If ``True``, the validation split is
        class-stratified.
    :param oversample_method: Oversampling method (e.g. ``'repeat'``).
        ``None`` disables oversampling.
    :param oversample_by: Target class size when oversampling: ``'mean'``,
        ``'median'``, or ``'max'``.
    :param monitor_type: ``'max'`` for higher-is-better metrics, ``'min'``
        for lower-is-better.
    :param monitor_metric: Dotted metric used to rank units between rounds,
        e.g. ``'test.accuracy'`` or ``'test.f1'``.
    :param retention_point: Units beyond this metric value are retained
        regardless of their rank.
    :param cutoff_point: Hard threshold: units not beyond this value are
        dropped.
    :param selection_ratio: Fraction of the squad to keep when ranked by the
        metric.
    :param relative_retention: If set, replaces ``retention_point`` by
        ``best - relative_retention`` (or ``best + relative_retention`` for
        ``'min'`` metrics), where ``best`` is the best out-of-fold score of
        the round. This makes the retention gate portable across datasets of
        different difficulty.
    :param final_gate: Metric source for the final retention filter:
        ``'oof'`` (default) uses the out-of-fold score of the last selection
        round; ``'test'`` uses ``test_dataset`` (required); ``'insample'``
        reproduces the pre-revision behaviour of scoring the refitted models
        on their own training data; ``'none'`` keeps every survivor.
    :param skip_final: If ``True``, skip the last mission retraining pass.
    :param seed: Random seed forwarded to the k-fold splitter.
    :param verbose: If ``True``, emit per-fold and per-round progress logs.
    :returns: The deck after selection with only the surviving units.
    """
    if kfold_selection_list is None:
        kfold_selection_list = [5]
    if monitor_type not in ('max', 'min'):
        raise ValueError("monitor_type must be 'max' or 'min'.")
    if final_gate not in FINAL_GATES:
        raise ValueError(f"final_gate must be one of {FINAL_GATES}.")
    if final_gate == 'test' and test_dataset is None:
        raise ValueError("final_gate='test' requires a test_dataset.")

    n_classes = len(query_dataset.label_dict)
    fea_names = query_dataset.adata.var.index.tolist()
    metric_name = monitor_metric.split('.', 1)[-1]

    deck = Deck(
        squad=hangar.sortie_generate(
            unit_types=using_model_types,
            unit_list=using_model_list,
        ),
        n_dataloader_workers=n_dataloader_workers,
        accelerator=accelerator,
        cuda_devices=cuda_devices,
    )
    assert len(deck.squad) > 0, "Not enough models in the squad."
    for unit in deck.squad.values():
        unit.report[operation_name] = dict()

    oof_scores: dict = {}
    for deploy_i, k_fold in enumerate(kfold_selection_list):
        start_time = time.time()
        expect_survival = max(int(len(deck.squad) * selection_ratio), 1)
        _logger.info(
            "Deployment %d / %d  |  k=%d  |  units=%d",
            deploy_i + 1,
            len(kfold_selection_list),
            k_fold,
            len(deck.squad),
        )

        train_list, valid_list, test_list = kfold_random_split(
            query_dataset,
            n_splits=k_fold,
            valid_fraction=valid_fraction,
            stratified_test=stratified_kfold_test,
            stratified_valid=stratified_kfold_valid,
            oversample_method=oversample_method,
            oversample_by=oversample_by,
            random_seed=seed,
        )
        assert len(train_list) == len(test_list) == k_fold, \
            "Train and test data lists must have the same length."

        round_report = deck.make_report()
        for i in range(k_fold):
            _logger.info("Fold %d / %d", i + 1, k_fold)
            deck.sortie(
                train_data=train_list[i],
                vali_data=valid_list[i],
                n_classes=n_classes,
                fea_names=fea_names,
                test_dataset=test_list[i],
                report=round_report,
                save_model=False,
                save_trainer=False,
                verbose=verbose,
            )

        oof_scores = {
            unit_id: _oof_summary(rec['test'])
            for unit_id, rec in round_report.items()
            if unit_id in deck.squad
        }
        unscored = [u for u, oof in oof_scores.items() if f'oof.{metric_name}' not in oof]
        if unscored:
            warn(f"Dropping units without '{monitor_metric}' records: {unscored}")
        unit_rank = sorted(
            (
                (unit_id, oof[f'oof.{metric_name}'])
                for unit_id, oof in oof_scores.items()
                if unit_id not in unscored
            ),
            key=lambda x: x[1],
            reverse=monitor_type == 'max',
        )
        assert len(unit_rank) > 0, "No unit produced out-of-fold scores."

        retention = retention_point
        if relative_retention is not None:
            best = unit_rank[0][1]
            retention = (
                best - relative_retention
                if monitor_type == 'max'
                else best + relative_retention
            )

        survivors = [
            unit_id
            for rank, (unit_id, score) in enumerate(unit_rank)
            if (rank < expect_survival or _better(score, retention, monitor_type))
            and _better(score, cutoff_point, monitor_type)
        ]
        if not survivors:
            warn(
                f"No unit passed the selection gates in round {deploy_i + 1}; "
                f"keeping the best unit ({unit_rank[0][0]})."
            )
            survivors = [unit_rank[0][0]]

        for unit_id, unit in deck.squad.items():
            unit.report[operation_name][f'round_{deploy_i + 1}'] = {
                'test': round_report[unit_id]['test'],
                'oof': oof_scores[unit_id],
            }
        deck.squad_update(survivors)

        _logger.info(
            "Survived: %d  |  elapsed: %.1fs",
            len(deck.squad),
            time.time() - start_time,
        )

    if not skip_final:
        _logger.info("Last mission start.")
        report = deck.sortie(
            report=deck.make_report(),
            train_data=query_dataset,
            vali_data=query_dataset,
            n_classes=n_classes,
            fea_names=fea_names,
            test_dataset=test_dataset,
            save_model=True,
            save_trainer=True,
            verbose=verbose,
        )

        gate_scores = {}
        for unit_id, unit in deck.squad.items():
            final = {
                'vali': report[unit_id]['vali'][0],
                'test': report[unit_id]['test'][0],
                'oof': oof_scores[unit_id],
            }
            unit.report[operation_name]['final'] = final
            if final_gate == 'oof':
                gate_scores[unit_id] = final['oof'][f'oof.{metric_name}']
            elif final_gate == 'test':
                gate_scores[unit_id] = float(final['test'][f'test.{metric_name}'])
            elif final_gate == 'insample':
                gate_scores[unit_id] = float(final['vali'][f'vali.{metric_name}'])

        if final_gate == 'none':
            final_list = list(deck.squad)
        else:
            retention = retention_point
            if relative_retention is not None:
                best = (max if monitor_type == 'max' else min)(gate_scores.values())
                retention = (
                    best - relative_retention
                    if monitor_type == 'max'
                    else best + relative_retention
                )
            final_list = [
                unit_id for unit_id, score in gate_scores.items()
                if score == retention or _better(score, retention, monitor_type)
            ]
            if not final_list:
                best_unit = (max if monitor_type == 'max' else min)(
                    gate_scores, key=gate_scores.get
                )
                warn(
                    "No unit passed the final retention gate; "
                    f"keeping the best unit ({best_unit})."
                )
                final_list = [best_unit]
        deck.squad_update(final_list)
        _logger.info("Last mission end. Survivors: %d", len(deck.squad))

    return deck
