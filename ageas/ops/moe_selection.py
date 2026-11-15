#!/usr/bin/env python3
"""Ageas v2 selection: route every unit to its key features, then integrate without harm.

In each fold every unit is trained on all features of the fold's training
part, its own attributions rank the features it relies on, and the pruning
ladder, judged on the fold's validation split, narrows it: as far as its
loss stays within ``relative_delta`` of its all-feature version's, or to the
version that validates best (``ladder_choice``). The chosen version
predicts the fold's held-out cells, which builds an out-of-fold (OOF)
record for every unit. :func:`~ageas.no_harm_selection` integrates the
units on that record. The units whose out-of-fold balanced accuracy is close
to the best are the committee's candidates. Each integrated or competent
unit is refitted on all terminal cells at the median of its per-fold levels.
"""
import copy
import logging
import os
import subprocess
import tempfile
from warnings import warn

import numpy as np
import pytorch_lightning as pl
import torch

import ageas
from ageas.hangar import Hangar
from ageas.tool import Trainer_Maker
from ageas.tool.corpus_loader import get_all_data
from ageas.unit import SUPPORTED_TYPES

from ..tool.committee import competent_units
from ..tool.feature_corpus import Feature_Corpus, kfold_indices
from ..moe_deck import Expert, MoE_Deck, predict_probs
from ..tool.unit_explain import explain_unit
from ..tool.routing import key_features, search_ladder
from ..tool.synergy import log_loss, no_harm_selection

_logger = logging.getLogger(__name__)


class _Router:
    """Trains, ranks and prunes one unit at a time with the run's settings."""

    def __init__(self, n_classes, accelerator, devices, cache_dir, n_dataloader_workers,
                 seed, mass_ladder, floor, relative_delta, ladder_choice, exp_sample_limit, explain_args):
        self.n_classes = n_classes
        self.accelerator = accelerator
        self.devices = devices
        self.cache_dir = cache_dir
        self.n_dataloader_workers = n_dataloader_workers
        self.seed = seed
        self.mass_ladder = mass_ladder
        self.floor = floor
        self.relative_delta = relative_delta
        self.ladder_choice = ladder_choice
        self.exp_sample_limit = exp_sample_limit
        self.explain_args = explain_args
        self.trainer_maker = Trainer_Maker()

    def train(self, unit, train, vali):
        """Fit ``unit`` on ``train``; every fit starts from the run's seed."""
        clf_object = unit.spike(accelerator=self.accelerator, available_devices=self.devices)
        config = copy.deepcopy(unit.config)
        config['default_root_dir'] = self.cache_dir
        config.setdefault('dataloader_workers', self.n_dataloader_workers)
        pl.seed_everything(self.seed, verbose=False)
        model, _ = self.trainer_maker.make(
            model_type=unit.type,
            clf_object=clf_object,
            device=unit.device,
            accelerator=unit.accelerator,
            train_data=train,
            vali_data=vali,
            n_classes=self.n_classes,
            fea_names=train.features,
            **config,
        )
        return model

    def explain(self, unit, model, corpus):
        """Attribution table of a trained unit on ``corpus``."""
        return explain_unit(
            model, unit.type, corpus,
            exp_sample_limit=self.exp_sample_limit,
            device='cuda' if unit.accelerator == 'cuda' else 'cpu',
            **self.explain_args,
        )

    def route(self, unit, train, vali, weights=None) -> tuple:
        """Pruning level of ``unit`` that ``vali`` chooses (see :func:`~ageas.search_ladder`).

        :param weights: Optional weight of each ``vali`` cell in the
            validation log-loss (shift-aware validation).
        :returns: ``(level, features, model, losses)``. ``level`` is
            ``None`` when the unit keeps all features; ``losses`` maps
            ``'full'`` and every level tried to its validation log-loss.
            With an empty ladder routing is off: the unit keeps all
            features and is not explained.
        """
        full = self.train(unit, train, vali)
        losses = {'full': log_loss(predict_probs(full, vali), vali.labels, weights)}
        if not self.mass_ladder:
            return None, list(train.features), full, losses
        attribution = self.explain(unit, full, train)
        versions = {}

        def score_at(level):
            features = key_features(attribution, level, self.floor)
            model = (
                full if len(features) == len(train.features)
                else self.train(unit, train.restrict(features), vali.restrict(features))
            )
            versions[level] = (features, model)
            losses[level] = log_loss(predict_probs(model, vali.restrict(features)), vali.labels, weights)
            return losses[level]

        level = search_ladder(self.mass_ladder, losses['full'], score_at, self.relative_delta, self.ladder_choice)
        if level is None:
            return None, list(train.features), full, losses
        features, model = versions[level]
        return level, features, model, losses

    def refit(self, unit, corpus, level) -> Expert:
        """Fit ``unit`` on every cell of ``corpus`` and narrow it to ``level``."""
        full = self.train(unit, corpus, corpus)
        if level is None:
            return Expert(unit, None, corpus.features, full)
        features = key_features(self.explain(unit, full, corpus), level, self.floor)
        if len(features) == len(corpus.features):
            return Expert(unit, level, features, full)
        narrow = corpus.restrict(features)
        return Expert(unit, level, features, self.train(unit, narrow, narrow))


def _provenance() -> dict:
    """Which ``ageas`` this run imported, and the git state of its checkout.

    ``ageas_commit`` and ``ageas_dirty`` are ``None`` when the package is not
    a git checkout, e.g. a site-packages install.
    """
    path = os.path.dirname(os.path.abspath(ageas.__file__))
    record = {'ageas_path': path, 'ageas_commit': None, 'ageas_dirty': None}

    def git(*args):
        return subprocess.run(
            ['git', '-c', 'safe.directory=*', '-C', path, *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    try:
        record['ageas_commit'] = git('rev-parse', 'HEAD')
        record['ageas_dirty'] = bool(git('status', '--porcelain'))
    except (OSError, subprocess.CalledProcessError):
        pass
    return record


def _median_level(levels: list, mass_ladder: list):
    """Lower median of per-fold levels along the ladder (``None`` = all features)."""
    ranks = sorted(0 if level is None else mass_ladder.index(level) + 1 for level in levels)
    rank = ranks[(len(ranks) - 1) // 2]
    return None if rank == 0 else mass_ladder[rank - 1]


def main(
    hangar: Hangar,
    query_dataset=None,
    cache_dir: str = None,
    accelerator: str = 'cpu',
    cuda_devices: list = None,
    using_model_types: list = None,
    using_model_list: list = None,
    kfold: int = 5,
    valid_fraction: float = 0.1,
    stratified_kfold_test: bool = False,
    stratified_kfold_valid: bool = False,
    n_dataloader_workers: int = 0,
    mass_ladder: list = None,
    floor: int = 20,
    relative_delta: float = 0.1,
    ladder_choice: str = 'sparsest',
    exp_sample_limit: int = 200,
    explain_args: dict = None,
    min_relative_gain: float = 0.01,
    max_steps: int = 50,
    competence_tolerance: float = 0.02,
    refit: str = 'competent',
    cell_weights=None,
    extend=None,
    seed: int = 42,
) -> MoE_Deck:
    """Route every unit to its key features and integrate the ones that help each other.

    :param hangar: Source hangar from which the candidate units are taken.
    :param query_dataset: Labelled terminal cells (``AnnData``-backed corpus).
    :param cache_dir: Folder for Lightning logs. ``None`` uses a fresh
        temporary folder; never point it inside Dropbox.
    :param accelerator: ``'cpu'`` or ``'cuda'``.
    :param cuda_devices: GPU indices; ``None`` uses every visible GPU.
    :param using_model_types: Whitelist of model types from the hangar.
    :param using_model_list: Whitelist of unit IDs from the hangar.
    :param kfold: Number of folds.
    :param valid_fraction: Fraction of each fold's training part that judges
        the pruning ladder. Must be positive.
    :param stratified_kfold_test: If ``True``, use class-stratified folds.
    :param stratified_kfold_valid: If ``True``, stratify the validation split.
    :param n_dataloader_workers: Dataloader workers for units whose config
        does not set ``dataloader_workers``.
    :param mass_ladder: Attribution-mass levels from weakest to strongest
        pruning. Defaults to ``[0.95, 0.8, 0.6, 0.4]``; an empty list turns
        routing off (every unit keeps all features).
    :param floor: Minimum number of positively attributed features kept per
        class.
    :param relative_delta: Largest validation log-loss increase accepted
        for a narrower version of a unit, as a fraction of the all-feature
        version's loss.
    :param ladder_choice: ``'sparsest'`` keeps the strongest level within
        ``relative_delta``; ``'best'`` keeps the version with the lowest
        validation loss the search reached, all features included (see
        :func:`~ageas.search_ladder`).
    :param exp_sample_limit: Cells per class explained by Integrated
        Gradients when ranking a neural network's features.
    :param explain_args: Extra keyword arguments for the attribution step
        (e.g. ``{'exp_step': 50}``).
    :param min_relative_gain: Smallest OOF log-loss improvement a unit must
        bring to be integrated, and to stay integrated, as a fraction of the
        ensemble's current loss.
    :param max_steps: Largest number of additions in the no-harm selection.
    :param competence_tolerance: Units whose out-of-fold balanced accuracy
        is within this of the best unit's are competent: candidates for the
        committee (:func:`~ageas.tool.committee.select_committee`). They are
        refitted like the integrated units.
    :param refit: ``'competent'`` refits the integrated and competent units;
        ``'all'`` refits every routed unit, so any committee can be drawn
        from the run's predictions afterwards.
    :param cell_weights: Weight of every terminal cell in each validation
        log-loss, for routing and integration alike, e.g. the shift-aware
        weights of :func:`~ageas.shift_weights`. ``None`` (ordinary
        validation) counts every cell alike.
    :param extend: Optional second round. Called once with ``unit_id ->
        out-of-fold log-loss`` (under ``cell_weights``) of the first round's
        units, it returns ``unit_id -> Unit`` to add; those are routed in
        the same folds and compete with the rest for integration.
    :param seed: Seed for the fold split and for every fit.
    :returns: The :class:`~ageas.MoE_Deck` of the run.
    """
    assert valid_fraction and valid_fraction > 0, \
        "The pruning ladder needs a validation split (valid_fraction > 0)."
    assert refit in ('competent', 'all'), "refit must be 'competent' or 'all'."
    if mass_ladder is None:
        mass_ladder = [0.95, 0.8, 0.6, 0.4]
    if explain_args is None:
        explain_args = {}
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix='ageas_moe_')

    devices = None
    if accelerator == 'cuda':
        devices = (
            list(cuda_devices) if cuda_devices is not None
            else list(range(torch.cuda.device_count()))
        )

    squad = hangar.sortie_generate(unit_types=using_model_types, unit_list=using_model_list)
    unsupported = [unit_id for unit_id, unit in squad.items() if unit.type not in SUPPORTED_TYPES]
    if unsupported:
        warn(f"Dropping units of unsupported types: {unsupported}")
        squad = {k: v for k, v in squad.items() if k not in unsupported}
    assert len(squad) > 0, "Not enough models in the squad."

    features = query_dataset.features
    label_dict = query_dataset.label_dict
    n_classes = len(label_dict)
    x_all, y_all = get_all_data(query_dataset)
    corpus = Feature_Corpus(x_all, y_all, features, label_dict, query_dataset.label_key)
    if cell_weights is not None:
        cell_weights = np.asarray(cell_weights, dtype=float)
        assert cell_weights.shape == (len(corpus),), "One weight per terminal cell is needed."

    router = _Router(
        n_classes=n_classes, accelerator=accelerator, devices=devices,
        cache_dir=cache_dir, n_dataloader_workers=n_dataloader_workers, seed=seed,
        mass_ladder=mass_ladder, floor=floor, relative_delta=relative_delta, ladder_choice=ladder_choice,
        exp_sample_limit=exp_sample_limit, explain_args=explain_args,
    )

    folds = kfold_indices(
        query_dataset,
        n_splits=kfold,
        valid_fraction=valid_fraction,
        stratified_test=stratified_kfold_test,
        stratified_valid=stratified_kfold_valid,
        random_seed=seed,
    )
    oof = {
        'probs': {}, 'labels': y_all.numpy(), 'folds': folds,
        'levels': {}, 'features': {}, 'losses': {}, 'cell_weights': cell_weights,
    }

    def route_all(units: dict) -> None:
        """Route ``units`` in every fold and add their out-of-fold record to ``oof``."""
        for unit_id in units:
            oof['probs'][unit_id] = np.full((len(corpus), n_classes), np.nan)
            oof['levels'][unit_id], oof['features'][unit_id], oof['losses'][unit_id] = [], [], []
        for k, fold in enumerate(folds):
            _logger.info("Fold %d / %d", k + 1, len(folds))
            train = corpus.subset(fold['train'])
            vali = corpus.subset(fold['valid'])
            test = corpus.subset(fold['test'])
            vali_weights = None if cell_weights is None else cell_weights[fold['valid']]
            for unit_id, unit in units.items():
                level, unit_features, model, losses = router.route(unit, train, vali, vali_weights)
                oof['probs'][unit_id][fold['test']] = predict_probs(model, test.restrict(unit_features))
                oof['levels'][unit_id].append(level)
                oof['features'][unit_id].append(unit_features)
                oof['losses'][unit_id].append(losses)
                _logger.info("\t%s: level %s, %d features", unit_id, level, len(unit_features))

    route_all(squad)
    if extend is not None:
        added = extend({u: log_loss(oof['probs'][u], oof['labels'], cell_weights) for u in squad})
        added = {u: unit for u, unit in (added or {}).items() if u not in squad}
        if added:
            _logger.info("Second round: %s", list(added))
            route_all(added)
            squad = {**squad, **added}

    weights = no_harm_selection(
        oof['probs'], oof['labels'], min_relative_gain=min_relative_gain, max_steps=max_steps,
        weights=cell_weights,
    )
    _logger.info("Integrated units: %s", weights)
    competent = competent_units(oof['probs'], oof['labels'], cell_weights, competence_tolerance)
    _logger.info("Competent units: %s", competent)

    chosen = list(weights) + [u for u in competent if u not in weights]
    if refit == 'all':
        chosen += [u for u in squad if u not in chosen]
    experts = {
        unit_id: router.refit(
            squad[unit_id], corpus, _median_level(oof['levels'][unit_id], mass_ladder),
        )
        for unit_id in chosen
    }
    return MoE_Deck(
        experts=experts, weights=weights, oof=oof, features=features,
        provenance=_provenance(), competent=competent,
    )
