#!/usr/bin/env python3
"""The Ageas v2 pipeline.

Profile a dataset without progenitor labels, design its panel from that
profile, weight the terminal cells toward the progenitors, route every unit
to its key features under that validation, extend a family's range one step
where validation prefers its edge, integrate the units two ways (the no-harm
mixture, and a committee of competent, distinct units that vote), then
check the panel strategy against validation. When the data holds unlabeled
cells between the progenitors and the terminal states, a last stage walks
through them and fuses the walk into the vote (:mod:`ageas.tool.graph`).
Progenitor features are used; progenitor labels never are.
"""
import os
from dataclasses import dataclass

import numpy as np
import torch

from ageas.hangar import Hangar
from ageas.tool.corpus_loader import get_all_data

from .tool.committee import select_committee, vote
from .tool.feature_corpus import Feature_Corpus
from .moe_deck import MoE_Deck
from .tool.graph import graph_stage
from .panel import design_panel, extend_panel, write_panel
from .tool.profile import profile_dataset
from .ops.moe_selection import main as moe_selection
from .tool.shift import shift_weights
from .tool.strategy import check_strategy


@dataclass
class PipelineResult:
    """Everything one pipeline run produced.

    :ivar deck: The integrated experts.
    :ivar profile: The dataset's label-free profile.
    :ivar panel: The :class:`~ageas.UnitSpec` list designed from it,
        followed by any units the second round added.
    :ivar report: :func:`~ageas.check_strategy` under ordinary and
        shift-aware validation.
    :ivar cell_weights: Shift-aware weight of every terminal cell (computed
        whichever validation steered the run).
    :ivar committee: The voting members, best first.
    :ivar query: The progenitors' predictions: ``members`` (every refitted
        expert), ``no_harm`` (the no-harm mixture), ``vote`` (the committee's
        average, joined by the mixture when ``with_mixture`` is set), per
        cell ``agreement`` (share of members backing that call) and
        ``disagreement`` (bits), ``walk`` (the graph stage's absorption
        probabilities, or ``None``) and ``final`` (the pipeline's output: the
        graph stage's fusion when it ran, the vote otherwise).
    :ivar graph: The graph stage's record: ``active``, ``reason``,
        ``n_context``, ``n_query``, ``share`` and ``weight``.
    """
    deck: MoE_Deck
    profile: dict
    panel: list
    report: dict
    cell_weights: np.ndarray
    committee: list
    query: dict
    graph: dict = None


def run_pipeline(
    terminal_dataset,
    query_x,
    workdir: str,
    selection: dict = None,
    routing: dict = None,
    synergy: dict = None,
    committee: dict = None,
    validation: str = 'shift',
    refine: bool = True,
    rules: bool = True,
    accelerator: str = 'cpu',
    cuda_devices: list = None,
    unit_types: list = None,
    seed: int = 0,
    graph: dict = None,
    graph_settings: dict = None,
) -> PipelineResult:
    """Run Ageas v2 on one dataset.

    :param terminal_dataset: Labelled terminal cells (``AnnData``-backed
        corpus, as :func:`~ageas.moe_selection` takes it).
    :param query_x: Progenitor cells' model input, ``(n_cells, ...,
        n_features)`` in the terminal cells' feature order. No labels.
    :param workdir: Folder for the generated panel (``panel/``) and the
        Lightning logs (``cache/``); never inside Dropbox.
    :param selection: Fold settings for :func:`~ageas.moe_selection`
        (``kfold``, ``valid_fraction``, ...).
    :param routing: Routing settings (``mass_ladder``, ``floor``, ...).
    :param synergy: Integration settings (``min_relative_gain``, ...).
    :param committee: Committee settings: ``tolerance`` (competence, see
        :func:`~ageas.tool.committee.competent_units`), ``max_jsd`` and
        ``min_majority`` (see :func:`~ageas.tool.committee.select_committee`),
        ``rule`` (``'mean'`` or ``'logit'``, see :func:`~ageas.tool.committee.vote`),
        ``balance`` (``'unit'``: every member one vote; ``'type'``: every
        model type one vote), ``with_mixture`` (if ``True``, the no-harm
        mixture counts as much as the whole committee in the vote) and
        ``refit`` (``'competent'`` or ``'all'``, see
        :func:`~ageas.moe_selection`).
    :param validation: ``'shift'`` steers routing and integration by the
        shift-aware weights; ``'ordinary'`` counts every terminal cell alike.
        The strategy check reports both either way.
    :param refine: If ``True``, a second round adds one unit past the edge
        of every family whose best-validated unit sits at that edge
        (:func:`~ageas.extend_panel`).
    :param rules: If ``False``, the profile's rules do not shape the panel
        (:func:`~ageas.design_panel` with ``use_rules=False``).
    :param accelerator: ``'cpu'`` or ``'cuda'``.
    :param cuda_devices: GPU indices.
    :param unit_types: Keep only these unit types of the designed panel
        (for quick checks; benchmark runs use the whole panel).
    :param seed: Seed for the profile, the weights, the folds and every fit.
    :param graph: Every cell for the graph stage, one preprocessing for all:
        ``features`` (terminal, query and unlabeled cells), ``labels``
        (terminal class index; ``-1`` for query and unlabeled cells),
        ``query_index`` (each query cell's row, in ``query_x`` order) and
        ``kind`` (``'pca'`` or ``'lsi'``, see :func:`~ageas.tool.graph.embed`).
        ``None``: no stage, and the vote is the output.
    :param graph_settings: The stage's settings (protocol ``graph``, see
        :func:`~ageas.tool.graph.graph_stage`).
    """
    assert validation in ('shift', 'ordinary'), "validation must be 'shift' or 'ordinary'."
    assert graph is None or graph_settings is not None, 'the graph stage needs its settings.'
    committee = dict(committee or {})
    x, y = get_all_data(terminal_dataset)
    x, labels = x.numpy(), y.numpy()

    profile = profile_dataset(x, labels, query_x, seed=seed)
    panel = design_panel(profile, use_rules=rules)
    if unit_types is not None:
        panel = [spec for spec in panel if spec.type in unit_types]
    hangar = Hangar(write_panel(panel, os.path.join(workdir, 'panel')))
    weights = shift_weights(x, query_x, seed=seed)
    added = []

    def extend(losses):
        added.extend(extend_panel(panel, losses))
        if not added:
            return {}
        extra = Hangar(write_panel(added, os.path.join(workdir, 'panel_round2')))
        return {spec.unit_id: extra.units[spec.unit_id] for spec in added}

    deck = moe_selection(
        hangar=hangar, query_dataset=terminal_dataset, cache_dir=os.path.join(workdir, 'cache'),
        accelerator=accelerator, cuda_devices=cuda_devices, seed=seed,
        cell_weights=weights if validation == 'shift' else None, extend=extend if refine else None,
        competence_tolerance=committee.get('tolerance', 0.02), refit=committee.get('refit', 'competent'),
        **(selection or {}), **(routing or {}), **(synergy or {}),
    )
    panel = panel + added
    report = check_strategy(panel, deck.oof['probs'], deck.oof['labels'], cell_weights=weights)

    unlabelled = Feature_Corpus(
        torch.as_tensor(query_x), torch.zeros(len(query_x), dtype=torch.long),
        terminal_dataset.features, terminal_dataset.label_dict,
    )
    no_harm, _, members = deck.predict(query_dataset=unlabelled, batch_size=512, num_workers=0, return_members=True)
    candidates = {u: members[u] for u in deck.competent}
    chosen = select_committee(
        candidates, deck.competent,
        max_jsd=committee.get('max_jsd', 0.01), min_majority=committee.get('min_majority', 0.5),
    )
    assert committee.get('balance', 'unit') in ('unit', 'type'), "committee balance must be 'unit' or 'type'."
    groups = {u: deck.experts[u].unit.type for u in chosen} if committee.get('balance') == 'type' else None
    voted = vote({u: members[u] for u in chosen}, rule=committee.get('rule', 'mean'), groups=groups,
                 mixture=no_harm if committee.get('with_mixture', False) else None)
    if graph is None:
        stage = {'probs': voted['probs'], 'active': False, 'reason': 'no unlabeled cells given',
                 'n_context': 0, 'n_query': len(query_x), 'walk': None, 'share': None, 'weight': None}
    else:
        stage = graph_stage(voted['probs'], graph['features'], graph['labels'], graph['query_index'],
                            graph_settings, kind=graph.get('kind', 'pca'), seed=seed)
    query = {'members': members, 'no_harm': no_harm, 'vote': voted['probs'],
             'agreement': voted['agreement'], 'disagreement': voted['disagreement'],
             'walk': stage['walk'], 'final': stage['probs']}
    record = {k: v for k, v in stage.items() if k not in ('probs', 'walk')}
    return PipelineResult(deck=deck, profile=profile, panel=panel, report=report, cell_weights=weights,
                          committee=chosen, query=query, graph=record)
