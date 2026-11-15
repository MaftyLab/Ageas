#!/usr/bin/env python3
"""
Ageas

AutoML-based Genetic regulatory Element extrAction System.

Top-level package exposing the model classifiers (classical and neural network),
core orchestration objects (``Unit``, ``Deck``, ``Hangar``), and the multi-model
operation pipelines (``n_kfold_selection``, ``n_iter_boost_selection``,
``n_iter_extraction``).

It also exposes the feature-routed pipeline of Ageas v2 (``run_pipeline``).
The pipeline profiles the data without progenitor labels and designs a
panel of units from that profile. It narrows every unit to the features it
relies on as far as validation allows (``moe_selection``, driven by
``MoE_Deck``). It then integrates the units twice: as the no-harm mixture
of those that improve the ensemble together, and as the vote of a committee
of competent, distinct units. Where unlabeled intermediate cells are given,
a graph stage walks through them toward the terminal classes.

author: jy
"""
from .nn import *  # noqa: F401,F403  (the network classifiers and blocks named in nn.__all__)
from .nn import __all__ as nn_all
from .classical import *  # noqa: F401,F403  (the classical classifiers named in classical.__all__)
from .classical import __all__ as classical_all
from .ops import n_iter_boost_selection, n_kfold_selection, n_iter_extraction, moe_selection
from .unit import Unit
from .deck import Deck
from .hangar import Hangar
# Ageas v2: the feature-routed pipeline and its parts.
from .panel import FAMILIES, UnitSpec, design_panel, extend_panel, write_panel
from .moe_deck import MoE_Deck
from .pipeline import PipelineResult, run_pipeline
from .tool.committee import competent_units, select_committee, vote
from .tool.graph import fuse, graph_fate, graph_stage
from .tool.profile import profile_dataset
from .tool.routing import key_features, search_ladder
from .tool.shift import shift_weights
from .tool.strategy import check_strategy
from .tool.synergy import no_harm_selection
from .tool.unit_explain import explain_unit


__all__ = classical_all + nn_all + [
    'n_iter_boost_selection',
    'n_kfold_selection',
    'n_iter_extraction',
    'Unit',
    'Deck',
    'Hangar',
    # Ageas v2
    'run_pipeline',
    'PipelineResult',
    'moe_selection',
    'MoE_Deck',
    'profile_dataset',
    'shift_weights',
    'FAMILIES',
    'UnitSpec',
    'design_panel',
    'extend_panel',
    'write_panel',
    'explain_unit',
    'key_features',
    'search_ladder',
    'no_harm_selection',
    'competent_units',
    'select_committee',
    'vote',
    'check_strategy',
    'graph_fate',
    'graph_stage',
    'fuse',
]
