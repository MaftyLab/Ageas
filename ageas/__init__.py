#!/usr/bin/env python3
"""
Ageas

AutoML-based Genetic regulatory Element extrAction System.

Top-level package exposing the model classifiers (classical and neural network),
core orchestration objects (``Unit``, ``Deck``, ``Hangar``), and the multi-model
operation pipelines (``n_kfold_selection``, ``n_iter_boost_selection``,
``n_iter_extraction``).

author: jy
"""
from .nn import __all__ as nn_all
from .classical import __all__ as classical_all
from .ops import n_iter_boost_selection, n_kfold_selection, n_iter_extraction
from .unit import Unit
from .deck import Deck
from .hangar import Hangar


__all__ = classical_all + nn_all + [
    'n_iter_boost_selection',
    'n_kfold_selection',
    'n_iter_extraction',
    'Unit',
    'Deck',
    'Hangar',
]
