#!/usr/bin/env python3
"""
Neural network classifiers for Ageas.

Re-exports the residual building blocks defined in
:mod:`ageas.nn.blocks` together with the high-level Lightning classifiers:

* :class:`~ageas.nn.NN_Classifier` — modular MLP/RNN classifier driven by
  the building blocks.
* :class:`~ageas.nn.Mixer_Classifier` — ResNet-style mixer classifier built
  on top of :class:`NN_Classifier`.

author: jy
"""
from .blocks import __all__ as blocks_all
from .classifier import NN_Classifier, Mixer_Classifier


__all__ = blocks_all + [
    'NN_Classifier',
    'Mixer_Classifier'
]
