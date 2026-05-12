#!/usr/bin/env python3
"""
Multi-model operation pipelines.

Top-level Ageas workflows that orchestrate model selection and feature
extraction over an entire :class:`~ageas.Hangar`.

* :func:`n_kfold_selection` performs n-iteration k-fold cross-validation
  to retain the strongest models.
* :func:`n_iter_boost_selection` iteratively shrinks the feature set toward
  the most informative genes to boost downstream model performance.
* :func:`n_iter_extraction` iteratively extracts the top regulatory factors
  (genes) per class.

author: jy
"""
from .n_iter_extraction import main as n_iter_extraction
from .n_kfold_selection import main as n_kfold_selection
from .n_iter_boost_selection import main as n_iter_boost_selection

__all__ = [
    "n_kfold_selection",
    "n_iter_boost_selection",
    "n_iter_extraction",
]
