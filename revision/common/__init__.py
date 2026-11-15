"""Shared utilities for the Cell Systems R1 revision experiments.

Importing this package puts the repository root and ``benchmark/fate`` on
``sys.path`` so experiment scripts can be run directly, e.g.
``python revision/E01_benchmark_v2/run_ageas.py --dataset larry``.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REVISION_ROOT = os.path.join(REPO_ROOT, 'revision')
for _path in (REPO_ROOT, os.path.join(REPO_ROOT, 'benchmark', 'fate')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from .registry import Prepared, load_protocol, load_registry, prepare_dataset  # noqa: E402
from .runner import (  # noqa: E402
    apply_selection_gates,
    load_predictions,
    predict_query,
    results_dir,
    run_ageas,
    save_predictions,
)

__all__ = [
    'REPO_ROOT',
    'REVISION_ROOT',
    'Prepared',
    'load_protocol',
    'load_registry',
    'prepare_dataset',
    'apply_selection_gates',
    'load_predictions',
    'predict_query',
    'results_dir',
    'run_ageas',
    'save_predictions',
]
