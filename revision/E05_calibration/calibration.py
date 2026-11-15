#!/usr/bin/env python3
"""E05 · Calibration and evaluation against continuous clonal fate.

For each method's predictions (Ageas seeds from E01 plus ``--method``
files) and a clone-matrix ground truth, reports:

* on unipotent clones: Brier score, ECE and a reliability table;
* on all informative progenitors: Spearman correlation between the
  predicted probability of the positive fate and the clonal fraction of
  that fate (no binarisation);
* on mixed clones only: mean absolute error between the predicted
  probability and the clonal fraction.

Outputs in ``revision/results/E05/<dataset>/``: ``calibration_summary.csv``
and ``reliability_<method>.csv``.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import load_predictions, prepare_dataset, results_dir  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.metrics import (  # noqa: E402
    evaluation_frame, expected_calibration_error, reliability_table,
)


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--method', action='append', default=[], help='NAME=PATH (repeatable).')
    parser.add_argument('--min-relatives', type=int, default=3,
                        help='Minimum fate-annotated relatives for the continuous analysis.')
    args = parser.parse_args(argv)
    registry, _, _, _ = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    gt = data.ground_truth
    assert gt is not None and 'n_total' in gt, 'E05 needs a clone-matrix ground truth.'
    negative, positive = data.classes[0], data.classes[1]
    out = results_dir('E05', args.dataset, args.results_root)

    methods = {}
    seed_files = [f for f in glob.glob(os.path.join(
        results_dir('E01', args.dataset, args.results_root), 'ageas_seed*.csv'))
        if not f.endswith('_units.csv')]
    if seed_files:
        preds = [load_predictions(f) for f in sorted(seed_files)]
        cols = [c for c in preds[0].columns if c.startswith('prob_')]
        methods['Ageas'] = sum(p[cols] for p in preds) / len(preds)
    for item in args.method:
        name, path = item.split('=', 1)
        methods[name] = load_predictions(path)

    informative = gt[gt['n_total'] >= args.min_relatives]
    rows = []
    for name, pred in methods.items():
        p_pos = pred[f'prob_{positive}'] / (pred[f'prob_{positive}'] + pred[f'prob_{negative}'])
        uni = evaluation_frame(pred, gt, positive, negative)
        reliability_table(uni['y'], uni['p']).to_csv(os.path.join(out, f'reliability_{name}.csv'), index=False)
        cont = informative.join(p_pos.rename('p'), how='inner')
        mixed = cont[cont['label'] == 'mixed']
        rows.append({
            'method': name,
            'n_unipotent': len(uni),
            'brier': float(np.mean((uni['p'] - uni['y']) ** 2)),
            'ece': expected_calibration_error(uni['y'], uni['p']),
            'n_continuous': len(cont),
            'spearman_vs_fraction': spearmanr(cont['p'], cont[f'frac_{positive}']).correlation,
            'n_mixed': len(mixed),
            'mae_mixed': float(np.mean(np.abs(mixed['p'] - mixed[f'frac_{positive}']))) if len(mixed) else np.nan,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out, 'calibration_summary.csv'), index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
