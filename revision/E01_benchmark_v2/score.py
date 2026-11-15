#!/usr/bin/env python3
"""E01 · Score every method against the shared ground truth, with uncertainty.

Reads the Ageas seed files written by ``run_ageas.py`` plus any other
method's predictions (``--method NAME=PATH``; CSV indexed by cell with
``prob_<class>`` columns, e.g. from E02 or converted legacy CoSpar/WOT/
CellRank outputs). Writes to ``revision/results/E01/<dataset>/``:

* ``metrics.csv`` — per method: accuracy, macro F1, AUROC, Brier, ECE with
  95% bootstrap CIs (Ageas: seed-averaged probabilities) and, for Ageas,
  mean ± s.d. over seeds;
* ``comparisons.csv`` — Ageas vs each method: DeLong AUROC test and paired
  bootstrap tests for accuracy/F1, Holm-adjusted.

``--subset`` restricts scoring to a cell subset (e.g. the R1.1
ambiguous-by-similarity quartile from E02) and ``--tag`` names the outputs.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import load_predictions, prepare_dataset, results_dir  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.metrics import (  # noqa: E402
    binary_scores, bootstrap_ci, delong_test, evaluation_frame, holm, paired_bootstrap,
)

METRICS = ['accuracy', 'f1_macro', 'auroc', 'brier', 'ece']


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--method', action='append', default=[],
                        help='Additional method predictions as NAME=PATH (repeatable).')
    parser.add_argument('--positive', default=None, help='Positive fate (default: 2nd class).')
    parser.add_argument('--subset', default=None,
                        help='CSV indexed by cell; score only rows whose (last) boolean '
                             'column is True, e.g. E02 ambiguous_by_similarity.csv.')
    parser.add_argument('--tag', default='', help='Suffix for output file names.')
    args = parser.parse_args(argv)
    registry, protocol, _, _ = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    assert data.ground_truth is not None, 'E01 scoring needs lineage ground truth.'
    negative, positive = data.classes[0], data.classes[1]
    if args.positive:
        positive = args.positive
        negative = [c for c in data.classes if c != positive][0]
    out = results_dir('E01', args.dataset, args.results_root)
    n_boot = protocol.get('bootstrap', 1000)

    seed_files = sorted(glob.glob(os.path.join(out, 'ageas_seed*.csv')))
    seed_files = [f for f in seed_files if not f.endswith('_units.csv')]
    seed_preds = [load_predictions(f) for f in seed_files]
    methods = {}
    if seed_preds:
        prob_cols = [c for c in seed_preds[0].columns if c.startswith('prob_')]
        methods['Ageas'] = sum(p[prob_cols] for p in seed_preds) / len(seed_preds)
    for item in args.method:
        name, path = item.split('=', 1)
        methods[name] = load_predictions(path)

    frames = {m: evaluation_frame(p, data.ground_truth, positive, negative) for m, p in methods.items()}
    common = set.intersection(*(set(f.index) for f in frames.values()))
    if args.subset:
        subset = pd.read_csv(args.subset, index_col=0)
        common &= set(subset.index[subset.iloc[:, -1].astype(bool)])
    common = sorted(common)
    frames = {m: f.loc[common] for m, f in frames.items()}
    print(f'{len(common)} progenitors with unipotent ground truth scored by all methods')

    rows = []
    for name, frame in frames.items():
        row = {'method': name, 'n': len(frame)}
        for metric in METRICS:
            est, low, high = bootstrap_ci(frame['y'], frame['p'], metric, n_boot)
            row.update({metric: est, f'{metric}_low': low, f'{metric}_high': high})
        if name == 'Ageas' and len(seed_preds) > 1:
            per_seed = [binary_scores(*evaluation_frame(p, data.ground_truth, positive, negative)
                                      .loc[common, ['y', 'p']].T.values) for p in seed_preds]
            for metric in METRICS:
                vals = [s[metric] for s in per_seed]
                row[f'{metric}_seed_mean'] = float(np.mean(vals))
                row[f'{metric}_seed_sd'] = float(np.std(vals, ddof=1))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, f'metrics{args.tag}.csv'), index=False)

    comparisons = []
    if 'Ageas' in frames:
        ref = frames['Ageas']
        for name, frame in frames.items():
            if name == 'Ageas':
                continue
            d = delong_test(ref['y'], ref['p'], frame['p'])
            comparisons.append({'vs': name, 'metric': 'auroc', 'diff': d['diff'], 'p': d['p']})
            for metric in ('accuracy', 'f1_macro'):
                b = paired_bootstrap(ref['y'], ref['p'], frame['p'], metric, n_boot)
                comparisons.append({'vs': name, 'metric': metric, 'diff': b['diff'],
                                    'low': b['low'], 'high': b['high'], 'p': b['p']})
    if comparisons:
        table = pd.DataFrame(comparisons)
        table['p_holm'] = holm(table['p'])
        table.to_csv(os.path.join(out, f'comparisons{args.tag}.csv'), index=False)
    print(pd.DataFrame(rows)[['method', 'n'] + METRICS].to_string(index=False))


if __name__ == '__main__':
    main()
