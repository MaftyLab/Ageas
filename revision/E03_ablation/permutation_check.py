#!/usr/bin/env python3
"""E03 · Gene-order permutation check for the recurrent family (R1.4).

Runs the frozen protocol on the original feature order and on several
random permutations of it (same seed), and compares the progenitor
predictions. Because every neural family starts with a dense projection
over all features and the recurrence runs along the modality axis, results
should agree up to initialisation noise. ``--rnn-only`` restricts the panel
to the recurrent configurations.

Writes ``permutation_check.csv``: per permutation, max/mean absolute
probability difference to the original order and both AUROCs (if ground
truth exists).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import predict_query, prepare_dataset, results_dir, run_ageas  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.metrics import binary_scores, evaluation_frame  # noqa: E402


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--n-perm', type=int, default=5)
    parser.add_argument('--rnn-only', action='store_true')
    args = parser.parse_args(argv)
    registry, protocol, config_dir, seeds = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    out = results_dir('E03', args.dataset, args.results_root)
    selection = dict(protocol['selection'])
    if args.rnn_only:
        selection['using_model_types'] = ['rnn']
    seed = seeds[0]
    rng = np.random.default_rng(seed)

    def run(order):
        deck, _ = run_ageas(data.train[:, order].copy(), data.classes, config_dir, seed,
                            selection, args.accelerator)
        probs, _ = predict_query(deck, data.query[:, order].copy(), data.classes)
        return probs

    base = run(np.arange(data.train.n_vars))
    col = f'prob_{data.classes[1]}'
    rows = []
    for k in range(args.n_perm):
        probs = run(rng.permutation(data.train.n_vars))
        diff = np.abs(probs[col].values - base[col].values)
        row = {'permutation': k, 'max_abs_diff': diff.max(), 'mean_abs_diff': diff.mean()}
        if data.ground_truth is not None:
            for name, p in (('original', base), ('permuted', probs)):
                f = evaluation_frame(p, data.ground_truth, data.classes[1], data.classes[0])
                row[f'auroc_{name}'] = binary_scores(f['y'], f['p'])['auroc']
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, 'permutation_check.csv'), index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
