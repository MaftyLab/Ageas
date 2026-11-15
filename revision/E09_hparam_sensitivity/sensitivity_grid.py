#!/usr/bin/env python3
"""E09 · Sensitivity of Ageas to the selection and masking parameters.

Selection grid (offline, from the E03 full-panel runs; no retraining):
rho (selection ratio) x tau_ret x tau_cut, plus relative-retention deltas.
For each setting and seed it re-selects configurations with the same gate
logic as ``n_kfold_selection`` and scores the resulting ensemble on the
progenitors. Reports the number of survivors and AUROC / macro F1.

Masking grid (``--masking``; retrains): alpha x N via ``n_iter_extraction``,
reporting top-k overlap of each setting's features with the protocol
default.

Outputs in ``revision/results/E09/<dataset>/``: ``selection_grid.csv`` and
``masking_grid.csv``.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import apply_selection_gates, prepare_dataset, results_dir  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.metrics import binary_scores, evaluation_frame  # noqa: E402
from revision.common.runner import _corpus, probs_frame  # noqa: E402
from revision.E03_ablation.ladder import load_panel  # noqa: E402


def selection_grid(data, panel_dir, metric):
    rows = []
    settings = [dict(selection_ratio=r, retention_point=t, cutoff_point=c, relative_retention=None)
                for r, t, c in itertools.product([0.25, 0.5, 0.75, 1.0], [0.8, 0.9, 0.95], [0.5, 0.6, 0.7])]
    settings += [dict(selection_ratio=0.0, retention_point=np.inf, cutoff_point=-np.inf,
                      relative_retention=d) for d in (0.01, 0.02, 0.05, 0.1)]
    for seed, (units, members, cells) in load_panel(panel_dir).items():
        oof = units[f'oof.{metric}']
        for setting in settings:
            keep = apply_selection_gates(oof, monitor_type='max', **setting)
            probs = np.mean([members[u] for u in keep], axis=0)
            frame = evaluation_frame(probs_frame(probs, cells, data.classes), data.ground_truth,
                                     data.classes[1], data.classes[0])
            scores = binary_scores(frame['y'], frame['p'])
            rows.append(dict(seed=seed, n_survivors=len(keep), **setting,
                             auroc=scores['auroc'], f1_macro=scores['f1_macro']))
    return pd.DataFrame(rows)


def masking_grid(data, protocol, config_dir, seed, accelerator, top_k):
    from pytorch_lightning import seed_everything
    from ageas import Hangar, n_iter_extraction

    corpus = _corpus(data.train, data.classes)
    results = {}
    for alpha, n_iter in itertools.product([1.5, 3, 10], [1, 2, 3]):
        seed_everything(seed, workers=True)
        _, final_exp = n_iter_extraction(
            hangar=Hangar(config_dir), query_dataset=corpus, accelerator=accelerator,
            max_extraction_iter=n_iter, outlier_iqr_factor=alpha, use_gene_names=False,
            seed=seed, selection_args=dict(protocol['selection']),
            explain_args=dict(protocol['debrief']),
        )
        results[(alpha, n_iter)] = {c: set(final_exp[c].sort_values(ascending=False).head(top_k).index)
                                    for c in final_exp.columns}
    default = results[(protocol['extraction']['outlier_iqr_factor'],
                       protocol['extraction']['max_extraction_iter'])]
    rows = []
    for (alpha, n_iter), tops in results.items():
        for c, feats in tops.items():
            rows.append({'alpha': alpha, 'N': n_iter, 'class': c,
                         'jaccard_vs_default': len(feats & default[c]) / len(feats | default[c])})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--masking', action='store_true', help='Also run the alpha x N grid (retrains).')
    parser.add_argument('--top-k', type=int, default=20)
    args = parser.parse_args(argv)
    registry, protocol, config_dir, seeds = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    out = results_dir('E09', args.dataset, args.results_root)
    metric = protocol['selection']['monitor_metric'].split('.', 1)[-1]

    if data.ground_truth is not None:
        grid = selection_grid(data, results_dir('E03', args.dataset, args.results_root), metric)
        grid.to_csv(os.path.join(out, 'selection_grid.csv'), index=False)
        keys = ['selection_ratio', 'retention_point', 'cutoff_point', 'relative_retention']
        print(grid.groupby(keys, dropna=False)[['n_survivors', 'auroc']].mean().to_string())
    if args.masking:
        table = masking_grid(data, protocol, config_dir, seeds[0], args.accelerator, args.top_k)
        table.to_csv(os.path.join(out, 'masking_grid.csv'), index=False)
        print(table.to_string(index=False))


if __name__ == '__main__':
    main()
