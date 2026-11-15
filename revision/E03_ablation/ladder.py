#!/usr/bin/env python3
"""E03 · Ablation ladder computed offline from the full-panel runs.

For each seed, builds the ensemble variants below from the per-unit
progenitor probabilities and out-of-fold (OOF) scores saved by
``train_full_panel.py``, and scores them against the shared ground truth.
Every choice uses terminal-cell OOF scores only, never progenitor labels.

Variants:

* ``best_<family>`` — best configuration of each model family by OOF score;
* ``best_single`` — best configuration overall;
* ``classical_only`` / ``nn_only`` — uniform mean of all configurations of
  the classical or neural families;
* ``all_configs`` — uniform mean of every configuration (no selection);
* ``ageas_protocol`` — configurations passing the frozen protocol gates,
  uniform mean (what Ageas does);
* ``ageas_protocol_weighted`` — same survivors, OOF-weighted mean;
* ``l2_logreg`` — E02 baseline, if present.

Writes ``ladder.csv`` (mean ± s.d. over seeds, plus pooled-seed bootstrap
CIs) and ``survival.csv`` (fraction of seeds in which each configuration
passes the protocol gates — the concrete "data-adaptive" evidence).
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import apply_selection_gates, load_predictions, prepare_dataset, results_dir  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.metrics import binary_scores, bootstrap_ci, evaluation_frame  # noqa: E402
from revision.common.runner import probs_frame  # noqa: E402

NN_TYPES = {'mlp', 'rnn', 'resnet'}


def load_panel(out: str) -> dict:
    """Map seed -> (units table, members dict, cells)."""
    panels = {}
    for path in glob.glob(os.path.join(out, 'panel_seed*_units.csv')):
        seed = int(re.search(r'panel_seed(\d+)_units', path).group(1))
        units = pd.read_csv(path, index_col=0)
        npz = np.load(path.replace('_units.csv', '_members.npz'), allow_pickle=True)
        cells = npz['cells']
        members = {k: npz[k] for k in npz.files if k != 'cells'}
        panels[seed] = (units, members, cells)
    return dict(sorted(panels.items()))


def variants(units: pd.DataFrame, members: dict, metric: str, gates: dict) -> dict:
    """Ensemble probability matrices for every ladder variant."""
    oof = units[f'oof.{metric}']
    mean = lambda ids: np.mean([members[i] for i in ids], axis=0)  # noqa: E731
    out = {}
    for family, rows in units.groupby('type'):
        out[f'best_{family}'] = members[rows[f'oof.{metric}'].idxmax()]
    out['best_single'] = members[oof.idxmax()]
    classical = units.index[~units['type'].isin(NN_TYPES)]
    nn = units.index[units['type'].isin(NN_TYPES)]
    if len(classical):
        out['classical_only'] = mean(classical)
    if len(nn):
        out['nn_only'] = mean(nn)
    out['all_configs'] = mean(units.index)

    survivors = apply_selection_gates(oof, **gates)
    out['ageas_protocol'] = mean(survivors)
    w = oof[survivors] / oof[survivors].sum()
    out['ageas_protocol_weighted'] = np.sum([w[i] * members[i] for i in survivors], axis=0)
    return out, survivors


def main(argv=None):
    parser = base_parser(__doc__)
    args = parser.parse_args(argv)
    registry, protocol, _, _ = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    assert data.ground_truth is not None, 'The ladder needs lineage ground truth.'
    out = results_dir('E03', args.dataset, args.results_root)
    negative, positive = data.classes[0], data.classes[1]
    sel = protocol['selection']
    metric = sel['monitor_metric'].split('.', 1)[-1]
    gates = {k: sel[k] for k in ('selection_ratio', 'retention_point', 'cutoff_point',
                                 'relative_retention', 'monitor_type')}

    rows, survival, pooled = [], [], {}
    for seed, (units, members, cells) in load_panel(out).items():
        built, survivors = variants(units, members, metric, gates)
        e02 = os.path.join(results_dir('E02', args.dataset, args.results_root), 'l2_logreg.csv')
        for name, probs in built.items():
            frame = evaluation_frame(probs_frame(probs, cells, data.classes),
                                     data.ground_truth, positive, negative)
            rows.append(dict(seed=seed, variant=name, **binary_scores(frame['y'], frame['p'])))
            pooled.setdefault(name, []).append(frame)
        if os.path.exists(e02):
            frame = evaluation_frame(load_predictions(e02), data.ground_truth, positive, negative)
            rows.append(dict(seed=seed, variant='l2_logreg', **binary_scores(frame['y'], frame['p'])))
        survival.append(pd.Series(1.0, index=survivors, name=seed))

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(out, 'ladder_per_seed.csv'), index=False)
    summary = per_seed.groupby('variant')[['accuracy', 'f1_macro', 'auroc']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    for name, frames in pooled.items():
        frame = pd.concat(frames)
        _, low, high = bootstrap_ci(frame['y'], frame['p'], 'auroc', protocol.get('bootstrap', 1000))
        summary.loc[name, ['auroc_low', 'auroc_high']] = low, high
    summary.sort_values('auroc_mean').to_csv(os.path.join(out, 'ladder.csv'))
    pd.concat(survival, axis=1).fillna(0.0).mean(axis=1).rename('survival_rate') \
        .sort_values(ascending=False).to_csv(os.path.join(out, 'survival.csv'))
    print(summary.sort_values('auroc_mean').to_string())


if __name__ == '__main__':
    main()
