#!/usr/bin/env python3
"""E04 · Does better terminal-state classification mean better fate transfer?

Uses the full-panel runs of E03. For every configuration and seed it pairs
the terminal-cell out-of-fold score (accuracy and macro F1) with the
progenitor fate-prediction AUROC, and reports the Spearman correlation per
dataset. With E03 ``--explain`` outputs it also measures how much of each
configuration's attribution falls on features that are barely detected in
progenitors ("terminal-only" features) and relates that to transfer.

Outputs in ``revision/results/E04/<dataset>/``: ``transfer_gap.csv`` (one
row per configuration × seed) and ``summary.csv`` (correlations).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import prepare_dataset, results_dir  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.metrics import binary_scores, evaluation_frame  # noqa: E402
from revision.common.registry import dense  # noqa: E402
from revision.common.runner import probs_frame  # noqa: E402
from revision.E03_ablation.ladder import load_panel  # noqa: E402


def terminal_only_mass(exp_path: str, detected: pd.Series, threshold: float) -> float:
    """Share of |attribution| on features detected in < ``threshold`` of progenitors."""
    table = pd.read_csv(exp_path, index_col=0)
    scores = table[[c for c in table.columns if c.endswith('_Scores')]].abs().sum(axis=1)
    rare = detected.reindex(scores.index).fillna(0) < threshold
    total = scores.sum()
    return float(scores[rare].sum() / total) if total > 0 else np.nan


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--detection-threshold', type=float, default=0.05,
                        help='A feature is "terminal-only" if detected in fewer progenitors.')
    args = parser.parse_args(argv)
    registry, _, _, _ = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    assert data.ground_truth is not None, 'E04 needs lineage ground truth.'
    panel_dir = results_dir('E03', args.dataset, args.results_root)
    out = results_dir('E04', args.dataset, args.results_root)
    negative, positive = data.classes[0], data.classes[1]
    detected = pd.Series((dense(data.query.X) > 0).mean(axis=0), index=data.query.var_names)

    rows = []
    for seed, (units, members, cells) in load_panel(panel_dir).items():
        for unit_id, unit in units.iterrows():
            frame = evaluation_frame(probs_frame(members[unit_id], cells, data.classes),
                                     data.ground_truth, positive, negative)
            row = dict(seed=seed, unit_id=unit_id, type=unit['type'],
                       oof_accuracy=unit.get('oof.accuracy'), oof_f1=unit.get('oof.f1'),
                       progenitor_auroc=binary_scores(frame['y'], frame['p'])['auroc'])
            exp_path = os.path.join(panel_dir, f'panel_seed{seed}_explain', f'{unit_id}.csv')
            if os.path.exists(exp_path):
                row['terminal_only_mass'] = terminal_only_mass(exp_path, detected,
                                                               args.detection_threshold)
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, 'transfer_gap.csv'), index=False)
    summary = []
    for x in ('oof_accuracy', 'oof_f1', 'terminal_only_mass'):
        if x in table and table[x].notna().sum() > 2:
            rho, p = spearmanr(table[x], table['progenitor_auroc'], nan_policy='omit')
            summary.append({'x': x, 'y': 'progenitor_auroc', 'spearman': rho, 'p': p,
                            'n': int(table[x].notna().sum())})
    pd.DataFrame(summary).to_csv(os.path.join(out, 'summary.csv'), index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
