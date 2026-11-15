#!/usr/bin/env python3
"""E03 · Train every configuration of the panel once per seed (no selection).

All gates are opened (``selection_ratio=1``, thresholds below any score,
``final_gate='none'``) so each configuration is cross-validated and refit
on all terminal cells. Ablation variants (E03 ladder), the transfer-gap
analysis (E04) and the sensitivity grid (E09) are then computed offline from
the saved per-unit outputs, without retraining.

Outputs in ``revision/results/E03/<dataset>/``:

* ``panel_seed<k>_units.csv`` — unit type and out-of-fold metrics;
* ``panel_seed<k>_members.npz`` — per-unit progenitor probabilities;
* ``panel_seed<k>_explain/<unit>.csv`` — per-unit explanations on terminal
  cells (``--explain``; used by E04).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import predict_query, prepare_dataset, results_dir, run_ageas  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.runner import unit_table  # noqa: E402

OPEN_GATES = dict(selection_ratio=1.0, retention_point=-np.inf, cutoff_point=-np.inf,
                  relative_retention=None, final_gate='none')


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--explain', action='store_true',
                        help='Also save each unit\'s explanation on terminal cells (for E04).')
    args = parser.parse_args(argv)
    registry, protocol, config_dir, seeds = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    out = results_dir('E03', args.dataset, args.results_root)
    selection = dict(protocol['selection'], **OPEN_GATES)

    for seed in seeds:
        deck, corpus = run_ageas(data.train, data.classes, config_dir, seed, selection, args.accelerator)
        _, members = predict_query(deck, data.query, data.classes)
        unit_table(deck).to_csv(os.path.join(out, f'panel_seed{seed}_units.csv'))
        np.savez_compressed(os.path.join(out, f'panel_seed{seed}_members.npz'),
                            cells=np.asarray(data.query.obs_names), **members)
        if args.explain:
            exp_dir = os.path.join(out, f'panel_seed{seed}_explain')
            os.makedirs(exp_dir, exist_ok=True)
            for unit_id, unit in deck.squad.items():
                table = unit.model.explain(dataset=corpus, device=unit.accelerator,
                                           n_dataloader_workers=0)
                table.to_csv(os.path.join(exp_dir, f'{unit_id}.csv'))
        print(f'seed {seed}: trained {len(deck.squad)} configurations')


if __name__ == '__main__':
    main()
