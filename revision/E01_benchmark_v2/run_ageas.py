#!/usr/bin/env python3
"""E01 · Run Ageas over all protocol seeds and store progenitor predictions.

Outputs in ``revision/results/E01/<dataset>/``:

* ``ageas_seed<k>.csv`` — ensemble probabilities per query cell;
* ``ageas_seed<k>_members.npz`` — per-unit probabilities (for E06);
* ``ageas_seed<k>_units.csv`` — surviving units and their out-of-fold scores.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import predict_query, prepare_dataset, results_dir, run_ageas, save_predictions  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.runner import unit_table  # noqa: E402


def main(argv=None):
    parser = base_parser(__doc__)
    args = parser.parse_args(argv)
    registry, protocol, config_dir, seeds = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    out = results_dir('E01', args.dataset, args.results_root)
    for seed in seeds:
        deck, _ = run_ageas(data.train, data.classes, config_dir, seed,
                            protocol['selection'], args.accelerator)
        probs, members = predict_query(deck, data.query, data.classes)
        save_predictions(probs, os.path.join(out, f'ageas_seed{seed}.csv'), 'Ageas', seed)
        np.savez_compressed(os.path.join(out, f'ageas_seed{seed}_members.npz'),
                            cells=np.asarray(data.query.obs_names), **members)
        unit_table(deck).to_csv(os.path.join(out, f'ageas_seed{seed}_units.csv'))
        print(f'seed {seed}: {len(deck.squad)} units kept')


if __name__ == '__main__':
    main()
