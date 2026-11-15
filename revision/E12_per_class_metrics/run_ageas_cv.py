#!/usr/bin/env python3
"""E12 · Nested 5-fold cross-validation of Ageas on a cell-type classification dataset.

Outer stratified 5-fold split over all annotated cells. Inside each outer
training fold Ageas runs its own k-fold selection (the frozen protocol), so
no outer test cell influences selection (the nesting R1.2 asks about).

Registry entry: ``terminal_classes`` = all cell types, ``query_values`` = [].
Writes ``revision/results/E12/<dataset>/predictions_ageas.csv`` with
columns ``cell, true, pred, fold, method``.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import predict_query, prepare_dataset, results_dir, run_ageas  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.registry import LABEL_COL  # noqa: E402


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args(argv)
    registry, protocol, config_dir, seeds = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    out = results_dir('E12', args.dataset, args.results_root)
    cells = data.train
    y = cells.obs[LABEL_COL].astype(str).values
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seeds[0])

    rows = []
    for fold, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y)):
        deck, _ = run_ageas(cells[tr].copy(), data.classes, config_dir, seeds[0],
                            protocol['selection'], args.accelerator)
        probs, _ = predict_query(deck, cells[te].copy(), data.classes)
        rows.append(pd.DataFrame({'cell': cells.obs_names[te], 'true': y[te],
                                  'pred': probs['pred'].values, 'fold': fold, 'method': 'Ageas'}))
        print(f'fold {fold}: accuracy {np.mean(rows[-1]["true"] == rows[-1]["pred"]):.3f}')
    pd.concat(rows).to_csv(os.path.join(out, 'predictions_ageas.csv'), index=False)


if __name__ == '__main__':
    main()
