#!/usr/bin/env python3
"""E12 · Per-class metrics, confusion matrices and gains vs the best baseline.

Reads every ``predictions_*.csv`` in ``revision/results/E12/<dataset>/``
(columns ``cell, true, pred, fold, method``; Ageas from
``run_ageas_cv.py``, baselines converted from the classification-benchmark
outputs of ``benchmark/clf``) and writes:

* ``per_class.csv`` — precision, recall and F1 per class and method
  (mean and s.d. over folds) with class abundance;
* ``confusion_<method>.csv`` — row-normalised confusion matrix;
* ``delta_vs_best_baseline.csv`` — per-class F1 of Ageas minus the best
  baseline (by macro F1), next to class abundance.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import results_dir  # noqa: E402
from revision.common.cli import base_parser  # noqa: E402


def main(argv=None):
    args = base_parser(__doc__).parse_args(argv)
    out = results_dir('E12', args.dataset, args.results_root)
    preds = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(out, 'predictions_*.csv'))])
    classes = sorted(preds['true'].unique())
    abundance = preds.drop_duplicates('cell')['true'].value_counts(normalize=True)

    rows, macro = [], {}
    for method, frame in preds.groupby('method'):
        per_fold = []
        for fold, f in frame.groupby('fold'):
            p, r, f1, _ = precision_recall_fscore_support(f['true'], f['pred'], labels=classes,
                                                          zero_division=0)
            per_fold.append(pd.DataFrame({'class': classes, 'precision': p, 'recall': r, 'f1': f1,
                                          'fold': fold}))
        per_fold = pd.concat(per_fold)
        agg = per_fold.groupby('class')[['precision', 'recall', 'f1']].agg(['mean', 'std'])
        agg.columns = ['_'.join(c) for c in agg.columns]
        agg['method'] = method
        agg['abundance'] = abundance.reindex(agg.index).values
        rows.append(agg.reset_index())
        macro[method] = f1_score(frame['true'], frame['pred'], average='macro')
        cm = confusion_matrix(frame['true'], frame['pred'], labels=classes, normalize='true')
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(out, f'confusion_{method}.csv'))

    table = pd.concat(rows)
    table.to_csv(os.path.join(out, 'per_class.csv'), index=False)
    baselines = {m: v for m, v in macro.items() if m != 'Ageas'}
    if 'Ageas' in macro and baselines:
        best = max(baselines, key=baselines.get)
        a = table[table['method'] == 'Ageas'].set_index('class')['f1_mean']
        b = table[table['method'] == best].set_index('class')['f1_mean']
        delta = pd.DataFrame({'ageas_f1': a, f'{best}_f1': b, 'delta_f1': a - b,
                              'abundance': abundance.reindex(a.index)})
        delta.sort_values('abundance').to_csv(os.path.join(out, 'delta_vs_best_baseline.csv'))
        print(f'best baseline: {best}; Spearman(delta, log abundance) = '
              f'{delta["delta_f1"].corr(np.log(delta["abundance"]), method="spearman"):.3f}')
    print(pd.Series(macro, name='macro_F1').to_string())


if __name__ == '__main__':
    main()
