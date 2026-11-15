#!/usr/bin/env python3
"""E15 · What does lineage tracing reveal that Ageas does (not) recover?

"Oracle" differential analysis uses the lineage labels themselves: Wilcoxon
rank-sum between progenitors of each ground-truth fate (what a
lineage-tracing study would report). It is compared with:

* Ageas features — selection frequency from E07 if available, otherwise a
  fresh debrief under the frozen protocol;
* terminal-state DE — Wilcoxon between terminal classes, which defines what
  any transfer approach *can* see.

Reports precision/recall of Ageas top-k against oracle top-k, and lists
oracle features missed by Ageas with a flag for whether they are DE between
terminal states (the expected blind spot of transfer learning).

Outputs in ``revision/results/E15/<dataset>/``.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import prepare_dataset, results_dir, run_ageas  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.registry import LABEL_COL, dense  # noqa: E402


def bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def wilcoxon_table(x: np.ndarray, groups: np.ndarray, target: str, names) -> pd.DataFrame:
    a, b = x[groups == target], x[groups != target]
    p = mannwhitneyu(a, b, axis=0, alternative='two-sided').pvalue
    return pd.DataFrame({'log_fc': np.log1p(a.mean(axis=0)) - np.log1p(b.mean(axis=0)),
                         'p': p, 'q': bh(np.nan_to_num(p, nan=1.0))}, index=names)


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--top-k', type=int, default=50)
    args = parser.parse_args(argv)
    registry, protocol, config_dir, seeds = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    gt = data.ground_truth
    assert gt is not None, 'E15 needs lineage ground truth.'
    out = results_dir('E15', args.dataset, args.results_root)
    names = data.query.var_names

    labelled = gt[gt['label'].isin(data.classes)].index.intersection(data.query.obs_names)
    qx = dense(data.query[labelled].X)
    qy = gt.loc[labelled, 'label'].values
    tx = dense(data.train.X)
    ty = data.train.obs[LABEL_COL].astype(str).values

    freq_path = os.path.join(results_dir('E07', args.dataset, args.results_root), 'selection_frequency.csv')
    if os.path.exists(freq_path):
        ageas_rank = pd.read_csv(freq_path, index_col=0)
    else:
        deck, corpus = run_ageas(data.train, data.classes, config_dir, seeds[0],
                                 protocol['selection'], args.accelerator)
        exp = deck.debrief(exp_dataset=corpus, verbose=False, **protocol['debrief'])
        ageas_rank = exp.set_axis(data.classes, axis=1)

    rows, missed = [], []
    for c in data.classes:
        oracle = wilcoxon_table(qx, qy, c, names)
        terminal = wilcoxon_table(tx, ty, c, names)
        oracle_up = oracle[(oracle['q'] < 0.05) & (oracle['log_fc'] > 0)].sort_values('p')
        oracle_top = oracle_up.head(args.top_k).index
        ageas_top = ageas_rank[c].sort_values(ascending=False).head(args.top_k).index
        hits = oracle_top.intersection(ageas_top)
        rows.append({'class': c, 'n_oracle_significant': len(oracle_up),
                     'precision': len(hits) / len(ageas_top) if len(ageas_top) else np.nan,
                     'recall': len(hits) / len(oracle_top) if len(oracle_top) else np.nan})
        for feature in oracle_top.difference(ageas_top):
            missed.append({'class': c, 'feature': feature,
                           'oracle_log_fc': oracle.loc[feature, 'log_fc'],
                           'oracle_q': oracle.loc[feature, 'q'],
                           'terminal_DE_q': terminal.loc[feature, 'q'],
                           'terminal_DE_same_direction': bool(terminal.loc[feature, 'log_fc'] > 0
                                                              and terminal.loc[feature, 'q'] < 0.05)})
        oracle.to_csv(os.path.join(out, f'oracle_de_{c}.csv'))

    pd.DataFrame(rows).to_csv(os.path.join(out, 'concordance.csv'), index=False)
    missed = pd.DataFrame(missed)
    missed.to_csv(os.path.join(out, 'missed_by_ageas.csv'), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    if not missed.empty:
        print('missed oracle features that are NOT terminal-DE (blind spot):',
              int((~missed['terminal_DE_same_direction']).sum()), 'of', len(missed))


if __name__ == '__main__':
    main()
