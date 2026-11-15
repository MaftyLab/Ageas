#!/usr/bin/env python3
"""Score Ageas v2 against Ageas (R1) on the simulated datasets.

The simulated datasets (``data/simulation``, scMultiSim) were never used to
build or refine v2, so this is a clean test of the final protocol. Every
dataset and modality is one setup with one run per method (seed 0): ``r1``,
``v2`` (the no-harm mixture) and ``v2_vote`` (the committee vote that the
mixture joins, v2's final output). Metrics are the R1 revision's fate metrics
(``binary_scores``): accuracy and macro F1 at 0.5 and AUROC, with ``progeny_B`` (the 30% minority
fate) as the positive class; Brier, ECE and weighted F1 are kept in
``scores.csv``.

Comparisons are paired by dataset. Per scenario (modality x non-lineage
factors x difficulty, ten simulation seeds each): the mean gain over R1, the
seeds where v2 is ahead and behind, and a two-sided Wilcoxon signed-rank
test, Holm-corrected over the scenarios of each metric and method. Overall:
the same over every dataset of a modality, and over all of them. The
simulation's held-out seeds (6-10) are summarised again on their own. The
noise-free oracle AUROC (``info.json``; RNA only) is shown as a ceiling; no
method sees it.

Writes ``<bench>/scores.csv`` (every run) and ``<bench>/summary.md``.

usage: score_sim.py --bench <bench dir> [--data <simulation data dir>]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the Ageas checkout
sys.path.insert(0, REPO)

from sklearn.metrics import f1_score  # noqa: E402
from revision.common import load_predictions  # noqa: E402
from revision.common.metrics import binary_scores, evaluation_frame  # noqa: E402

METHODS = ('r1', 'v2', 'v2_vote')
TARGET = ('accuracy', 'f1_macro', 'auroc')
METRICS = TARGET + ('f1_weighted', 'brier', 'ece')
LABEL = {'accuracy': 'Accuracy', 'f1_macro': 'Macro F1', 'auroc': 'AUROC'}
SCENARIO = ['modality', 'non_lineage_factors', 'difficulty']
ORDER = {'modality': ['rna', 'atac'], 'difficulty': ['hard', 'medium', 'easy']}


def fate_scores(frame, truth, classes) -> dict:
    """Binary fate metrics against the lineage ground truth."""
    joined = evaluation_frame(frame, truth, classes[1], classes[0])
    assert len(joined), 'no prediction matched a lineage-traced cell'
    scores = binary_scores(joined['y'], joined['p'])
    scores['f1_weighted'] = f1_score(joined['y'], (joined['p'] >= 0.5).astype(int), average='weighted')
    return {m: float(scores[m]) for m in METRICS}


def score_all(bench, registry, data) -> pd.DataFrame:
    """One row per finished run: its scenario, cost and fate metrics."""
    rows = []
    for setup, spec in registry.items():
        folder = os.path.join(bench, 'results', setup)
        if not os.path.exists(os.path.join(folder, 'truth.csv')):
            continue
        truth = pd.read_csv(os.path.join(folder, 'truth.csv'), index_col=0)
        truth.index = truth.index.astype(str)
        sim = spec['simulation']
        with open(os.path.join(data, 'datasets', sim['dataset'], 'info.json'), encoding='utf-8') as fh:
            oracle = json.load(fh).get('oracle_noise_free_CV_AUROC') if sim['modality'] == 'rna' else None
        for method in METHODS:
            path = os.path.join(folder, f'{method}_seed0.csv')
            if not (os.path.exists(path) and os.path.exists(path[:-4] + '.json')):
                continue
            frame = load_predictions(path)
            frame.index = frame.index.astype(str)
            with open(path[:-4] + '.json', encoding='utf-8') as fh:
                record = json.load(fh)
            rows.append(dict(setup=setup, **sim, method=method, minutes=record['minutes'],
                             peak_gb=record['peak_rss_gb'], oracle_auroc=oracle,
                             **fate_scores(frame, truth, list(spec['terminal_classes']))))
    return pd.DataFrame(rows)


def holm(pvalues) -> np.ndarray:
    """Holm-adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    adjusted, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def compare(wide, method, metric) -> dict:
    """Paired comparison of ``method`` with R1 on one metric over the rows of ``wide``."""
    mine, r1 = wide[(metric, method)], wide[(metric, 'r1')]
    diff = (mine - r1).to_numpy()
    p = wilcoxon(diff).pvalue if len(diff) > 1 and np.any(diff != 0) else 1.0
    return {'n': len(diff), 'r1': r1.mean(), 'v2': mine.mean(), 'gain': diff.mean(),
            'ahead': int((diff > 0).sum()), 'behind': int((diff < 0).sum()), 'p': float(p)}


def scenario_table(wide, method) -> pd.DataFrame:
    """Per scenario and metric: the paired comparison, Holm-corrected over scenarios per metric."""
    rows = []
    for key, part in wide.groupby(level=SCENARIO, sort=False):
        for metric in TARGET:
            rows.append(dict(zip(SCENARIO, key), metric=metric, **compare(part, method, metric)))
    table = pd.DataFrame(rows)
    table['p_holm'] = np.nan
    for metric in TARGET:
        mask = table['metric'] == metric
        table.loc[mask, 'p_holm'] = holm(table.loc[mask, 'p'])
    return table


def sort_scenarios(frame) -> pd.DataFrame:
    keys = {c: frame[c].map({v: i for i, v in enumerate(ORDER[c])}) for c in ORDER if c in frame}
    order = pd.DataFrame(keys).assign(non_lineage_factors=frame['non_lineage_factors'])
    return frame.loc[order.sort_values(['modality', 'non_lineage_factors', 'difficulty'], kind='stable').index]


def fmt_cell(row) -> str:
    return (f"{row['r1']:.3f} → {row['v2']:.3f} ({row['gain']:+.3f}; {row['ahead']}/{row['behind']}; "
            f"p {row['p_holm']:.2g})")


def summary(scores, wide, bench) -> str:
    lines = ['# Ageas v2 against R1 on the simulated datasets', '',
             'A clean test: v2 was built and refined without these data. One run per method and dataset '
             '(seed 0). Gains are v2 minus R1, paired by dataset; "ahead/behind" counts datasets. '
             'Wilcoxon signed-rank, two-sided.', '']
    complete = wide.index.get_level_values('setup')
    lines += [f'Datasets with every method finished: {len(complete)} of {scores["setup"].nunique()} scored.', '']

    lines += ['## Overall', '', '| datasets | method | metric | n | R1 | v2 | gain | ahead / behind | p |',
              '|---|---|---|---|---|---|---|---|---|']
    modality, seed = wide.index.get_level_values('modality'), wide.index.get_level_values('seed')
    subsets = [('all', wide), ('RNA', wide[modality == 'rna']), ('ATAC', wide[modality == 'atac']),
               ('held-out seeds 6-10', wide[seed >= 6])]
    for name, part in subsets:
        if not len(part):
            continue
        for method in ('v2_vote', 'v2'):
            for metric in TARGET:
                c = compare(part, method, metric)
                lines.append(f"| {name} | {method} | {LABEL[metric]} | {c['n']} | {c['r1']:.4f} | {c['v2']:.4f} | "
                             f"{c['gain']:+.4f} | {c['ahead']} / {c['behind']} | {c['p']:.2g} |")
    lines.append('')

    verdict = []
    for method, title in (('v2_vote', 'v2 (committee vote with the mixture; final output)'),
                          ('v2', 'v2 no-harm mixture alone')):
        table = sort_scenarios(scenario_table(wide, method))
        lines += [f'## By scenario: {title}', '',
                  'Each cell: R1 → v2 (gain; datasets ahead/behind; Holm-corrected p).', '',
                  '| modality | non-lineage | difficulty | n | ' + ' | '.join(LABEL[m] for m in TARGET)
                  + ' | oracle AUROC |', '|---|---|---|---|' + '---|' * (len(TARGET) + 1)]
        oracle = scores[scores['method'] == 'r1'].groupby(SCENARIO)['oracle_auroc'].mean()
        better = 0
        for key, part in table.groupby(SCENARIO, sort=False):
            cells = {r['metric']: r for _, r in part.iterrows()}
            o = oracle.get(key, np.nan)
            lines.append(f'| {key[0].upper()} | {key[1]} | {key[2]} | {cells["accuracy"]["n"]} | '
                         + ' | '.join(fmt_cell(cells[m]) for m in TARGET)
                         + f" | {'—' if pd.isna(o) else f'{o:.3f}'} |")
            better += all(cells[m]['gain'] > 0 for m in TARGET)
        n_scen = table.groupby(SCENARIO).ngroups
        verdict.append(f'- {method}: mean ahead of R1 on all three metrics in {better} of {n_scen} scenarios.')
        lines.append('')

    lines += ['## Verdict', ''] + verdict + ['']

    lines += ['## Cost and v2 design', '', '| method | runs | median minutes | max peak GB |', '|---|---|---|---|']
    for method in ('r1', 'v2'):
        part = scores[scores['method'] == method]
        if len(part):
            lines.append(f"| {method} | {len(part)} | {part['minutes'].median():.1f} | {part['peak_gb'].max():.1f} |")
    rules, committees = collections.Counter(), []
    for setup in scores.loc[scores['method'] == 'v2', 'setup']:
        with open(os.path.join(bench, 'results', setup, 'v2_seed0.json'), encoding='utf-8') as fh:
            record = json.load(fh)
        rules.update(record.get('rules') or [])
        committees.append(len(record.get('committee') or []))
    if committees:
        lines += ['', f'v2 rules fired (of {len(committees)} runs): '
                  + ', '.join(f'{r} {n}' for r, n in rules.most_common())
                  + f'. Committee size: median {np.median(committees):.0f} (range {min(committees)}-{max(committees)}).']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bench', required=True, help='Benchmark folder (registry.json, results/).')
    parser.add_argument('--data', default='/home/jack/ageas_work/data/simulation')
    args = parser.parse_args(argv)

    with open(os.path.join(args.bench, 'registry.json'), encoding='utf-8') as fh:
        registry = {k: v for k, v in json.load(fh).items() if not k.startswith('_')}
    scores = score_all(args.bench, registry, args.data)
    scores.to_csv(os.path.join(args.bench, 'scores.csv'), index=False)
    wide = scores.pivot_table(index=['setup'] + SCENARIO + ['seed'], columns='method', values=list(TARGET))
    wide = wide[[c for c in wide.columns if c[1] in METHODS]].dropna()
    text = summary(scores, wide, args.bench)
    with open(os.path.join(args.bench, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(text)


if __name__ == '__main__':
    main()
