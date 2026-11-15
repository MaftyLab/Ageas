#!/usr/bin/env python3
"""E07 · Stability of Ageas feature attributions under seeds and subsampling.

For each seed and each of ``--n-subsamples`` stratified subsamples of the
terminal cells (``--fraction``), runs the frozen protocol and debriefs the
deck. Then reports, per class:

* pairwise top-k Jaccard index and rank-biased overlap (RBO) between runs;
* per-feature selection frequency (share of runs with the feature in the
  top-k) — features at >= 0.8 are "stable";
* the same at the level of co-expression modules (features clustered by
  absolute correlation in terminal cells; module score = sum of |A_c|),
  which is identifiable even when single correlated genes are not;
* overlap of the stable features with Wilcoxon DE between terminal classes.

Outputs in ``revision/results/E07/<dataset>/``: ``runs/*.csv``,
``stability_summary.csv``, ``selection_frequency.csv``, ``modules.csv``.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import prepare_dataset, results_dir, run_ageas  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.registry import LABEL_COL, dense  # noqa: E402


def rbo(a: list, b: list, p: float = 0.9) -> float:
    """Rank-biased overlap (Webber et al. 2010), truncated at the list length."""
    depth = min(len(a), len(b))
    score = sum((len(set(a[:d]) & set(b[:d])) / d) * p ** (d - 1) for d in range(1, depth + 1))
    return (1 - p) * score


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan


def stratified_subsample(labels: pd.Series, fraction: float, rng) -> np.ndarray:
    keep = []
    for _, idx in labels.groupby(labels).groups.items():
        idx = np.asarray(idx)
        keep.extend(rng.choice(idx, max(1, int(round(fraction * len(idx)))), replace=False))
    return np.asarray(keep)


def modules(matrix: np.ndarray, names, threshold: float, candidates=None) -> pd.Series:
    """Co-expression modules: average-linkage clusters at 1 - |r| <= 1 - threshold.

    Only ``candidates`` (e.g. features ever ranked highly) are clustered, which
    keeps large peak matrices tractable; every other feature is its own module.
    """
    names = pd.Index(names)
    candidates = names if candidates is None else names.intersection(candidates)
    cols = names.get_indexer(candidates)
    corr = np.nan_to_num(np.corrcoef(matrix[:, cols], rowvar=False))
    dist = np.clip(1 - np.abs(corr), 0, None)
    np.fill_diagonal(dist, 0)
    clustered = fcluster(linkage(squareform(dist, checks=False), 'average'),
                         t=1 - threshold, criterion='distance')
    labels = pd.Series(np.arange(len(names)) + clustered.max() + 1, index=names, name='module')
    labels[candidates] = clustered
    return labels


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--n-subsamples', type=int, default=2)
    parser.add_argument('--fraction', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--module-threshold', type=float, default=0.6)
    args = parser.parse_args(argv)
    registry, protocol, config_dir, seeds = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    out = results_dir('E07', args.dataset, args.results_root)
    run_dir = os.path.join(out, 'runs')
    os.makedirs(run_dir, exist_ok=True)
    labels = data.train.obs[LABEL_COL].astype(str)

    runs = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for s in range(args.n_subsamples):
            idx = stratified_subsample(pd.Series(labels.values), args.fraction, rng)
            train = data.train[np.sort(idx)].copy()
            deck, corpus = run_ageas(train, data.classes, config_dir, seed,
                                     protocol['selection'], args.accelerator)
            exp = deck.debrief(exp_dataset=corpus, verbose=False, **protocol['debrief'])
            exp.columns = [data.classes[int(c.split('_')[1])] for c in exp.columns]
            exp.to_csv(os.path.join(run_dir, f'seed{seed}_sub{s}.csv'))
            runs[(seed, s)] = exp

    x = dense(data.train.X)
    candidates = {f for r in runs.values() for c in r.columns
                  for f in r[c].sort_values(ascending=False).head(5 * args.top_k).index}
    module_of = modules(x, data.train.var_names, args.module_threshold, sorted(candidates))
    module_of.to_csv(os.path.join(out, 'modules.csv'))
    y = labels.values

    summary, freq = [], {}
    for c in data.classes:
        tops = {k: r[c].sort_values(ascending=False).head(args.top_k).index.tolist() for k, r in runs.items()}
        mod_scores = {k: r[c].clip(lower=0).groupby(module_of).sum().sort_values(ascending=False)
                      for k, r in runs.items()}
        mod_tops = {k: m.head(max(5, args.top_k // 5)).index.tolist() for k, m in mod_scores.items()}
        pairs = list(itertools.combinations(runs, 2))
        counts = pd.Series([f for t in tops.values() for f in t]).value_counts() / len(runs)
        freq[c] = counts
        stable = counts[counts >= 0.8].index.tolist()

        others = y != c
        de_p = mannwhitneyu(x[y == c], x[others], axis=0, alternative='greater').pvalue
        de_top = pd.Series(de_p, index=data.train.var_names).sort_values().head(args.top_k).index
        summary.append({
            'class': c,
            'runs': len(runs),
            'gene_jaccard_mean': np.mean([jaccard(tops[a], tops[b]) for a, b in pairs]) if pairs else np.nan,
            'gene_rbo_mean': np.mean([rbo(tops[a], tops[b]) for a, b in pairs]) if pairs else np.nan,
            'module_jaccard_mean': np.mean([jaccard(mod_tops[a], mod_tops[b]) for a, b in pairs]) if pairs else np.nan,
            'n_stable_features': len(stable),
            'stable_in_DE_top_k': len(set(stable) & set(de_top)),
            'jaccard_stable_vs_DE': jaccard(stable, de_top),
        })

    pd.DataFrame(summary).to_csv(os.path.join(out, 'stability_summary.csv'), index=False)
    pd.DataFrame(freq).fillna(0).to_csv(os.path.join(out, 'selection_frequency.csv'))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
