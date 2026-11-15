#!/usr/bin/env python3
"""E17 · Compare each configuration, and Ageas ensembles built from them, with the manuscript.

Reads the outputs of ``run_configs.py`` and writes to
``<results-root>/E17/<dataset>/``:

* ``configs.csv`` — per configuration: query metrics (mean and s.d. over
  seeds), out-of-fold terminal accuracy, source panels and files, and whether
  it is no worse than the manuscript on every metric (on the seed mean, and
  in how many seeds);
* ``ensembles.csv`` — ensembles rebuilt offline for every panel and seed:
  ``protocol`` keeps the units that pass the frozen protocol's gates on
  out-of-fold terminal scores (what Ageas does); ``uniform`` averages every
  unit of the panel; ``query_selected`` averages the configurations that pass
  on the query cells, i.e. chosen with the evaluation labels;
* ``optimism.csv`` — split-half check of choosing a configuration by its
  query performance: choose on a random half of the query cells, score on
  the other half (lineage-traced datasets only);
* ``summary.md`` — the same results in prose-ready tables.

Lineage-traced datasets are scored against their ground truth (accuracy,
macro F1, weighted F1, AUROC). Axolotl has no ground truth and is scored by
the manuscript's dorsal-enrichment description (see ``targets.json``).
"""
import glob
import json
import os
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import apply_selection_gates, prepare_dataset, results_dir  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.metrics import evaluation_frame  # noqa: E402
from revision.common.registry import resolve_path  # noqa: E402
from revision.common.runner import probs_frame  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_targets(dataset: str, path: str) -> dict:
    """Manuscript targets for ``dataset`` (suffixed entries use their base name)."""
    with open(path, encoding='utf-8') as fh:
        targets = json.load(fh)
    return targets[dataset if dataset in targets else re.sub(r'_(hvg\d+|published)$', '', dataset)]


def protocol_members(oof: pd.Series, selection: dict) -> list:
    """Units Ageas keeps under ``selection``: the round gate, then the final OOF gate."""
    better_is_max = selection.get('monitor_type', 'max') == 'max'
    kept = apply_selection_gates(oof, selection['selection_ratio'], selection['retention_point'],
                                 selection['cutoff_point'], selection.get('relative_retention'),
                                 selection.get('monitor_type', 'max'))
    if selection.get('final_gate', 'oof') == 'none':
        return kept
    scores = oof[kept]
    retention = selection['retention_point']
    if selection.get('relative_retention') is not None:
        delta = selection['relative_retention']
        retention = scores.max() - delta if better_is_max else scores.min() + delta
    final = [u for u in kept if (scores[u] >= retention if better_is_max else scores[u] <= retention)]
    return final or [scores.idxmax() if better_is_max else scores.idxmin()]


class FateScorer:
    """Binary fate metrics against lineage ground truth, on all or part of the query cells."""

    metrics = ['accuracy', 'f1_macro', 'auroc']
    columns = ['n', 'accuracy', 'f1_macro', 'f1_weighted', 'auroc']

    def __init__(self, data, cells):
        self.classes = data.classes
        self.cells = cells
        frame = evaluation_frame(probs_frame(np.full((len(cells), len(data.classes)), 0.5), cells, data.classes),
                                 data.ground_truth, data.classes[1], data.classes[0])
        self.rows = pd.Index(cells).get_indexer(frame.index)
        self.y = frame['y'].to_numpy()

    def score(self, probs, subset=None) -> dict:
        p = probs[self.rows, 1] / probs[self.rows].sum(axis=1)
        y = self.y
        if subset is not None:
            p, y = p[subset], y[subset]
        pred = (p >= 0.5).astype(int)
        return dict(n=len(y), accuracy=accuracy_score(y, pred), f1_macro=f1_score(y, pred, average='macro'),
                    f1_weighted=f1_score(y, pred, average='weighted'), auroc=roc_auc_score(y, p))

    @staticmethod
    def passes(scores: dict, targets: dict) -> bool:
        return all(scores[m] >= targets[m] for m in FateScorer.metrics)

    @staticmethod
    def passes_weighted(scores: dict, targets: dict) -> bool:
        """Same check with weighted F1 in place of macro F1.

        The reported iEP ATAC "macro F1" (.783) exceeds its accuracy (.736),
        which only weighted F1 can do for two classes, so the manuscript's F1
        may be weighted.
        """
        return (scores['accuracy'] >= targets['accuracy'] and scores['auroc'] >= targets['auroc']
                and scores['f1_weighted'] >= targets['f1_macro'])

    @staticmethod
    def margin(scores: dict, targets: dict) -> float:
        """Smallest gap to the manuscript over the three metrics (>= 0 means no worse)."""
        return min(scores[m] - targets[m] for m in FateScorer.metrics)


class AxisScorer:
    """Lineage-2 probability along the dorsal axis (no ground truth).

    Scores raw probabilities and, as ``*_minmax``, probabilities min-max
    rescaled over the whole query probability matrix, which is what
    ``docs/examples/spatial_example.ipynb`` plotted for the manuscript figure.
    """

    metrics = ['ventral_mean', 'dorsal_mean', 'spearman', 'ventral_mean_minmax', 'dorsal_mean_minmax']
    columns = ['n'] + metrics

    def __init__(self, data, cells):
        spec = data.spec
        raw = ad.read_h5ad(resolve_path(spec), backed='r')
        obs = raw.obs[[spec['label_key']]].copy()
        xy = np.asarray(raw.obsm[spec.get('spatial_obsm', 'spatial')])[:, :2].astype(float)
        raw.file.close()
        states = obs[spec['label_key']].astype(str).to_numpy()
        start = xy[states == spec['dorsal_axis']['from_state']].mean(axis=0)
        end = xy[states == spec['dorsal_axis']['to_state']].mean(axis=0)
        unit = (end - start) / np.linalg.norm(end - start)
        dorsal = pd.Series((xy - start) @ unit, index=obs.index)
        self.dorsal = dorsal.loc[cells].to_numpy()
        order = np.argsort(self.dorsal)
        fifth = max(len(order) // 5, 1)
        self.ventral_idx, self.dorsal_idx = order[:fifth], order[-fifth:]

    def score(self, probs, subset=None) -> dict:
        p = probs[:, 1] / probs.sum(axis=1)
        low, high = probs.min(), probs.max()
        scaled = (probs[:, 1] - low) / (high - low) if high > low else np.zeros(len(p))
        return dict(n=len(p), ventral_mean=float(p[self.ventral_idx].mean()),
                    dorsal_mean=float(p[self.dorsal_idx].mean()),
                    spearman=float(spearmanr(self.dorsal, p).correlation),
                    ventral_mean_minmax=float(scaled[self.ventral_idx].mean()),
                    dorsal_mean_minmax=float(scaled[self.dorsal_idx].mean()))

    @staticmethod
    def passes(scores: dict, targets: dict) -> bool:
        return scores['ventral_mean'] <= targets['ventral_max'] and scores['dorsal_mean'] > targets['dorsal_min']

    @staticmethod
    def passes_minmax(scores: dict, targets: dict) -> bool:
        return (scores['ventral_mean_minmax'] <= targets['ventral_max']
                and scores['dorsal_mean_minmax'] > targets['dorsal_min'])


def load_runs(out: str):
    """Successful runs (all ``runs*.csv`` files), ``{seed: {unit_id: probs}}`` and the query cell order."""
    runs = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(out, 'runs*.csv')))],
                     ignore_index=True)
    runs = runs[runs['status'] == 'ok'].drop_duplicates(['unit_id', 'seed'], keep='last')
    probs, cells = {}, None
    for row in runs.itertuples():
        npz = np.load(os.path.join(out, 'probs', f'{row.unit_id}_seed{row.seed}.npz'), allow_pickle=True)
        probs.setdefault(int(row.seed), {})[row.unit_id] = npz['probs']
        if cells is None:
            cells = npz['cells']
    return runs, probs, cells


def unit_sources(manifest: pd.DataFrame, unit_id: str) -> dict:
    """Panels, source files and legacy translation of one pooled unit."""
    src = manifest[manifest['unit_id'] == unit_id]
    return dict(panels='; '.join(sorted(src['panel'].unique())), sources='; '.join(src['source']),
                translation='; '.join(sorted(set(src['translation'].fillna('')) - {''})))


def config_table(runs, probs, scorer, targets, manifest) -> pd.DataFrame:
    """One row per configuration: seed-mean query metrics, OOF accuracy, pass flags, sources."""
    rows = []
    for unit_id, unit_runs in runs.groupby('unit_id'):
        per_seed = [scorer.score(probs[int(s)][unit_id]) for s in unit_runs['seed']]
        frame = pd.DataFrame(per_seed)
        mean = frame.mean().to_dict()
        row = dict(unit_id=unit_id, type=unit_runs['type'].iloc[0], n_seeds=len(frame))
        for m in scorer.columns[1:]:
            row[m] = mean[m]
            row[f'{m}_sd'] = frame[m].std() if len(frame) > 1 else np.nan
        row['oof.accuracy'] = unit_runs['oof.accuracy'].mean()
        row['passes_mean'] = scorer.passes(mean, targets)
        row['passes_seeds'] = int(sum(scorer.passes(s, targets) for s in per_seed))
        if hasattr(scorer, 'passes_minmax'):
            row['passes_mean_minmax'] = scorer.passes_minmax(mean, targets)
            row['passes_seeds_minmax'] = int(sum(scorer.passes_minmax(s, targets) for s in per_seed))
        if hasattr(scorer, 'passes_weighted'):
            row['passes_mean_weighted'] = scorer.passes_weighted(mean, targets)
            row['passes_seeds_weighted'] = int(sum(scorer.passes_weighted(s, targets) for s in per_seed))
        row['seconds'] = unit_runs['seconds'].mean()
        row.update(unit_sources(manifest, unit_id))
        rows.append(row)
    table = pd.DataFrame(rows).set_index('unit_id')
    key = 'auroc' if 'auroc' in table else 'spearman'
    return table.sort_values(['passes_mean', key], ascending=False)


def ensemble_table(runs, probs, scorer, targets, manifest, selection, passing) -> tuple:
    """Per-seed and seed-mean scores of the offline ensembles."""
    panels = manifest.groupby('panel')['unit_id'].apply(lambda s: sorted(set(s))).to_dict()
    panels['all'] = sorted(manifest['unit_id'].unique())
    # Seeds run with --no-cv have no out-of-fold scores; their gates use the unit's mean over CV seeds.
    oof_mean = runs.groupby('unit_id')['oof.accuracy'].mean()
    rows = []
    for seed, members in probs.items():
        oof = runs[runs['seed'] == seed].set_index('unit_id')['oof.accuracy']
        oof = oof.fillna(oof_mean.reindex(oof.index))
        built = {}
        for panel, ids in panels.items():
            ids = [u for u in ids if u in members]
            scored = [u for u in ids if pd.notna(oof.get(u))]
            if scored:
                built[(panel, 'protocol')] = protocol_members(oof[scored], selection)
            if ids:
                built[(panel, 'uniform')] = ids
        chosen = [u for u in passing if u in members]
        if chosen:
            built[('all', 'query_selected')] = chosen
        for (panel, variant), ids in built.items():
            score = scorer.score(np.mean([members[u] for u in ids], axis=0))
            rows.append(dict(panel=panel, variant=variant, seed=seed, n_units=len(ids),
                             units=';'.join(ids), **score))
    per_seed = pd.DataFrame(rows)
    cols = [c for c in scorer.columns[1:]] + ['n_units']
    summary = per_seed.groupby(['panel', 'variant'])[cols].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    means = [{m: summary.loc[i, f'{m}_mean'] for m in scorer.metrics} for i in summary.index]
    summary['passes_mean'] = [scorer.passes(m, targets) for m in means]
    if hasattr(scorer, 'passes_minmax'):
        summary['passes_mean_minmax'] = [scorer.passes_minmax(m, targets) for m in means]
    if hasattr(scorer, 'passes_weighted'):
        means = [{m: summary.loc[i, f'{m}_mean'] for m in scorer.columns[1:]} for i in summary.index]
        summary['passes_mean_weighted'] = [scorer.passes_weighted(m, targets) for m in means]
    return per_seed, summary


def optimism_table(runs, probs, scorer, targets, n_splits: int = 200, seed: int = 0) -> pd.DataFrame:
    """Choose the configuration with the best margin to the manuscript on half the query cells; score it on the rest."""
    rng = np.random.default_rng(seed)
    units = sorted(runs['unit_id'].unique())
    seeds = {u: [int(s) for s in runs.loc[runs['unit_id'] == u, 'seed']] for u in units}
    rows = []
    for split in range(n_splits):
        half = np.zeros(len(scorer.y), dtype=bool)
        for cls in (0, 1):
            idx = np.flatnonzero(scorer.y == cls)
            half[rng.permutation(idx)[: len(idx) // 2]] = True

        def mean_scores(unit, subset):
            return pd.DataFrame([scorer.score(probs[s][unit], subset) for s in seeds[unit]]).mean().to_dict()

        chosen_scores = {u: mean_scores(u, half) for u in units}
        best = max(units, key=lambda u: FateScorer.margin(chosen_scores[u], targets))
        held_out = mean_scores(best, ~half)
        passing = [u for u in units if FateScorer.passes(chosen_scores[u], targets)]
        rows.append(dict(split=split, chosen=best,
                         **{f'{m}_selection_half': chosen_scores[best][m] for m in FateScorer.metrics},
                         **{f'{m}_held_out': held_out[m] for m in FateScorer.metrics},
                         passes_selection_half=FateScorer.passes(chosen_scores[best], targets),
                         passes_held_out=FateScorer.passes(held_out, targets),
                         n_passing_selection_half=len(passing),
                         n_passing_also_held_out=sum(FateScorer.passes(mean_scores(u, ~half), targets)
                                                     for u in passing)))
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, digits: int = 3) -> str:
    """Markdown table (without the optional ``tabulate`` dependency)."""
    frame = frame.round(digits)
    names = [n for n in frame.index.names if n] or ['']
    head = [' / '.join(names)] + [str(c) for c in frame.columns]
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for idx, values in frame.iterrows():
        idx = ' / '.join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
        lines.append('| ' + ' | '.join([idx] + [str(v) for v in values]) + ' |')
    return '\n'.join(lines)


def markdown(dataset, targets, configs, ensembles, optimism, scorer) -> str:
    """Plain-text summary for the report."""
    metric_cols = scorer.columns[1:]
    flags = [c for c in ('passes_mean', 'passes_mean_minmax', 'passes_mean_weighted') if c in configs]
    lines = [f'# E17: {dataset}', '',
             'Targets: ' + ', '.join(f'{k} {v}' for k, v in targets.items() if k != 'source')
             + f" ({targets.get('source', '')})", '']
    alternates = [c for flag in flags[1:] for c in (flag, flag.replace('_mean', '_seeds'))]
    show = ['type', 'n_seeds', 'passes_seeds'] + alternates + list(metric_cols) + [
        'oof.accuracy', 'panels', 'sources', 'translation']
    for flag in flags:
        passing = configs[configs[flag]]
        what = {'passes_mean': 'on the seed mean', 'passes_mean_minmax': 'after min-max rescaling',
                'passes_mean_weighted': 'with weighted F1 in place of macro F1'}[flag]
        lines += [f'## Configurations no worse than the manuscript, {what} ({len(passing)} of {len(configs)})', '']
        if not len(passing):
            lines += ['None. The ten best by the ranking metric:', '']
        lines += [md_table((passing if len(passing) else configs.head(10))[show]), '']
    lines += ['## Ensembles (mean over seeds)', '',
              md_table(ensembles[[f'{m}_mean' for m in metric_cols] + ['n_units_mean'] + flags]), '']
    if optimism is not None and len(optimism):
        cols = [c for c in optimism.columns
                if c.endswith(('_selection_half', '_held_out')) and not c.startswith('passes')]
        lines += ['## Choosing by query performance: split-half check', '',
                  f'{len(optimism)} random halves. The configuration closest to the manuscript on one half '
                  f"is no worse than it on that half in {optimism['passes_selection_half'].mean():.0%} of splits, "
                  f"and on the other half in {optimism['passes_held_out'].mean():.0%}.", '',
                  md_table(optimism[cols].mean().to_frame('mean')), '']
    return '\n'.join(lines)


CV_METRICS = ['accuracy', 'f1_macro', 'f1_weighted']


def class_scores(y, probs) -> dict:
    """Multi-class accuracy, macro F1 and weighted F1 of the argmax predictions."""
    pred = np.asarray(probs).argmax(axis=1)
    return dict(n=len(y), accuracy=accuracy_score(y, pred), f1_macro=f1_score(y, pred, average='macro'),
                f1_weighted=f1_score(y, pred, average='weighted'))


def score_cv(args, spec: dict, targets: dict) -> str:
    """Classification benchmark: each configuration's median over held-out folds vs the manuscript.

    Reads ``<results-root>/E17/<dataset>_fold<k>/`` (written by
    ``run_configs.py --fold k``) and writes ``configs.csv``, ``ensembles.csv``
    and ``summary.md`` to ``<results-root>/E17/<dataset>/``. A configuration
    passes when its median accuracy and median macro F1 over all folds reach
    the manuscript's. ``uniform`` ensembles average every unit of a panel;
    ``query_selected`` averages the configurations that pass, chosen with the
    held-out labels and therefore optimistic.
    """
    classes = list(spec['terminal_classes'])
    n_folds = spec['cv'].get('n_folds', 5)
    rows, folds, manifest = [], {}, None
    for fold in range(n_folds):
        fold_dir = results_dir('E17', f'{args.dataset}_fold{fold}', args.results_root)
        if not glob.glob(os.path.join(fold_dir, 'runs*.csv')):
            continue
        manifest = pd.read_csv(os.path.join(fold_dir, 'pool_manifest.csv'))
        _, probs, cells = load_runs(fold_dir)
        truth = pd.read_csv(os.path.join(fold_dir, 'truth.csv'), index_col=0)['label'].astype(str)
        y = pd.Categorical(truth.reindex(cells), categories=classes).codes
        folds[fold] = (y, probs)
        for seed, members in probs.items():
            for unit_id, p in members.items():
                rows.append(dict(fold=fold, seed=seed, unit_id=unit_id, type=unit_id.rsplit('_', 1)[0],
                                 **class_scores(y, p)))
    per_fold = pd.DataFrame(rows)
    by_fold = per_fold.groupby(['unit_id', 'fold'])[CV_METRICS].mean()
    table = by_fold.groupby('unit_id').agg(['median', 'mean'])
    table.columns = ['_'.join(c) for c in table.columns]
    table['n_folds'] = by_fold.groupby('unit_id').size()
    table['type'] = [u.rsplit('_', 1)[0] for u in table.index]
    table['passes'] = ((table['n_folds'] == n_folds)
                       & (table['accuracy_median'] >= targets['accuracy'])
                       & (table['f1_macro_median'] >= targets['f1_macro']))
    table = table.join(pd.DataFrame({u: unit_sources(manifest, u) for u in table.index}).T)
    table = table.sort_values(['passes', 'accuracy_median'], ascending=False)

    passing = list(table.index[table['passes']])
    panels = manifest.groupby('panel')['unit_id'].apply(lambda s: sorted(set(s))).to_dict()
    panels['all'] = sorted(manifest['unit_id'].unique())
    variants = {(panel, 'uniform'): ids for panel, ids in panels.items()}
    if passing:
        variants[('all', 'query_selected')] = passing
    ens_rows = []
    for fold, (y, probs) in folds.items():
        for seed, members in probs.items():
            for (panel, variant), ids in variants.items():
                ids = [u for u in ids if u in members]
                if ids:
                    ens_rows.append(dict(fold=fold, seed=seed, panel=panel, variant=variant, n_units=len(ids),
                                         **class_scores(y, np.mean([members[u] for u in ids], axis=0))))
    ens = pd.DataFrame(ens_rows)
    ens_fold = ens.groupby(['panel', 'variant', 'fold'])[CV_METRICS + ['n_units']].mean()
    ensembles = ens_fold.groupby(['panel', 'variant']).agg(['median', 'mean'])
    ensembles.columns = ['_'.join(c) for c in ensembles.columns]
    ensembles['n_folds'] = ens_fold.groupby(['panel', 'variant']).size()
    ensembles['passes'] = ((ensembles['n_folds'] == n_folds)
                           & (ensembles['accuracy_median'] >= targets['accuracy'])
                           & (ensembles['f1_macro_median'] >= targets['f1_macro']))

    out = results_dir('E17', args.dataset, args.results_root)
    per_fold.to_csv(os.path.join(out, 'configs_per_fold.csv'), index=False)
    table.to_csv(os.path.join(out, 'configs.csv'))
    ensembles.to_csv(os.path.join(out, 'ensembles.csv'))
    show = ['type', 'n_folds', 'accuracy_median', 'f1_macro_median', 'f1_weighted_median', 'accuracy_mean',
            'f1_macro_mean', 'panels', 'sources', 'translation']
    lines = [f'# E17: {args.dataset}', '',
             f"Targets: median accuracy {targets['accuracy']}, median macro F1 {targets['f1_macro']} over "
             f"{n_folds} folds ({targets.get('source', '')})", '',
             f'## Configurations no worse than the manuscript ({len(passing)} of {len(table)})', '']
    if not passing:
        lines += ['None. The ten best by median accuracy:', '']
    lines += [md_table((table.loc[passing] if passing else table.head(10))[show]), '',
              '## Ensembles (median over folds)', '',
              md_table(ensembles[[f'{m}_median' for m in CV_METRICS] + ['n_units_median', 'n_folds', 'passes']]), '']
    text = '\n'.join(lines)
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    return text


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--targets', default=os.path.join(HERE, 'targets.json'))
    parser.add_argument('--splits', type=int, default=200, help='Random halves for the optimism check.')
    args = parser.parse_args(argv)
    registry, protocol, _, _ = resolve(args)

    targets = load_targets(args.dataset, args.targets)
    if 'cv' in registry[args.dataset]:
        print(score_cv(args, registry[args.dataset], targets))
        return
    out = results_dir('E17', args.dataset, args.results_root)
    manifest = pd.read_csv(os.path.join(out, 'pool_manifest.csv'))
    runs, probs, cells = load_runs(out)
    if args.seeds:  # score only these seeds (e.g. --seeds 0 while later seeds are still running)
        runs = runs[runs['seed'].isin(args.seeds)]
        probs = {s: p for s, p in probs.items() if s in args.seeds}
    data = prepare_dataset(args.dataset, registry)
    scorer = FateScorer(data, cells) if data.ground_truth is not None else AxisScorer(data, cells)

    configs = config_table(runs, probs, scorer, targets, manifest)
    configs.to_csv(os.path.join(out, 'configs.csv'))
    per_seed, ensembles = ensemble_table(runs, probs, scorer, targets, manifest,
                                         protocol['selection'], list(configs.index[configs['passes_mean']]))
    per_seed.to_csv(os.path.join(out, 'ensembles_per_seed.csv'), index=False)
    ensembles.to_csv(os.path.join(out, 'ensembles.csv'))
    optimism = None
    if isinstance(scorer, FateScorer) and args.splits:
        optimism = optimism_table(runs, probs, scorer, targets, args.splits)
        optimism.to_csv(os.path.join(out, 'optimism.csv'), index=False)
    text = markdown(args.dataset, targets, configs, ensembles, optimism, scorer)
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(text)


if __name__ == '__main__':
    main()
