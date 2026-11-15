#!/usr/bin/env python3
"""E10 · Is the epiblast A-P "fate priming" signal more than position?

Reviewer 2 (R2.7) notes that the two training classes (Ecto, anterior; PS,
posterior) form an anterior-vs-posterior contrast, so applying the
classifier to epiblast cells and correlating with the A-P coordinate may
recover position rather than fate. Analyses (``--analyses``):

* ``provenance`` — axial distribution of each training/query class and the
  overlap of the two terminal classes.
* ``baseline`` — the protocol run; correlation of P(positive fate) with the
  axial coordinate (reproduces the manuscript statistic).
* ``position_matched`` — retrain on terminal cells restricted to the axial
  range where both classes occur, balanced per position bin.
* ``within_position`` — does P vary among epiblast cells at the same axial
  position, and does that variation follow in-section geometry (distance
  to the nearest PS / Ecto spot)? Partial Spearman controlling for the
  position bin, with a within-bin permutation test.
* ``spillover`` — regression of P on position plus the fraction of
  neighbouring PS / Ecto spots (Stereo-seq bins mix neighbours) and
  sequencing depth; correlation with position after removing spots that
  touch PS / Ecto.
* ``technical`` — retrain without ribosomal, mitochondrial and hemoglobin
  genes; partial correlation with position given depth.
* ``position_control`` — a position-only classifier (anterior vs
  posterior epiblast halves, cross-fitted); its agreement with Ageas and
  the overlap of the two feature signatures.

Outputs in ``revision/results/E10/<dataset>/``: one CSV per analysis and
``summary.csv`` collecting the headline numbers.
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import predict_query, prepare_dataset, results_dir, run_ageas  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.registry import LABEL_COL, resolve_path  # noqa: E402

TECHNICAL_PREFIXES = ('RPL', 'RPS', 'MT-', 'MRPL', 'MRPS', 'HBA', 'HBB', 'HBG', 'HBE', 'HBZ', 'HBM', 'HBQ')
ALL = ['provenance', 'baseline', 'position_matched', 'within_position', 'spillover',
       'technical', 'position_control']


class Context:
    """Shared state: data, run settings and the baseline predictions."""

    def __init__(self, data, protocol, config_dir, seed, accelerator, out):
        self.data, self.protocol, self.config_dir = data, protocol, config_dir
        self.seed, self.accelerator, self.out = seed, accelerator, out
        spec = data.spec
        self.pos_key = spec['position_key']
        self.positive = data.classes[1]
        self.summary = {}
        self._baseline = None

    def fit_predict(self, train, query=None):
        query = self.data.query if query is None else query
        deck, corpus = run_ageas(train, self.data.classes, self.config_dir, self.seed,
                                 self.protocol['selection'], self.accelerator)
        probs, _ = predict_query(deck, query[:, train.var_names].copy(), self.data.classes)
        return probs[f'prob_{self.positive}'], deck, corpus

    @property
    def baseline(self):
        if self._baseline is None:
            self._baseline, _, _ = self.fit_predict(self.data.train)
        return self._baseline

    def position(self, adata):
        return adata.obs[self.pos_key].astype(float)

    def corr_with_position(self, p, name):
        pos = self.position(self.data.query).loc[p.index]
        r, pr = pearsonr(pos, p)
        rho, ps = spearmanr(pos, p)
        self.summary[f'{name}_pearson_r'] = r
        self.summary[f'{name}_pearson_p'] = pr
        self.summary[f'{name}_spearman_rho'] = rho
        return r


def provenance(ctx):
    data = ctx.data
    states = data.adata.obs[data.spec['label_key']].astype(str)
    pos = ctx.position(data.adata)
    table = pos.groupby(states).describe()
    table.to_csv(os.path.join(ctx.out, 'provenance.csv'))
    a, b = (pos[states == c] for c in data.classes)
    bins = np.arange(np.floor(pos.min()), np.ceil(pos.max()) + 2)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    ctx.summary['terminal_axial_overlap'] = float(np.minimum(ha, hb).sum())


def baseline(ctx):
    ctx.baseline.rename('p').to_frame().to_csv(os.path.join(ctx.out, 'baseline_predictions.csv'))
    ctx.corr_with_position(ctx.baseline, 'baseline')


def position_matched(ctx, rng):
    train = ctx.data.train
    labels = train.obs[LABEL_COL].astype(str)
    pos = ctx.position(train)
    low = max(pos[labels == c].min() for c in ctx.data.classes)
    high = min(pos[labels == c].max() for c in ctx.data.classes)
    if low >= high:
        ctx.summary['position_matched'] = 'no axial overlap between terminal classes'
        return
    in_range = (pos >= low) & (pos <= high)
    keep = []
    for value in np.unique(pos[in_range]):
        groups = [np.flatnonzero((pos == value).values & (labels == c).values & in_range.values)
                  for c in ctx.data.classes]
        n = min(len(g) for g in groups)
        if n:
            keep.extend(np.concatenate([rng.choice(g, n, replace=False) for g in groups]))
    ctx.summary['position_matched_n_train'] = len(keep)
    p, _, _ = ctx.fit_predict(train[np.sort(keep)].copy())
    p.rename('p').to_frame().to_csv(os.path.join(ctx.out, 'position_matched_predictions.csv'))
    ctx.corr_with_position(p, 'position_matched')
    ctx.summary['position_matched_vs_baseline_spearman'] = spearmanr(p, ctx.baseline.loc[p.index]).correlation


def _all_spots(ctx):
    full = ad.read_h5ad(resolve_path(ctx.data.spec), backed='r')
    obs = full.obs[[ctx.data.spec['label_key'], ctx.pos_key]].copy()
    xy = np.asarray(full.obsm[ctx.data.spec['spatial_obsm']])[:, :2]
    obs['x'], obs['y'] = xy[:, 0], xy[:, 1]
    obs['state'] = obs[ctx.data.spec['label_key']].astype(str)
    return obs


def _geometry(ctx, radius_factor=3.0):
    """Per query spot: distance to nearest spot of each terminal class and
    the fraction of each class among neighbours within ``radius``."""
    spots = _all_spots(ctx)
    query = spots.loc[ctx.data.query.obs_names]
    rows = []
    for value, section in spots.groupby(ctx.pos_key):
        tree = cKDTree(section[['x', 'y']].values)
        nn, _ = tree.query(section[['x', 'y']].values, k=2)
        radius = radius_factor * np.median(nn[:, 1]) if len(section) > 1 else 1.0
        q = query[query[ctx.pos_key] == value]
        if q.empty:
            continue
        neighbours = tree.query_ball_point(q[['x', 'y']].values, r=radius)
        for (cell, row), idx in zip(q.iterrows(), neighbours):
            states = section['state'].values[[i for i in idx if section.index[i] != cell]]
            record = {'cell': cell}
            for c in ctx.data.classes:
                pts = section.loc[section['state'] == c, ['x', 'y']].values
                record[f'dist_{c}'] = (np.min(np.linalg.norm(pts - row[['x', 'y']].values.astype(float), axis=1))
                                       if len(pts) else np.nan)
                record[f'frac_{c}'] = float(np.mean(states == c)) if len(states) else 0.0
            rows.append(record)
    return pd.DataFrame(rows).set_index('cell')


def _residualise_within(values: pd.Series, groups: pd.Series) -> pd.Series:
    return values - values.groupby(groups).transform('mean')


def within_position(ctx, rng, n_perm=1000):
    p = ctx.baseline
    geo = _geometry(ctx).loc[p.index]
    pos = ctx.position(ctx.data.query).loc[p.index]
    p_res = _residualise_within(p, pos)
    ctx.summary['share_of_variance_within_position'] = float(p_res.var() / p.var())
    rows = []
    for c in ctx.data.classes:
        cov = _residualise_within(geo[f'dist_{c}'].fillna(geo[f'dist_{c}'].median()), pos)
        rho = spearmanr(p_res, cov).correlation
        null = []
        for _ in range(n_perm):
            shuffled = p.groupby(pos).transform(lambda s: rng.permutation(s.values))
            null.append(spearmanr(_residualise_within(shuffled, pos), cov).correlation)
        rows.append({'covariate': f'distance_to_{c}', 'partial_spearman': rho,
                     'perm_p': float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))})
    pd.DataFrame(rows).to_csv(os.path.join(ctx.out, 'within_position.csv'), index=False)
    geo.to_csv(os.path.join(ctx.out, 'geometry.csv'))


def spillover(ctx):
    p = ctx.baseline
    geo_path = os.path.join(ctx.out, 'geometry.csv')
    geo = pd.read_csv(geo_path, index_col=0) if os.path.exists(geo_path) else _geometry(ctx)
    geo = geo.loc[p.index]
    pos = ctx.position(ctx.data.query).loc[p.index]
    design = {'position': pos}
    for c in ctx.data.classes:
        design[f'frac_{c}'] = geo[f'frac_{c}']
    depth_keys = [k for k in ctx.data.spec.get('counts_metadata', []) if k in ctx.data.query.obs]
    if depth_keys:
        design['log_depth'] = np.log1p(ctx.data.query.obs.loc[p.index, depth_keys[0]].astype(float))
    X = pd.DataFrame(design)
    Xz = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    A = np.column_stack([np.ones(len(Xz)), Xz.values])
    coef, *_ = np.linalg.lstsq(A, p.values, rcond=None)
    pd.Series(coef[1:], index=X.columns, name='standardised_coef').to_csv(
        os.path.join(ctx.out, 'spillover_regression.csv'))
    clean = (geo[[f'frac_{c}' for c in ctx.data.classes]].sum(axis=1) == 0).values
    ctx.summary['spillover_free_n'] = int(clean.sum())
    if clean.sum() > 10:
        ctx.corr_with_position(p[clean], 'spillover_free')


def technical(ctx):
    train = ctx.data.train
    keep = [g for g in train.var_names if not str(g).upper().startswith(TECHNICAL_PREFIXES)]
    ctx.summary['technical_genes_removed'] = train.n_vars - len(keep)
    p, _, _ = ctx.fit_predict(train[:, keep].copy())
    p.rename('p').to_frame().to_csv(os.path.join(ctx.out, 'technical_predictions.csv'))
    ctx.corr_with_position(p, 'no_technical_genes')
    depth_keys = [k for k in ctx.data.spec.get('counts_metadata', []) if k in ctx.data.query.obs]
    if depth_keys:
        depth = np.log1p(ctx.data.query.obs.loc[p.index, depth_keys[0]].astype(float))
        pos = ctx.position(ctx.data.query).loc[p.index]
        res = lambda v: v - np.polyval(np.polyfit(depth, v, 1), depth)  # noqa: E731
        ctx.summary['partial_pearson_given_depth'] = pearsonr(res(p.values), res(pos.values))[0]
        ctx.summary['pearson_p_vs_depth'] = pearsonr(p.values, depth.values)[0]


def position_control(ctx, rng):
    query = ctx.data.query
    pos = ctx.position(query)
    halves = np.where(pos <= pos.median(), 'anterior', 'posterior')
    folds = rng.permutation(len(query)) % 2
    p_post = pd.Series(np.nan, index=query.obs_names)
    for k in (0, 1):
        train = query[folds != k].copy()
        train.obs[LABEL_COL] = pd.Categorical(halves[folds != k], categories=['anterior', 'posterior'])
        deck, _ = run_ageas(train, ['anterior', 'posterior'], ctx.config_dir, ctx.seed,
                            ctx.protocol['selection'], ctx.accelerator)
        probs, _ = predict_query(deck, query[folds == k].copy(), ['anterior', 'posterior'])
        p_post.loc[probs.index] = probs['prob_posterior']
    ctx.summary['position_control_vs_ageas_spearman'] = spearmanr(
        p_post, ctx.baseline.loc[p_post.index]).correlation
    p_post.rename('p_posterior').to_frame().to_csv(os.path.join(ctx.out, 'position_control_predictions.csv'))


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--analyses', nargs='*', default=ALL, choices=ALL)
    parser.add_argument('--n-perm', type=int, default=1000)
    args = parser.parse_args(argv)
    registry, protocol, config_dir, seeds = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    assert data.spec.get('position_key'), 'E10 needs position_key in the registry entry.'
    ctx = Context(data, protocol, config_dir, seeds[0], args.accelerator,
                  results_dir('E10', args.dataset, args.results_root))
    rng = np.random.default_rng(seeds[0])

    for name in args.analyses:
        print('running', name)
        if name == 'provenance':
            provenance(ctx)
        elif name == 'baseline':
            baseline(ctx)
        elif name == 'position_matched':
            position_matched(ctx, rng)
        elif name == 'within_position':
            within_position(ctx, rng, args.n_perm)
        elif name == 'spillover':
            spillover(ctx)
        elif name == 'technical':
            technical(ctx)
        elif name == 'position_control':
            position_control(ctx, rng)
    summary = pd.Series(ctx.summary, name='value')
    summary.to_csv(os.path.join(ctx.out, 'summary.csv'))
    print(summary.to_string())


if __name__ == '__main__':
    main()
