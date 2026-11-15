#!/usr/bin/env python3
"""E08 · Iterative masking: per-iteration diagnostics and redundancy of later tiers.

Re-implements the ``n_iter_extraction`` loop step by step so each iteration's
ensemble can also score the progenitors:

per iteration — features in, units kept, mean out-of-fold score,
progenitor AUROC (if ground truth), features masked (outliers, and
correlated features with ``--correlation-masking``).

For every feature masked in iteration t >= 2 it then measures its
redundancy with features masked earlier: the maximum |Spearman| correlation
and the cross-validated R^2 of predicting it from all earlier-tier features
(ridge), in terminal cells and in progenitors, against an
expression-matched random-feature null.

Outputs in ``revision/results/E08/<dataset>/``: ``iterations.csv``,
``tiers.csv``, ``redundancy.csv`` and ``redundancy_summary.csv``.
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import predict_query, prepare_dataset, results_dir, run_ageas  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.metrics import binary_scores, evaluation_frame  # noqa: E402
from revision.common.registry import dense  # noqa: E402
from revision.common.runner import unit_table  # noqa: E402

extraction = importlib.import_module('ageas.ops.n_iter_extraction')


def redundancy(target: np.ndarray, earlier: np.ndarray) -> tuple:
    """Max |Spearman| with earlier-tier features and CV R^2 from them."""
    rho = [abs(spearmanr(target, earlier[:, j]).correlation) for j in range(earlier.shape[1])]
    r2 = cross_val_score(RidgeCV(alphas=np.logspace(-3, 3, 7)), earlier, target, cv=5, scoring='r2')
    return float(np.nanmax(rho)), float(np.mean(r2))


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--iterations', type=int, default=None, help='N (default: protocol).')
    parser.add_argument('--alpha', type=float, default=None, help='IQR multiplier (default: protocol).')
    parser.add_argument('--correlation-masking', type=float, default=None)
    args = parser.parse_args(argv)
    registry, protocol, config_dir, seeds = resolve(args)
    n_iter = args.iterations or protocol['extraction']['max_extraction_iter']
    alpha = args.alpha if args.alpha is not None else protocol['extraction']['outlier_iqr_factor']

    data = prepare_dataset(args.dataset, registry)
    out = results_dir('E08', args.dataset, args.results_root)
    seed = seeds[0]
    features = list(data.train.var_names)
    iterations, tiers = [], {}

    for i in range(n_iter):
        train, query = data.train[:, features].copy(), data.query[:, features].copy()
        deck, corpus = run_ageas(train, data.classes, config_dir, seed,
                                 protocol['selection'], args.accelerator)
        units = unit_table(deck)
        row = {'iteration': i + 1, 'n_features_in': len(features), 'n_units': len(units),
               'mean_oof_accuracy': units['oof.accuracy'].mean()}
        if data.ground_truth is not None:
            probs, _ = predict_query(deck, query, data.classes)
            frame = evaluation_frame(probs, data.ground_truth, data.classes[1], data.classes[0])
            row['progenitor_auroc'] = binary_scores(frame['y'], frame['p'])['auroc']

        exp = deck.debrief(exp_dataset=corpus, verbose=False, **protocol['debrief'])
        exp = exp / exp.abs().sum()
        _, _, masked, correlated = extraction._find_outliers(
            exp, {}, i, corpus.copy(), outlier_iqr_factor=alpha,
            correlation_masking=args.correlation_masking,
        )
        row.update({'n_masked_outliers': len(masked), 'n_masked_correlated': len(correlated)})
        iterations.append(row)
        for f in masked + correlated:
            tiers[f] = i + 1
        features = [f for f in features if f not in tiers]
        print(row)
        if not masked or not features:
            break

    pd.DataFrame(iterations).to_csv(os.path.join(out, 'iterations.csv'), index=False)
    pd.Series(tiers, name='tier').to_csv(os.path.join(out, 'tiers.csv'))

    rng = np.random.default_rng(seed)
    names = pd.Index(data.train.var_names)
    pools = {'terminal': dense(data.train.X), 'progenitor': dense(data.query.X)}
    mean_expr = pd.Series(pools['terminal'].mean(axis=0), index=names)
    decile = pd.qcut(mean_expr.rank(method='first'), 10, labels=False)
    untouched = names.difference(list(tiers))

    rows = []
    for feature, tier in tiers.items():
        earlier = [f for f, t in tiers.items() if t < tier]
        if tier < 2 or not earlier:
            continue
        candidates = untouched[decile[untouched] == decile[feature]]
        null_feature = rng.choice(candidates) if len(candidates) else None
        for pool, x in pools.items():
            e = x[:, names.get_indexer(earlier)]
            rho, r2 = redundancy(x[:, names.get_loc(feature)], e)
            row = {'feature': feature, 'tier': tier, 'pool': pool, 'max_abs_spearman': rho, 'cv_r2': r2}
            if null_feature is not None:
                row['null_max_abs_spearman'], row['null_cv_r2'] = redundancy(
                    x[:, names.get_loc(null_feature)], e)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, 'redundancy.csv'), index=False)
    if not table.empty:
        summary = table.groupby('pool')[[c for c in table.columns if 'spearman' in c or 'r2' in c]].median()
        summary['share_redundant(|rho|>=0.5)'] = table.groupby('pool')['max_abs_spearman'] \
            .apply(lambda s: float((s >= 0.5).mean()))
        summary.to_csv(os.path.join(out, 'redundancy_summary.csv'))
        print(summary.to_string())


if __name__ == '__main__':
    main()
