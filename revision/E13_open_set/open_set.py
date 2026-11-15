#!/usr/bin/env python3
"""E13 · Open-set behaviour: unseen fates, unlisted terminal cells, senescence.

Ageas predicts relative bias among the user-listed fates. This experiment
measures how well the E06 uncertainty/OOD outputs flag cells for which that
assumption fails:

1. ``other_fate_progenitors`` — progenitors whose clonal relatives are
   mostly an unlisted fate (requires a clone matrix and ``extra_keep_values``
   with the other terminal states). Reported: AUROC of each score
   (OOD distance, entropy, epistemic, 1 - max prob) for separating them from
   progenitors of the listed fates.
2. ``unlisted_terminal`` — terminal cells of unlisted types scored as query;
   fraction flagged OOD vs the fraction of listed-fate progenitors flagged.
3. ``senescence`` (``--senescence-genes FILE``, e.g. SenMayo) — Spearman
   correlation of a senescence program score with each uncertainty score.

Outputs in ``revision/results/E13/<dataset>/``.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import prepare_dataset, results_dir, run_ageas  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.runner import _corpus  # noqa: E402
from ground_truth import clonal_fate_table  # noqa: E402
from ageas.tool.uncertainty import decompose_uncertainty, ood_scores, program_scores  # noqa: E402

SCORES = ['knn_distance', 'total_entropy', 'epistemic', 'one_minus_max_prob']


def score_cells(deck, data, adata):
    corpus = _corpus(adata, data.classes)
    _, _, members = deck.predict(corpus, batch_size=512, num_workers=0, return_members=True)
    table = decompose_uncertainty(members, index=adata.obs_names)
    table['one_minus_max_prob'] = 1 - table['max_prob']
    return table.join(ood_scores(data.train.X, adata.X, query_index=adata.obs_names))


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--senescence-genes', default=None, help='Text file, one gene per line.')
    args = parser.parse_args(argv)
    registry, protocol, config_dir, seeds = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    spec = data.spec
    out = results_dir('E13', args.dataset, args.results_root)
    deck, _ = run_ageas(data.train, data.classes, config_dir, seeds[0], protocol['selection'],
                        args.accelerator)
    query_scores = score_cells(deck, data, data.query)
    query_scores.to_csv(os.path.join(out, 'query_scores.csv'))
    rows = []

    others = spec.get('extra_keep_values') or []
    gt_spec = spec.get('ground_truth') or {}
    if others and gt_spec.get('type') == 'clone_matrix':
        import anndata as ad
        from revision.common.registry import resolve_path
        raw = ad.read_h5ad(resolve_path(spec))
        raw.obs_names_make_unique()
        fates = tuple(data.classes) + tuple(others)
        table = clonal_fate_table(
            raw, fates=fates, progenitor_mask=raw.obs[spec['label_key']].astype(str).isin(spec['query_values']),
            clone_obsm=gt_spec.get('clone_obsm', 'X_clone'), state_key=gt_spec.get('state_key', spec['label_key']),
            time_key=gt_spec.get('time_key'), min_descendants=1,
        )
        listed = table[[f'n_{f}' for f in data.classes]].sum(axis=1)
        unlisted = table[[f'n_{f}' for f in others]].sum(axis=1)
        informative = table.index[(listed + unlisted) > 0].intersection(query_scores.index)
        is_other = (unlisted > listed).loc[informative].astype(int)
        for score in SCORES:
            if is_other.nunique() == 2:
                rows.append({'test': 'other_fate_progenitors', 'score': score,
                             'auroc': roc_auc_score(is_other, query_scores.loc[informative, score]),
                             'n_other': int(is_other.sum()), 'n_listed': int((1 - is_other).sum())})

    if others:
        states = data.adata.obs[spec['label_key']].astype(str)
        unlisted_cells = data.adata[states.isin(others).values].copy()
        if unlisted_cells.n_obs:
            u = score_cells(deck, data, unlisted_cells)
            u.to_csv(os.path.join(out, 'unlisted_terminal_scores.csv'))
            rows.append({'test': 'unlisted_terminal', 'score': 'is_ood',
                         'flagged_unlisted': float(u['is_ood'].mean()),
                         'flagged_query': float(query_scores['is_ood'].mean())})

    if args.senescence_genes:
        genes = [g.strip() for g in open(args.senescence_genes) if g.strip()]
        sen = program_scores(data.query.X, data.query.var_names, {'senescence': genes},
                             reference=data.train.X, index=data.query.obs_names)['senescence']
        for score in SCORES:
            rows.append({'test': 'senescence', 'score': score,
                         'spearman': spearmanr(sen, query_scores[score]).correlation,
                         'n_genes_found': int(np.isin(genes, data.query.var_names).sum())})

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, 'open_set.csv'), index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
