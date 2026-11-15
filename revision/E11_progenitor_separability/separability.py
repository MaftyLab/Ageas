#!/usr/bin/env python3
"""E11 · How separable are the fate groups of progenitors without supervision?

For progenitors with unipotent lineage labels, in an unsupervised embedding
fitted on all prepared cells (PCA for dense data; truncated SVD/LSI without
its depth-dominated first component for sparse data):

* silhouette score of the two ground-truth fate groups;
* 5-fold cross-validated accuracy of logistic regression and kNN on the
  embedding alone (an "oracle separability" ceiling using progenitor labels);
* neighbourhood composition: for each progenitor, the share of its k
  nearest terminal cells that belong to its eventual fate (how close the
  progenitors already sit to their terminal manifold).

Run it for several datasets to compare, e.g. ``--dataset iep_atac_peak
iep_rna larry``. Writes ``revision/results/E11/separability.csv``.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import prepare_dataset, results_dir  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.registry import LABEL_COL, dense  # noqa: E402


def embedding(matrix, n_components=30, seed=0):
    if sparse.issparse(matrix):
        return TruncatedSVD(n_components=n_components + 1, random_state=seed).fit_transform(matrix)[:, 1:]
    return PCA(n_components=min(n_components, matrix.shape[1] - 1), random_state=seed).fit_transform(dense(matrix))


def main(argv=None):
    parser = base_parser(__doc__, dataset_required=False)
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--k', type=int, default=15)
    args = parser.parse_args(argv)
    registry, _, _, _ = resolve(args)
    names = args.datasets or [args.dataset]

    rows = []
    for name in names:
        data = prepare_dataset(name, registry)
        gt = data.ground_truth
        emb = pd.DataFrame(embedding(data.adata.X), index=data.adata.obs_names)
        labelled = gt[gt['label'].isin(data.classes)].index.intersection(data.query.obs_names)
        y = gt.loc[labelled, 'label'].values
        x = emb.loc[labelled].values

        terminal = data.train.obs_names
        t_labels = data.train.obs[LABEL_COL].astype(str).values
        nn = NearestNeighbors(n_neighbors=args.k).fit(emb.loc[terminal].values)
        _, idx = nn.kneighbors(x)
        own_share = np.mean(t_labels[idx] == y[:, None], axis=1)

        rows.append({
            'dataset': name,
            'n_progenitors': len(labelled),
            'silhouette': silhouette_score(x, y) if len(set(y)) > 1 else np.nan,
            'cv_acc_logreg': cross_val_score(LogisticRegression(max_iter=2000), x, y, cv=5).mean(),
            'cv_acc_knn': cross_val_score(KNeighborsClassifier(args.k), x, y, cv=5).mean(),
            'mean_own_fate_terminal_neighbours': float(own_share.mean()),
        })
    table = pd.DataFrame(rows)
    out = results_dir('E11', '', args.results_root)
    table.to_csv(os.path.join(out, 'separability.csv'), index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
