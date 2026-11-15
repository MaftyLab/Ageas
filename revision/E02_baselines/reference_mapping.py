#!/usr/bin/env python3
"""E02 · Transparent reference-mapping and simple supervised baselines.

Every method sees exactly what Ageas sees: terminal cells (with labels) for
fitting and query progenitors for prediction, in the same feature space.

* ``centroid_pearson`` / ``centroid_spearman`` — correlation of each
  progenitor with each terminal-class centroid; probabilities are a softmax
  of the correlations (temperature ``--tau``).
* ``knn15`` / ``knn30`` — kNN label transfer in a PCA (RNA) or truncated-SVD
  /LSI (sparse, ATAC) space fitted on terminal cells only; probability =
  fraction of neighbours per class.
* ``l2_logreg`` — L2-regularised logistic regression on standardised
  features (C chosen by 5-fold CV on terminal cells).

Also writes ``ambiguous_by_similarity.csv``: per progenitor, the centroid
correlation gap and a flag for the lowest quartile (cells that terminal-state
similarity cannot place; the key R1.1 stratum).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import prepare_dataset, results_dir, save_predictions  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.registry import LABEL_COL, dense  # noqa: E402
from revision.common.runner import probs_frame  # noqa: E402


def _softmax(scores: np.ndarray, tau: float) -> np.ndarray:
    z = scores / tau
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def centroid_correlation(train_x, labels, query_x, classes, method='pearson'):
    """Correlation of each query cell with each class centroid."""
    if method == 'spearman':
        train_x = np.apply_along_axis(rankdata, 1, train_x)
        query_x = np.apply_along_axis(rankdata, 1, query_x)
    centroids = np.vstack([train_x[labels == c].mean(axis=0) for c in classes])
    qc = query_x - query_x.mean(axis=1, keepdims=True)
    cc = centroids - centroids.mean(axis=1, keepdims=True)
    num = qc @ cc.T
    den = np.linalg.norm(qc, axis=1)[:, None] * np.linalg.norm(cc, axis=1)[None, :]
    return num / np.where(den == 0, 1, den)


def embed(train_matrix, query_matrix, n_components=50, seed=0):
    """PCA (dense) or truncated SVD (sparse/LSI) fitted on terminal cells."""
    n_components = min(n_components, train_matrix.shape[1] - 1, train_matrix.shape[0] - 1)
    if sparse.issparse(train_matrix):
        model = TruncatedSVD(n_components=n_components + 1, random_state=seed)
        tr, qu = model.fit_transform(train_matrix), model.transform(query_matrix)
        return tr[:, 1:], qu[:, 1:]          # drop the depth-dominated first LSI component
    model = PCA(n_components=n_components, random_state=seed)
    return model.fit_transform(dense(train_matrix)), model.transform(dense(query_matrix))


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--tau', type=float, default=0.05, help='Softmax temperature for centroids.')
    parser.add_argument('--n-components', type=int, default=50)
    args = parser.parse_args(argv)
    registry, _, _, _ = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    out = results_dir('E02', args.dataset, args.results_root)
    classes = data.classes
    labels = data.train.obs[LABEL_COL].astype(str).values
    train_x, query_x = dense(data.train.X), dense(data.query.X)
    cells = data.query.obs_names

    corr = {}
    for method in ('pearson', 'spearman'):
        corr[method] = centroid_correlation(train_x, labels, query_x, classes, method)
        save_predictions(probs_frame(_softmax(corr[method], args.tau), cells, classes),
                         os.path.join(out, f'centroid_{method}.csv'), f'centroid_{method}')

    train_emb, query_emb = embed(data.train.X, data.query.X, args.n_components)
    for k in (15, 30):
        knn = KNeighborsClassifier(n_neighbors=k).fit(train_emb, labels)
        probs = knn.predict_proba(query_emb)[:, [list(knn.classes_).index(c) for c in classes]]
        save_predictions(probs_frame(probs, cells, classes), os.path.join(out, f'knn{k}.csv'), f'knn{k}')

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=10, cv=5, penalty='l2', max_iter=5000, n_jobs=-1),
    ).fit(train_x, labels)
    clf = logreg[-1]
    probs = logreg.predict_proba(query_x)[:, [list(clf.classes_).index(c) for c in classes]]
    save_predictions(probs_frame(probs, cells, classes), os.path.join(out, 'l2_logreg.csv'), 'l2_logreg')

    gap = np.abs(corr['pearson'][:, 0] - corr['pearson'][:, 1]) if len(classes) == 2 else \
        np.sort(corr['pearson'], axis=1)[:, -1] - np.sort(corr['pearson'], axis=1)[:, -2]
    ambiguous = pd.DataFrame({'corr_gap': gap}, index=cells)
    ambiguous['ambiguous'] = ambiguous['corr_gap'] <= ambiguous['corr_gap'].quantile(0.25)
    ambiguous.to_csv(os.path.join(out, 'ambiguous_by_similarity.csv'))
    print(f'wrote baselines for {len(cells)} progenitors to {out}')


if __name__ == '__main__':
    main()
