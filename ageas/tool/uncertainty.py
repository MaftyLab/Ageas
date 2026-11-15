#!/usr/bin/env python3
"""Per-cell uncertainty and out-of-distribution scores for fate predictions.

A class posterior ``P(fate | x)`` near 0.5 can arise from cells on the
decision boundary, cells co-expressing both lineage programs (hybrid),
cells expressing neither (unprimed) or cells unlike any training cell
(out-of-distribution). This module provides the quantities needed to tell
these apart:

* :func:`decompose_uncertainty` splits the ensemble's predictive entropy
  into its expected member entropy (aleatoric part) and the mutual
  information between prediction and ensemble member (disagreement).
* :func:`ood_scores` measures how far each query cell lies from the
  terminal reference cells relative to the spacing among reference cells.
* :func:`program_scores` and :func:`program_thresholds` score lineage
  programs (e.g. the top Ageas features of each fate) per cell.
* :func:`call_fate_states` combines the three into one call per cell.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors

_EPS = 1e-12


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy (natural log) along ``axis``."""
    probs = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0)
    return -np.sum(probs * np.log(probs), axis=axis)


def decompose_uncertainty(member_probs, index=None) -> pd.DataFrame:
    """Decompose ensemble predictive uncertainty per cell.

    Args:
        member_probs: Per-member class probabilities, either an array of
            shape ``(n_members, n_cells, n_classes)`` or the ``member_preds``
            dict returned by ``Deck.predict(..., return_members=True)``.
        index: Optional cell index for the returned table.

    Returns:
        Table with columns ``total_entropy`` (``H[mean_m p_m]``),
        ``aleatoric`` (``mean_m H[p_m]``), ``epistemic`` (their difference,
        the mutual information), ``member_std`` (mean over classes of the
        across-member standard deviation) and ``max_prob``.
    """
    if isinstance(member_probs, dict):
        member_probs = np.stack(list(member_probs.values()))
    member_probs = np.asarray(member_probs, dtype=float)
    assert member_probs.ndim == 3, "Expected (n_members, n_cells, n_classes)."

    mean_probs = member_probs.mean(axis=0)
    total = entropy(mean_probs)
    aleatoric = entropy(member_probs).mean(axis=0)
    return pd.DataFrame(
        {
            'total_entropy': total,
            'aleatoric': aleatoric,
            'epistemic': np.clip(total - aleatoric, 0.0, None),
            'member_std': member_probs.std(axis=0).mean(axis=-1),
            'max_prob': mean_probs.max(axis=-1),
        },
        index=index,
    )


def _embed(reference, query, n_components: int, random_state: int):
    """Fit a linear embedding on ``reference`` and project both matrices."""
    n_components = int(min(n_components, reference.shape[1] - 1, reference.shape[0] - 1))
    if sparse.issparse(reference):
        model = TruncatedSVD(n_components=n_components, random_state=random_state)
    else:
        model = PCA(n_components=n_components, random_state=random_state)
    ref_emb = model.fit_transform(reference)
    query_emb = model.transform(query)
    return ref_emb, query_emb


def ood_scores(
    reference,
    query,
    n_neighbors: int = 15,
    n_components: int = 50,
    quantile: float = 0.99,
    query_index=None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Distance of query cells to the terminal reference population.

    Both matrices are projected with a PCA (dense) or truncated SVD
    (sparse) fitted on the reference cells. Each query cell's mean distance
    to its ``n_neighbors`` nearest reference cells is compared with the same
    statistic for reference cells (excluding themselves).

    Args:
        reference: Reference (terminal, training) feature matrix,
            ``(n_reference, n_features)``.
        query: Query feature matrix with the same features.
        n_neighbors: Number of nearest reference neighbours.
        n_components: Embedding dimension.
        quantile: Reference quantile above which a query cell is flagged
            out-of-distribution.
        query_index: Optional cell index for the returned table.
        random_state: Seed for the embedding.

    Returns:
        Table with ``knn_distance``, ``reference_quantile`` (fraction of
        reference cells with a smaller distance) and boolean ``is_ood``.
    """
    ref_emb, query_emb = _embed(reference, query, n_components, random_state)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(ref_emb)

    ref_dist, _ = nn.kneighbors(ref_emb, n_neighbors=n_neighbors + 1)
    ref_score = np.sort(ref_dist[:, 1:].mean(axis=1))

    query_dist, _ = nn.kneighbors(query_emb, n_neighbors=n_neighbors)
    query_score = query_dist.mean(axis=1)

    ref_quantile = np.searchsorted(ref_score, query_score, side='right') / len(ref_score)
    threshold = np.quantile(ref_score, quantile)
    return pd.DataFrame(
        {
            'knn_distance': query_score,
            'reference_quantile': ref_quantile,
            'is_ood': query_score > threshold,
        },
        index=query_index,
    )


def program_scores(
    matrix,
    feature_names,
    programs: dict,
    reference=None,
    index=None,
) -> pd.DataFrame:
    """Mean z-scored activity of each feature program per cell.

    Args:
        matrix: Cell-by-feature matrix to score.
        feature_names: Feature names of the matrix columns.
        programs: Mapping ``program name -> list of features``.
        reference: Optional matrix (same features) whose mean and standard
            deviation define the z-scores; defaults to ``matrix``.
        index: Optional cell index for the returned table.

    Returns:
        Table with one column per program.
    """
    def _dense(m):
        return m.toarray() if sparse.issparse(m) else np.asarray(m, dtype=float)

    feature_names = pd.Index(feature_names)
    reference = matrix if reference is None else reference
    out = {}
    for name, features in programs.items():
        cols = feature_names.get_indexer([f for f in features if f in feature_names])
        if len(cols) == 0:
            out[name] = np.full(matrix.shape[0], np.nan)
            continue
        ref = _dense(reference[:, cols])
        mean, std = ref.mean(axis=0), ref.std(axis=0)
        std[std == 0] = 1.0
        out[name] = ((_dense(matrix[:, cols]) - mean) / std).mean(axis=1)
    return pd.DataFrame(out, index=index)


def program_thresholds(reference_scores: pd.DataFrame, reference_labels) -> dict:
    """High/low activity thresholds for each program from terminal cells.

    A program named after a class is "high" at or above its median in that
    class's terminal cells and "low" at or below its median in the other
    terminal cells.

    Args:
        reference_scores: :func:`program_scores` output for terminal cells;
            column names must match class labels.
        reference_labels: Terminal class label per row of
            ``reference_scores``.

    Returns:
        Mapping ``program -> (low, high)``.
    """
    labels = np.asarray(reference_labels)
    thresholds = {}
    for program in reference_scores.columns:
        own = reference_scores.loc[labels == program, program]
        other = reference_scores.loc[labels != program, program]
        thresholds[program] = (float(np.median(other)), float(np.median(own)))
    return thresholds


def call_fate_states(
    probs: pd.DataFrame,
    uncertainty: pd.DataFrame,
    ood: pd.DataFrame = None,
    programs: pd.DataFrame = None,
    thresholds: dict = None,
    confident_prob: float = 0.7,
    max_epistemic: float = 0.05,
) -> pd.Series:
    """Assign each cell one interpretable fate state.

    Order of precedence: ``ood`` (if ``ood`` is given and flags the cell),
    ``model_disagreement`` (epistemic uncertainty above ``max_epistemic``),
    ``confident:<class>`` (top probability at least ``confident_prob``),
    then, for the remaining ambiguous cells and when two programs with
    thresholds are given, ``hybrid`` (both programs high), ``unprimed``
    (both low) or ``boundary`` (otherwise).

    Args:
        probs: Ensemble class probabilities, one column per class.
        uncertainty: Output of :func:`decompose_uncertainty` (same index).
        ood: Optional output of :func:`ood_scores` (same index).
        programs: Optional output of :func:`program_scores` with one column
            per class.
        thresholds: Output of :func:`program_thresholds` for ``programs``.
        confident_prob: Minimum top-class probability of a confident call.
        max_epistemic: Maximum mutual information of a confident call.

    Returns:
        Categorical series of fate states.
    """
    states = pd.Series('boundary', index=probs.index, dtype=object)
    top_class = probs.idxmax(axis=1)
    top_prob = probs.max(axis=1)

    ambiguous = top_prob < confident_prob
    if programs is not None and thresholds is not None and programs.shape[1] == 2:
        (name_a, name_b) = programs.columns
        low_a, high_a = thresholds[name_a]
        low_b, high_b = thresholds[name_b]
        hybrid = (programs[name_a] >= high_a) & (programs[name_b] >= high_b)
        unprimed = (programs[name_a] <= low_a) & (programs[name_b] <= low_b)
        states[ambiguous & hybrid] = 'hybrid'
        states[ambiguous & unprimed] = 'unprimed'

    confident = ~ambiguous
    states[confident] = 'confident:' + top_class[confident].astype(str)
    states[uncertainty['epistemic'] > max_epistemic] = 'model_disagreement'
    if ood is not None:
        states[ood['is_ood'].astype(bool)] = 'ood'
    return states.astype('category')


__all__ = [
    'entropy',
    'decompose_uncertainty',
    'ood_scores',
    'program_scores',
    'program_thresholds',
    'call_fate_states',
]
