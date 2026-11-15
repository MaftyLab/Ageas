"""Evaluation statistics shared by the revision experiments.

Binary fate metrics, bootstrap confidence intervals, DeLong's test for
paired AUROCs, paired bootstrap tests, calibration (ECE / Brier /
reliability bins) and Holm correction. Used for R2.5 (uncertainty of
performance estimates), R2.3/R2.8 (calibration, continuous ground truth) and
every benchmark table.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def binary_scores(y_true, p_pos, threshold: float = 0.5) -> dict:
    """Accuracy, macro F1, AUROC, Brier score and ECE for one binary task.

    Args:
        y_true: 0/1 labels (1 = positive fate).
        p_pos: Predicted probability of the positive fate.
        threshold: Decision threshold for accuracy and F1.
    """
    y_true = np.asarray(y_true, dtype=int)
    p_pos = np.asarray(p_pos, dtype=float)
    pred = (p_pos >= threshold).astype(int)
    out = {
        'n': len(y_true),
        'accuracy': accuracy_score(y_true, pred),
        'f1_macro': f1_score(y_true, pred, average='macro'),
        'brier': float(np.mean((p_pos - y_true) ** 2)),
        'ece': expected_calibration_error(y_true, p_pos),
    }
    out['auroc'] = roc_auc_score(y_true, p_pos) if len(np.unique(y_true)) == 2 else np.nan
    return out


def bootstrap_ci(y_true, p_pos, metric: str = 'auroc', n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile bootstrap CI of one metric over cells.

    Returns:
        ``(estimate, low, high)``.
    """
    y_true = np.asarray(y_true, dtype=int)
    p_pos = np.asarray(p_pos, dtype=float)
    rng = np.random.default_rng(seed)
    estimate = binary_scores(y_true, p_pos)[metric]
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if metric == 'auroc' and len(np.unique(y_true[idx])) < 2:
            continue
        values.append(binary_scores(y_true[idx], p_pos[idx])[metric])
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return estimate, float(low), float(high)


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_test(y_true, p_a, p_b) -> dict:
    """DeLong's test for two correlated AUROCs on the same cells.

    Implements the fast midrank algorithm of Sun & Xu (2014).

    Returns:
        Dict with ``auc_a``, ``auc_b``, ``diff`` (a - b), ``z`` and two-sided
        ``p``.
    """
    y_true = np.asarray(y_true, dtype=int)
    preds = np.vstack([np.asarray(p_a, float), np.asarray(p_b, float)])
    pos, neg = preds[:, y_true == 1], preds[:, y_true == 0]
    m, n = pos.shape[1], neg.shape[1]

    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {
        'auc_a': float(aucs[0]), 'auc_b': float(aucs[1]), 'diff': float(diff),
        'z': float(z), 'p': float(2 * stats.norm.sf(abs(z))),
    }


def paired_bootstrap(y_true, p_a, p_b, metric: str = 'f1_macro', n_boot: int = 1000,
                     seed: int = 0) -> dict:
    """Paired bootstrap of a metric difference between two methods.

    Returns:
        Dict with ``diff`` (a - b), its 95% ``low``/``high`` and a two-sided
        bootstrap ``p``.
    """
    y_true = np.asarray(y_true, dtype=int)
    p_a, p_b = np.asarray(p_a, float), np.asarray(p_b, float)
    rng = np.random.default_rng(seed)
    observed = binary_scores(y_true, p_a)[metric] - binary_scores(y_true, p_b)[metric]
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(binary_scores(y_true[idx], p_a[idx])[metric]
                     - binary_scores(y_true[idx], p_b[idx])[metric])
    diffs = np.asarray(diffs)
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        'diff': float(observed),
        'low': float(np.quantile(diffs, 0.025)),
        'high': float(np.quantile(diffs, 0.975)),
        'p': float(min(p, 1.0)),
    }


def reliability_table(y_true, p_pos, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width reliability bins: mean confidence vs observed frequency."""
    y_true = np.asarray(y_true, dtype=float)
    p_pos = np.asarray(p_pos, dtype=float)
    bins = np.clip((p_pos * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            rows.append({'bin': b, 'n': int(mask.sum()),
                         'mean_prob': float(p_pos[mask].mean()),
                         'observed': float(y_true[mask].mean())})
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, p_pos, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width bins."""
    table = reliability_table(y_true, p_pos, n_bins)
    if table.empty:
        return np.nan
    weights = table['n'] / table['n'].sum()
    return float(np.sum(weights * np.abs(table['mean_prob'] - table['observed'])))


def holm(pvalues) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def evaluation_frame(predictions: pd.DataFrame, ground_truth: pd.DataFrame, positive: str,
                     negative: str, label_col: str = 'label') -> pd.DataFrame:
    """Join predictions with unipotent ground-truth labels for one binary task.

    Returns:
        Frame with ``y`` (1 = ``positive``) and ``p`` (probability of
        ``positive``) for cells labelled ``positive`` or ``negative``.
    """
    gt = ground_truth[ground_truth[label_col].isin([positive, negative])]
    joined = gt.join(predictions[[f'prob_{positive}', f'prob_{negative}']], how='inner')
    total = joined[f'prob_{positive}'] + joined[f'prob_{negative}']
    return pd.DataFrame({
        'y': (joined[label_col] == positive).astype(int),
        'p': joined[f'prob_{positive}'] / total.replace(0, np.nan),
    }, index=joined.index).dropna()
