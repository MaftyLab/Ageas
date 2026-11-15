#!/usr/bin/env python3
"""Single source of truth for lineage-tracing ground truth.

Every fate-inference method in the benchmark must be scored against the same
label vector for the same cells. Earlier scripts each re-derived labels
inline, with conflicting orientations of ``NeuMon_fate_bias`` (CoSpar mapped
``> 0.999`` to Neutrophil, CellRank to Monocyte) and with mixed-fate clones
forced into one class. This module derives labels directly from the clone
matrix, so the orientation is unambiguous, and keeps mixed clones as their
own category.

Typical use::

    from ground_truth import clonal_fate_table
    gt = clonal_fate_table(
        adata, fates=('Monocyte', 'Neutrophil'),
        progenitor_mask=adata.obs['state_info'] == 'undiff',
        clone_obsm='X_clone', state_key='state_info', time_key='time_info',
    )
    gt.to_csv('larry_ground_truth.csv')
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

LABEL_MIXED = 'mixed'
LABEL_NO_INFO = 'no_info'


def clonal_fate_table(
    adata,
    fates: tuple,
    progenitor_mask,
    clone_obsm: str = 'X_clone',
    state_key: str = 'state_info',
    time_key: str = None,
    later_only: bool = True,
    pure_threshold: float = 1.0,
    min_descendants: int = 1,
) -> pd.DataFrame:
    """Per-progenitor clonal fate counts, fractions and labels.

    For each progenitor cell, all clones it belongs to are collected, and
    the clonally related cells annotated as each terminal fate are counted
    (excluding the progenitor itself). With ``time_key`` and
    ``later_only=True`` only relatives sampled strictly later than the
    progenitor count, which matches the usual LARRY/CoSpar definition.

    Args:
        adata: AnnData holding a cell-by-clone matrix in ``obsm``.
        fates: The two (or more) terminal fate names in ``obs[state_key]``.
            The first fate is ``A`` in ``frac_<A>``.
        progenitor_mask: Boolean mask (array or Series) of query progenitors.
        clone_obsm: ``obsm`` key of the cell-by-clone matrix.
        state_key: ``obs`` column with cell states.
        time_key: Optional ``obs`` column with (numeric or orderable)
            sampling times.
        later_only: Only count relatives sampled later than the progenitor.
        pure_threshold: Minimum fraction of one fate for a unipotent label;
            ``1.0`` requires all relatives to share the fate.
        min_descendants: Minimum number of fate-annotated relatives for a
            progenitor to receive a label (otherwise ``no_info``).

    Returns:
        Table indexed by progenitor cell with columns ``n_<fate>`` per fate,
        ``n_total``, ``frac_<fate>`` per fate and ``label`` (a fate name,
        ``mixed`` or ``no_info``).
    """
    fates = tuple(fates)
    clones = adata.obsm[clone_obsm]
    clones = sparse.csr_matrix(clones) if not sparse.issparse(clones) else clones.tocsr()
    clones = (clones > 0).astype(np.int64)

    states = adata.obs[state_key].astype(str).values
    mask = np.asarray(progenitor_mask, dtype=bool)
    prog_idx = np.flatnonzero(mask)

    times = adata.obs[time_key].values if time_key is not None else None
    counts = np.zeros((len(prog_idx), len(fates)), dtype=np.int64)

    groups = (
        [(t, prog_idx[times[prog_idx] == t]) for t in pd.unique(times[prog_idx])]
        if times is not None and later_only
        else [(None, prog_idx)]
    )
    position = {cell: k for k, cell in enumerate(prog_idx)}
    for t, cells in groups:
        for j, fate in enumerate(fates):
            relatives = states == fate
            if t is not None:
                relatives &= times > t
            # clone_sizes[c] = number of fate cells (sampled later) in clone c
            clone_sizes = np.asarray(clones[relatives].sum(axis=0)).ravel()
            per_cell = clones[cells] @ clone_sizes
            if not later_only or t is None:
                # remove the progenitor itself if it carries that fate label
                per_cell = per_cell - (states[cells] == fate) * clones[cells].sum(axis=1).A1
            for cell, value in zip(cells, per_cell):
                counts[position[cell], j] = value

    table = pd.DataFrame(
        counts, index=adata.obs_names[prog_idx], columns=[f'n_{f}' for f in fates]
    )
    table['n_total'] = counts.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        for j, fate in enumerate(fates):
            table[f'frac_{fate}'] = counts[:, j] / table['n_total'].values

    labels = np.full(len(table), LABEL_MIXED, dtype=object)
    for fate in fates:
        labels[table[f'frac_{fate}'].values >= pure_threshold] = fate
    labels[table['n_total'].values < min_descendants] = LABEL_NO_INFO
    table['label'] = labels
    return table


def labels_from_obs(adata, key: str, mapping: dict = None, mask=None) -> pd.DataFrame:
    """Ground-truth table from an existing ``obs`` label column.

    Args:
        adata: AnnData with the label column.
        key: ``obs`` column holding lineage-tracing labels.
        mapping: Optional mapping from raw labels to fate names (e.g.
            ``{'Early-Repr': 'reprogramming'}``); unmapped labels become
            ``no_info``.
        mask: Optional boolean mask restricting the cells.

    Returns:
        Table indexed by cell with a ``label`` column.
    """
    obs = adata.obs if mask is None else adata.obs[np.asarray(mask, dtype=bool)]
    raw = obs[key].astype(str)
    labels = raw.map(mapping).fillna(LABEL_NO_INFO) if mapping else raw
    return pd.DataFrame({'label': labels.values}, index=obs.index)


def check_bias_orientation(table: pd.DataFrame, bias: pd.Series, fate: str) -> float:
    """Spearman correlation between a precomputed bias column and ``frac_<fate>``.

    Use this to establish what a column such as LARRY's ``NeuMon_fate_bias``
    measures: a correlation near +1 means it is the fraction of ``fate``.

    Args:
        table: Output of :func:`clonal_fate_table`.
        bias: Precomputed bias values indexed by cell.
        fate: Fate whose fraction is compared.

    Returns:
        Spearman correlation over cells with information.
    """
    frac = table[f'frac_{fate}']
    common = frac.dropna().index.intersection(bias.dropna().index)
    return float(spearmanr(frac.loc[common], bias.loc[common]).correlation)


def binary_eval_view(table: pd.DataFrame, fates: tuple, include_mixed: bool = False) -> pd.DataFrame:
    """Rows usable for binary evaluation.

    Args:
        table: Output of :func:`clonal_fate_table` or :func:`labels_from_obs`.
        fates: The two fates of the binary task.
        include_mixed: If ``True``, mixed clones are kept and labelled by
            their majority fate (ties dropped); otherwise only unipotent
            progenitors are returned.

    Returns:
        Subset of ``table`` with a ``binary_label`` column.
    """
    out = table[table['label'].isin(fates)].copy()
    out['binary_label'] = out['label']
    if include_mixed and 'n_total' in table:
        mixed = table[table['label'] == LABEL_MIXED].copy()
        a, b = fates
        majority = np.where(
            mixed[f'n_{a}'] > mixed[f'n_{b}'], a,
            np.where(mixed[f'n_{b}'] > mixed[f'n_{a}'], b, None),
        )
        mixed['binary_label'] = majority
        out = pd.concat([out, mixed[mixed['binary_label'].notna()]])
    return out
