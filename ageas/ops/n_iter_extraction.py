#!/usr/bin/env python3
"""N-iteration extraction of top regulatory factors.

Wraps :func:`~ageas.n_kfold_selection` and feeds its surviving deck through
:meth:`~ageas.Deck.debrief` to obtain per-class explanation scores.
Outliers are pruned between iterations and the per-class top-N factors are
ranked across iterations to yield the final regulatory factor table.
"""
import logging
from warnings import warn

import numpy as np
import pandas as pd
from scipy import sparse

from ageas.hangar import Hangar
from ageas.tool import l1_normalize
from .n_kfold_selection import main as n_kfold_selection

_logger = logging.getLogger(__name__)


def main(
    hangar: Hangar,
    operation_name: str = 'extraction',
    accelerator: str = 'cpu',
    cuda_devices: list = None,
    query_dataset=None,
    test_dataset=None,
    exp_dataset=None,
    max_extraction_iter: int = 1,
    extract_top_n: int = 10,
    use_gene_names: bool = True,
    remove_outliers: str = 'upper',
    outlier_iqr_factor: float = 10,
    correlation_masking: float = None,
    return_history: bool = False,
    seed: int = 42,
    verbose: bool = None,
    selection_args: dict = None,
    explain_args: dict = None,
) -> tuple:
    """N-iteration extraction of top regulatory factors per class.

    Each iteration runs :func:`~ageas.n_kfold_selection`, debriefs the
    surviving deck to obtain per-class explanation scores, masks outlier
    features, and feeds the trimmed dataset into the next iteration. Each
    iteration's explanation is L1-normalised per class before the
    iterations are summed, so iterations contribute equally regardless of
    how many units survived; the top-N factors per class are then ranked.

    Features revealed in later iterations may be independent signals or
    correlated substitutes for features masked earlier. Setting
    ``correlation_masking`` also masks every feature whose absolute Pearson
    correlation with a masked outlier reaches that value, so later
    iterations are not dominated by such surrogates.

    :param hangar: Source hangar from which the operating squad is generated.
    :param operation_name: Operation name under which per-iteration reports
        are stored.
    :param accelerator: Accelerator hint forwarded to selection (``'cpu'``
        or ``'cuda'``).
    :param cuda_devices: Optional GPU device indices.
    :param query_dataset: Required dataset that drives selection and (by
        default) the explanation pass.
    :param test_dataset: Optional held-out test dataset.
    :param exp_dataset: Optional explanation dataset. Defaults to
        ``test_dataset`` when provided, otherwise ``query_dataset``. It is
        copied, so the caller's object is not modified.
    :param max_extraction_iter: Number of extraction iterations (``N``).
    :param extract_top_n: Number of top factors to retain per class in the
        final ranking.
    :param use_gene_names: If ``True``, the returned table uses
        ``adata.var['name']`` as the index instead of feature IDs.
    :param remove_outliers: Tail masked between iterations: ``'upper'``,
        ``'lower'``, or a falsy value to disable masking.
    :param outlier_iqr_factor: IQR multiplier ``alpha`` of the outlier
        threshold ``Q3 + alpha * IQR``.
    :param correlation_masking: Optional absolute-correlation threshold for
        also masking features correlated with the outliers (e.g. ``0.6``).
    :param return_history: If ``True``, also return a per-iteration history
        (units, out-of-fold scores, masked features, explanations).
    :param seed: Random seed forwarded to selection.
    :param verbose: If ``True``, emit per-iteration progress logs.
    :param selection_args: Extra keyword arguments forwarded to
        :func:`~ageas.n_kfold_selection`.
    :param explain_args: Extra keyword arguments forwarded to
        :meth:`~ageas.Deck.debrief`.
    :returns: ``(top_factors, final_exp)``, or
        ``(top_factors, final_exp, history)`` with ``return_history=True``.
        ``top_factors`` is the per-class ranked factor table (with an
        ``Outlier_Iter`` column flagging masked features), and
        ``final_exp`` is the L1-normalised integrated explanation table.
    """
    if selection_args is None:
        selection_args = {}
    if explain_args is None:
        explain_args = {}

    assert query_dataset is not None, \
        "Query dataset must be provided for extraction."

    temp_query = query_dataset.copy()
    temp_test = test_dataset.copy() if test_dataset is not None else None
    if exp_dataset is None:
        exp_dataset = temp_test if temp_test is not None else temp_query
    else:
        exp_dataset = exp_dataset.copy()

    answers = []
    history = []
    hold_out = {}

    for i in range(max_extraction_iter):
        n_features_in = temp_query.adata.n_vars
        deck = n_kfold_selection(
            hangar=hangar,
            operation_name=f'{operation_name}_{i+1}',
            accelerator=accelerator,
            cuda_devices=cuda_devices,
            query_dataset=temp_query,
            test_dataset=temp_test,
            seed=seed,
            verbose=verbose,
            **selection_args,
        )
        _logger.info("Iteration %d — selected %d units.", i + 1, len(deck.squad))

        if len(deck.squad) == 0:
            _logger.warning("No units survived. Stopping extraction.")
            break

        integrated_exps = deck.debrief(
            operation=f'{operation_name}_{i+1}',
            exp_dataset=exp_dataset,
            verbose=verbose,
            **explain_args,
        )
        if integrated_exps is None:
            warn(f"No valid explanation for iteration {i+1}. Skipping it.")
            continue

        integrated_exps = integrated_exps.apply(
            lambda col: l1_normalize(col.values, mode='numpy')
        )
        answers.append(integrated_exps)

        n_masked_before = len(hold_out)
        temp_query, hold_out, masked, correlated = _find_outliers(
            integrated_exps,
            hold_out,
            i,
            temp_query,
            remove_outliers=remove_outliers,
            outlier_iqr_factor=outlier_iqr_factor,
            correlation_masking=correlation_masking,
        )
        history.append({
            'iteration': i + 1,
            'n_features_in': n_features_in,
            'n_units': len(deck.squad),
            'units': list(deck.squad),
            'oof': {
                unit_id: unit.report[f'{operation_name}_{i+1}']
                .get('final', {}).get('oof')
                for unit_id, unit in deck.squad.items()
            },
            'masked_outliers': masked,
            'masked_correlated': correlated,
            'n_masked_total': len(hold_out),
            'explanation': integrated_exps,
        })
        _logger.info(
            "Iteration %d — masked %d features (%d correlated).",
            i + 1, len(hold_out) - n_masked_before, len(correlated),
        )

        if temp_test is not None:
            temp_test.adata = temp_test.adata[:, temp_query.adata.var.index]
        exp_dataset.adata = exp_dataset.adata[:, temp_query.adata.var.index]

        if len(temp_query.adata.var.index) == 0:
            warn("No features left in the query dataset. Stopping extraction.")
            break

    if not answers:
        raise RuntimeError("Extraction produced no valid explanation.")

    # Sum the per-iteration (already L1-normalised) explanations over the
    # features that survived every iteration, then re-normalise.
    final_exp = answers[-1].copy() * 0.0
    for c in temp_query.label_dict:
        col = f"Class_{c}_Scores"
        for exp_df in answers:
            final_exp[col] += exp_df.loc[final_exp.index, col]
        final_exp[col] = l1_normalize(final_exp[col], mode='numpy')

    top_factors = pd.DataFrame(columns=final_exp.columns)
    for col in final_exp.columns:
        head = final_exp[col].sort_values(ascending=False).head(extract_top_n)
        for rank, gene in enumerate(head.index):
            if gene not in top_factors.index:
                top_factors.loc[gene, :] = 0
            top_factors.loc[gene, col] += len(head) - rank

    def _rename_col(df):
        l_map = temp_query.label_dict
        return df.rename(
            columns={
                col: f"Pro_{l_map[int(col.split('_')[1])]}_Scores"
                for col in df.columns
            }
        )

    top_factors = _rename_col(top_factors)
    final_exp = _rename_col(final_exp)

    top_factors['Outlier_Iter'] = -1
    for outlier in hold_out:
        top_factors.loc[outlier, :] = hold_out[outlier]

    if use_gene_names:
        names = query_dataset.adata.var
        assert 'name' in names.columns, \
            "Gene names are not available in the query dataset."
        top_factors.index = [names.loc[gene, 'name'] for gene in top_factors.index]

    if return_history:
        return top_factors, final_exp, history
    return top_factors, final_exp


def _correlated_features(
    adata,
    seeds: list,
    threshold: float,
    max_cells: int = 5000,
    chunk_size: int = 2000,
    random_state: int = 0,
) -> list:
    """Features whose absolute Pearson correlation with any seed reaches ``threshold``.

    Correlations are computed on at most ``max_cells`` randomly sampled
    cells, in feature chunks, so large peak matrices stay tractable.
    """
    if not seeds:
        return []
    rng = np.random.default_rng(random_state)
    n_obs = adata.n_obs
    rows = (
        np.sort(rng.choice(n_obs, max_cells, replace=False))
        if n_obs > max_cells else np.arange(n_obs)
    )
    matrix = adata.X[rows]

    def _standardize(block):
        block = block.toarray() if sparse.issparse(block) else np.asarray(block)
        block = block.astype(np.float64)
        block -= block.mean(axis=0)
        std = block.std(axis=0)
        std[std == 0] = np.inf
        return block / std

    var_index = adata.var.index
    seed_cols = var_index.get_indexer(seeds)
    seed_z = _standardize(matrix[:, seed_cols])

    hits = []
    for start in range(0, adata.n_vars, chunk_size):
        stop = min(start + chunk_size, adata.n_vars)
        corr = _standardize(matrix[:, start:stop]).T @ seed_z / len(rows)
        mask = np.max(np.abs(corr), axis=1) >= threshold
        hits.extend(var_index[start:stop][mask].tolist())
    return [f for f in hits if f not in set(seeds)]


def _find_outliers(
    exp_df: pd.DataFrame,
    hold_out: dict,
    extraction_iter: int,
    query_dset,
    remove_outliers: str = 'upper',
    outlier_iqr_factor: float = 10,
    correlation_masking: float = None,
) -> tuple:
    """Detect IQR-based outlier features in an explanation dataframe.

    Outlier features are stored in ``hold_out`` together with their
    explanation scores and the iteration in which they were removed, then
    excluded from the next iteration's query dataset. With
    ``correlation_masking`` set, features correlated with the outliers are
    masked as well (recorded with their own scores).

    :param exp_df: Per-class explanation table from
        :meth:`~ageas.Deck.debrief`.
    :param hold_out: Running dict of held-out features (modified in place).
    :param extraction_iter: Current extraction iteration index.
    :param query_dset: Query dataset whose features are pruned.
    :param remove_outliers: Tail to remove: ``'upper'``, ``'lower'``, or a
        falsy value to disable outlier removal.
    :param outlier_iqr_factor: IQR multiplier defining the outlier threshold.
    :param correlation_masking: Optional absolute-correlation threshold.
    :returns: ``(query_dset, hold_out, outliers, correlated)`` — the query
        dataset with masked features removed, the updated hold-out dict, and
        the lists of outliers and correlated features masked this iteration.
    """
    if not remove_outliers or outlier_iqr_factor is None or exp_df is None:
        return query_dset, hold_out, [], []

    assert outlier_iqr_factor >= 0, "Outlier IQR factor must be non-negative."
    total_outliers = {}
    for col in exp_df.columns:
        q1, q3 = exp_df[col].quantile(0.25), exp_df[col].quantile(0.75)
        outlier_distance = outlier_iqr_factor * (q3 - q1)
        if remove_outliers.upper() == 'UPPER':
            outliers = exp_df[exp_df[col] >= q3 + outlier_distance].index.tolist()
        elif remove_outliers.upper() == 'LOWER':
            outliers = exp_df[exp_df[col] <= q1 - outlier_distance].index.tolist()
        else:
            outliers = []
        for outlier in outliers:
            total_outliers[outlier] = None

    outliers = list(total_outliers)
    correlated = []
    if correlation_masking is not None and outliers:
        correlated = _correlated_features(
            query_dset.adata, outliers, correlation_masking
        )

    for feature in outliers + correlated:
        if feature in exp_df.index:
            hold_out[feature] = (
                exp_df.loc[feature, :].values.tolist() + [extraction_iter]
            )

    to_mask = set(outliers) | set(correlated)
    if to_mask:
        query_dset.adata = query_dset.adata[
            :, ~query_dset.adata.var.index.isin(list(to_mask))
        ]
    return query_dset, hold_out, outliers, correlated
