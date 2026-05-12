#!/usr/bin/env python3
"""N-iteration extraction of top regulatory factors.

Wraps :func:`~ageas.n_kfold_selection` and feeds its surviving deck through
:meth:`~ageas.Deck.debrief` to obtain per-class explanation scores.
Outliers are pruned between iterations and the per-class top-N factors are
ranked across iterations to yield the final regulatory factor table.
"""
import logging
from warnings import warn

import pandas as pd

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
    seed: int = 42,
    verbose: bool = None,
    selection_args: dict = None,
    explain_args: dict = None,
) -> tuple:
    """N-iteration extraction of top regulatory factors per class.

    Each iteration runs :func:`~ageas.n_kfold_selection`, debriefs the
    surviving deck to obtain per-class explanation scores, prunes
    outlier features, and feeds the trimmed dataset back into the next
    iteration. The per-iteration explanations are L1-aggregated and the
    top-N factors per class are ranked across iterations.

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
        ``test_dataset`` when provided, otherwise ``query_dataset``.
    :param max_extraction_iter: Number of extraction iterations.
    :param extract_top_n: Number of top factors to retain per class in the
        final ranking.
    :param use_gene_names: If ``True``, the returned table uses
        ``adata.var['name']`` as the index instead of feature IDs.
    :param seed: Random seed forwarded to selection.
    :param verbose: If ``True``, emit per-iteration progress logs.
    :param selection_args: Extra keyword arguments forwarded to
        :func:`~ageas.n_kfold_selection`.
    :param explain_args: Extra keyword arguments forwarded to
        :meth:`~ageas.Deck.debrief`.
    :returns: ``(top_factors, final_exp)``. ``top_factors`` is the per-class
        ranked factor table (with an ``Outlier_Iter`` column flagging
        held-out features), and ``final_exp`` is the L1-normalised
        integrated explanation table.
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

    answers = []
    hold_out = {}
    integrated_exps = None

    for i in range(max_extraction_iter):
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

        assert exp_dataset is not None, "Explain dataset becomes None"

        temp_query, hold_out = _find_outliers(
            integrated_exps, hold_out, i, temp_query
        )
        if temp_test is not None:
            temp_test.adata = temp_test.adata[:, temp_query.adata.var.index]
        exp_dataset.adata = exp_dataset.adata[:, temp_query.adata.var.index]

        if len(temp_query.adata.var.index) == 0:
            warn("No features left in the query dataset. Stopping extraction.")
            break

        answers.append(integrated_exps)

    final_exp = integrated_exps.copy() * 0.0
    for c in temp_query.label_dict:
        for exp_df in answers:
            iter_df = exp_df.loc[final_exp.index, f"Class_{c}_Scores"]
            final_exp[f"Class_{c}_Scores"] += iter_df
        final_exp[f"Class_{c}_Scores"] = l1_normalize(
            final_exp[f"Class_{c}_Scores"],
            mode='numpy',
        )

    top_factors = pd.DataFrame(columns=final_exp.columns)
    for col in final_exp.columns:
        head = final_exp[col].sort_values(ascending=False).head(extract_top_n)
        class_tops = pd.DataFrame(
            range(len(head), 0, -1),
            index=head.index,
        )
        for gene in class_tops.index:
            if gene not in top_factors:
                top_factors.loc[gene, col] = 0
            top_factors.loc[gene, col] += class_tops.loc[gene, 0]

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
        assert 'name' in temp_query.adata.var.columns, \
            "Gene names are not available in the query dataset."
        top_factors.index = [
            temp_query.adata.var.loc[gene, 'name']
            for gene in top_factors.index
        ]
    return top_factors, final_exp


def _find_outliers(
    exp_df: pd.DataFrame,
    hold_out: dict,
    extraction_iter: int,
    query_dset,
    remove_outliers: str = 'upper',
    outlier_iqr_factor: int = 10,
) -> tuple:
    """Detect IQR-based outlier features in an explanation dataframe.

    Outlier features are stored in ``hold_out`` together with their
    explanation scores and the iteration in which they were removed, then
    excluded from the next iteration's query dataset.

    :param exp_df: Per-class explanation table from
        :meth:`~ageas.Deck.debrief`.
    :param hold_out: Running dict of held-out features (modified in place).
    :param extraction_iter: Current extraction iteration index.
    :param query_dset: Query dataset whose features are pruned.
    :param remove_outliers: Tail to remove: ``'upper'``, ``'lower'``, or a
        falsy value to disable outlier removal.
    :param outlier_iqr_factor: IQR multiplier defining the outlier threshold.
    :returns: ``(query_dset, hold_out)`` — the query dataset with outlier
        features removed and the updated hold-out dict.
    """
    if not remove_outliers or outlier_iqr_factor is None or exp_df is None:
        return query_dset, hold_out

    assert outlier_iqr_factor >= 0, "Outlier IQR factor must be non-negative."
    total_outliers = {}
    for col in exp_df.columns:
        iqr = exp_df[col].quantile(0.75) - exp_df[col].quantile(0.25)
        outlier_distance = outlier_iqr_factor * iqr
        if remove_outliers.upper() == 'UPPER':
            outliers = exp_df[
                exp_df[col] >= exp_df[col].quantile(0.75) + outlier_distance
            ].index.tolist()
        elif remove_outliers.upper() == 'LOWER':
            outliers = exp_df[
                exp_df[col] <= exp_df[col].quantile(0.25) - outlier_distance
            ].index.tolist()
        else:
            outliers = []

        for outlier in outliers:
            hold_out[outlier] = (
                exp_df.loc[outlier, :].values.tolist() + [extraction_iter]
            )
            total_outliers[outlier] = None

    if total_outliers:
        query_dset.adata = query_dset.adata[
            :, ~query_dset.adata.var.index.isin(list(total_outliers.keys()))
        ]
    return query_dset, hold_out
