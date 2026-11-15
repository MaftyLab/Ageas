"""A8: one ground-truth table for every benchmarked method.

Supports R2.3/R2.5/R2.8: labels come from the clone matrix (unambiguous
orientation), mixed clones are kept as their own category, and the
continuous fate fraction is available for calibration analyses.
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark", "fate"))
from ground_truth import binary_eval_view, check_bias_orientation, clonal_fate_table  # noqa: E402


def _toy():
    # cells: p0..p3 progenitors (day 2); 6 descendants (day 4)
    obs = pd.DataFrame({
        "state": ["undiff"] * 4 + ["Mono", "Mono", "Neu", "Neu", "Neu", "Mono"],
        "day": [2] * 4 + [4] * 6,
    }, index=[f"p{i}" for i in range(4)] + [f"d{i}" for i in range(6)])
    # clones: c0 = {p0, d0, d1} (Mono); c1 = {p1, d2, d3} (Neu);
    #         c2 = {p2, d4, d5} (mixed); p3 has no relatives
    clone = np.zeros((10, 3), dtype=int)
    for cell, c in [(0, 0), (4, 0), (5, 0), (1, 1), (6, 1), (7, 1), (2, 2), (8, 2), (9, 2), (3, 2)]:
        clone[cell, c] = 1
    clone[3, 2] = 0
    adata = ad.AnnData(np.zeros((10, 1)), obs=obs)
    adata.obsm["X_clone"] = sparse.csr_matrix(clone)
    return adata


def test_counts_labels_and_fractions():
    adata = _toy()
    gt = clonal_fate_table(adata, fates=("Mono", "Neu"),
                           progenitor_mask=adata.obs["state"] == "undiff",
                           state_key="state", time_key="day")
    assert gt.loc["p0", "label"] == "Mono"
    assert gt.loc["p1", "label"] == "Neu"
    assert gt.loc["p2", "label"] == "mixed"
    assert gt.loc["p3", "label"] == "no_info"
    assert gt.loc["p2", "frac_Mono"] == 0.5
    assert list(gt.loc["p0", ["n_Mono", "n_Neu"]]) == [2, 0]


def test_later_only_excludes_same_day_relatives():
    adata = _toy()
    adata.obs.loc["d0", "day"] = 2  # same day as the progenitor
    gt = clonal_fate_table(adata, fates=("Mono", "Neu"),
                           progenitor_mask=adata.obs["state"] == "undiff",
                           state_key="state", time_key="day")
    assert gt.loc["p0", "n_Mono"] == 1


def test_binary_view_and_orientation():
    adata = _toy()
    gt = clonal_fate_table(adata, fates=("Mono", "Neu"),
                           progenitor_mask=adata.obs["state"] == "undiff",
                           state_key="state", time_key="day")
    strict = binary_eval_view(gt, ("Mono", "Neu"))
    assert set(strict.index) == {"p0", "p1"}
    with_mixed = binary_eval_view(gt, ("Mono", "Neu"), include_mixed=True)
    assert set(with_mixed.index) == {"p0", "p1"}  # p2 is a tie and is dropped

    bias = pd.Series({"p0": 0.0, "p1": 1.0, "p2": 0.5})
    assert check_bias_orientation(gt, bias, "Neu") > 0.99
