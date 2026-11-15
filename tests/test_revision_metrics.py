"""Statistics used for the revision benchmarks (R2.5, R2.3, R2.8)."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revision.common import apply_selection_gates  # noqa: E402
from revision.common.metrics import (  # noqa: E402
    bootstrap_ci, delong_test, expected_calibration_error, holm, paired_bootstrap,
)


def _data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p_good = np.clip(0.5 + 0.35 * (2 * y - 1) + rng.normal(0, 0.2, n), 0, 1)
    p_bad = np.clip(0.5 + 0.05 * (2 * y - 1) + rng.normal(0, 0.2, n), 0, 1)
    return y, p_good, p_bad


def test_delong_matches_sklearn_and_detects_difference():
    y, good, bad = _data()
    res = delong_test(y, good, bad)
    assert abs(res["auc_a"] - roc_auc_score(y, good)) < 1e-9
    assert abs(res["auc_b"] - roc_auc_score(y, bad)) < 1e-9
    assert res["p"] < 1e-6
    assert delong_test(y, good, good)["p"] == 1.0


def test_bootstrap_ci_brackets_estimate():
    y, good, _ = _data()
    est, low, high = bootstrap_ci(y, good, "auroc", n_boot=200)
    assert low <= est <= high


def test_paired_bootstrap_sign():
    y, good, bad = _data()
    res = paired_bootstrap(y, good, bad, "accuracy", n_boot=200)
    assert res["diff"] > 0 and res["low"] > 0


def test_calibration_and_holm():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_offline_gates_match_rules():
    oof = pd.Series({"a": 0.95, "b": 0.85, "c": 0.75, "d": 0.6})
    assert apply_selection_gates(oof, 0.25, 0.9, 0.7) == ["a"]
    assert apply_selection_gates(oof, 0.5, 0.9, 0.7) == ["a", "b"]
    assert apply_selection_gates(oof, 0.0, 1.0, 0.7, relative_retention=0.15) == ["a", "b"]
    assert apply_selection_gates(oof, 0.5, 1.0, 0.99) == ["a"]
