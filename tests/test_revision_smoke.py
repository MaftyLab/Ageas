"""Smoke test: run the revision experiment scripts end to end on synthetic data.

Checks the plumbing (registry -> preparation -> Ageas -> outputs -> scoring)
of experiments E01-E15 with a tiny classical model panel. It does not check
scientific results; those come from the real datasets.
"""
import importlib
import json
import os

import pandas as pd
import pytest

from revision.common.synthetic import make_synthetic


def _script(name):
    return importlib.import_module(f"revision.{name}")


@pytest.fixture(scope="module")
def env(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("revision")
    h5ad = root / "synthetic.h5ad"
    make_synthetic(n_genes=30, n_clones=80, seed=0).write_h5ad(h5ad)

    example = json.load(open(os.path.join(os.path.dirname(__file__), "..", "revision",
                                          "datasets.example.json")))
    registry = {"synthetic": dict(example["synthetic"], h5ad=str(h5ad))}
    reg_path = root / "datasets.json"
    reg_path.write_text(json.dumps(registry))

    protocol = json.load(open(os.path.join(os.path.dirname(__file__), "..", "revision", "protocol.json")))
    protocol["seeds"] = [0, 1]
    protocol["bootstrap"] = 50
    protocol["selection"].update(kfold_selection_list=[2], retention_point=0.6, cutoff_point=0.5)
    protocol["extraction"].update(max_extraction_iter=2, outlier_iqr_factor=1.5)
    proto_path = root / "protocol.json"
    proto_path.write_text(json.dumps(protocol))

    config = root / "configs"
    for model_type, name, params in [
        ("logreg", "lr", {"penalty": "l2", "dual": False, "tol": 1e-4, "C": 1.0, "fit_intercept": True,
                          "intercept_scaling": 1.0, "class_weight": None, "random_state": 0, "solver": "lbfgs"}),
        ("logreg", "lr_weak", {"penalty": "l2", "dual": False, "tol": 1e-4, "C": 0.01, "fit_intercept": True,
                               "intercept_scaling": 1.0, "class_weight": None, "random_state": 0,
                               "solver": "lbfgs"}),
        ("svc", "svc", {"scaler_copy": True, "scaler_with_mean": True, "scaler_with_std": True, "C": 1.0,
                        "kernel": "linear", "degree": 3, "gamma": "scale", "coef0": 0.0, "shrinking": True,
                        "tol": 1e-3, "cache_size": 200, "class_weight": None, "verbose": False,
                        "max_iter": -1, "decision_func_shape": "ovr", "break_ties": False, "random_state": 0}),
    ]:
        (config / model_type).mkdir(parents=True, exist_ok=True)
        (config / model_type / f"{name}.json").write_text(
            json.dumps({"max_epochs": 1, "dataloader_workers": 0, "model_params": params}))

    common = ["--dataset", "synthetic", "--registry", str(reg_path), "--protocol", str(proto_path),
              "--config", str(config), "--results-root", str(root / "results")]
    return {"root": root, "args": common, "results": root / "results"}


def _read(env, exp, name):
    return pd.read_csv(env["results"] / exp / "synthetic" / name)


def test_e01_and_e02(env):
    _script("E01_benchmark_v2.run_ageas").main(env["args"])
    _script("E02_baselines.reference_mapping").main(env["args"])
    knn = env["results"] / "E02" / "synthetic" / "knn15.csv"
    _script("E01_benchmark_v2.score").main(env["args"] + ["--method", f"kNN15={knn}"])
    metrics = _read(env, "E01", "metrics.csv")
    assert set(metrics["method"]) == {"Ageas", "kNN15"}
    assert metrics["auroc"].between(0, 1).all()
    assert (_read(env, "E01", "comparisons.csv")["p_holm"] <= 1).all()

    amb = env["results"] / "E02" / "synthetic" / "ambiguous_by_similarity.csv"
    _script("E01_benchmark_v2.score").main(env["args"] + ["--method", f"kNN15={knn}",
                                                          "--subset", str(amb), "--tag", "_amb"])
    assert _read(env, "E01", "metrics_amb.csv")["n"].iloc[0] < metrics["n"].iloc[0]


def test_e03_e04_e09(env):
    _script("E03_ablation.train_full_panel").main(env["args"] + ["--explain"])
    _script("E03_ablation.ladder").main(env["args"])
    ladder = pd.read_csv(env["results"] / "E03" / "synthetic" / "ladder.csv", index_col=0)
    assert {"all_configs", "best_single", "ageas_protocol"} <= set(ladder.index)
    _script("E04_transfer_gap.transfer_gap").main(env["args"])
    assert "terminal_only_mass" in _read(env, "E04", "transfer_gap.csv")
    _script("E09_hparam_sensitivity.sensitivity_grid").main(env["args"])
    assert len(_read(env, "E09", "selection_grid.csv")) > 0


def test_e05_e06_e13(env):
    _script("E05_calibration.calibration").main(env["args"])
    assert "spearman_vs_fraction" in _read(env, "E05", "calibration_summary.csv")
    _script("E06_uncertainty.run_uncertainty").main(env["args"])
    cells = pd.read_csv(env["results"] / "E06" / "synthetic" / "cell_uncertainty.csv", index_col=0)
    assert {"epistemic", "is_ood", "state"} <= set(cells.columns)
    _script("E13_open_set.open_set").main(env["args"])
    assert "other_fate_progenitors" in set(_read(env, "E13", "open_set.csv")["test"])


def test_e07_e08_e15(env):
    _script("E07_attribution_stability.stability").main(env["args"] + ["--n-subsamples", "1", "--top-k", "5"])
    assert len(_read(env, "E07", "stability_summary.csv")) == 2
    _script("E08_masking_diagnostics.masking_diagnostics").main(env["args"])
    assert len(_read(env, "E08", "iterations.csv")) >= 1
    _script("E15_lineage_concordance.oracle_de").main(env["args"] + ["--top-k", "5"])
    assert {"precision", "recall"} <= set(_read(env, "E15", "concordance.csv").columns)


def test_e03_permutation_e12_e14_export(env):
    _script("E03_ablation.permutation_check").main(env["args"] + ["--n-perm", "1"])
    perm = pd.read_csv(env["results"] / "E03" / "synthetic" / "permutation_check.csv")
    assert perm["max_abs_diff"].iloc[0] < 1e-6   # classical panel: exactly order-invariant

    _script("E14_axolotl_temporal.axolotl_temporal").main(env["args"])
    assert (env["results"] / "E14" / "synthetic" / "axis_summary.csv").exists()

    _script("E02_baselines.export_for_r").main(env["args"])
    assert (env["results"] / "E02" / "synthetic" / "r_input" / "matrix.mtx").exists()

    _script("E12_per_class_metrics.run_ageas_cv").main(env["args"] + ["--folds", "2"])
    preds = pd.read_csv(env["results"] / "E12" / "synthetic" / "predictions_ageas.csv")
    baseline = preds.assign(method="majority", pred=preds["true"].mode()[0])
    baseline.to_csv(env["results"] / "E12" / "synthetic" / "predictions_majority.csv", index=False)
    _script("E12_per_class_metrics.per_class").main(env["args"])
    assert (env["results"] / "E12" / "synthetic" / "delta_vs_best_baseline.csv").exists()


def test_e10_and_e11(env):
    _script("E10_embryo_deconfounding.deconfound").main(env["args"] + ["--n-perm", "20"])
    summary = pd.read_csv(env["results"] / "E10" / "synthetic" / "summary.csv", index_col=0)["value"]
    for key in ("baseline_pearson_r", "position_matched_pearson_r", "share_of_variance_within_position",
                "no_technical_genes_pearson_r", "position_control_vs_ageas_spearman"):
        assert key in summary.index
    _script("E11_progenitor_separability.separability").main(
        [a for a in env["args"] if a not in ("--dataset", "synthetic")] + ["--datasets", "synthetic"])
    assert (env["results"] / "E11" / "separability.csv").exists()
