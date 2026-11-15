"""Shared fixtures for the Ageas regression tests."""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

LOGREG_PARAMS = {
    "penalty": "l2", "dual": False, "tol": 1e-4, "C": 1.0, "fit_intercept": True,
    "intercept_scaling": 1.0, "class_weight": None, "random_state": 0, "solver": "lbfgs",
}
SVC_PARAMS = {
    "scaler_copy": True, "scaler_with_mean": True, "scaler_with_std": True,
    "C": 1.0, "kernel": "linear", "degree": 3, "gamma": "scale", "coef0": 0.0,
    "shrinking": True, "tol": 1e-3, "cache_size": 200, "class_weight": None,
    "verbose": False, "max_iter": -1, "decision_func_shape": "ovr",
    "break_ties": False, "random_state": 0,
}
MLP_TRAIN_CONFIG = {
    "batch_size": 32, "loss_reduction": "mean", "optimizer": "adamw",
    "learning_rate": 0.01, "weight_decay": 0.0, "betas": [0.9, 0.999],
    "momentum": 0.9, "scheduler": None,
}


def marker_data(n_per_class=60, n_genes=12, n_classes=2, seed=0):
    """Gaussian noise plus one strong marker gene per class.

    Gene ``marker_c`` is shifted up in class ``c``; all other genes are noise.

    Returns:
        ``(x, y, names)`` with ``x`` shaped ``(n, 1, n_genes)`` like an
        Ageas batch.
    """
    rng = np.random.default_rng(seed)
    n = n_per_class * n_classes
    y = np.repeat(np.arange(n_classes), n_per_class)
    x = rng.normal(0.0, 1.0, size=(n, n_genes))
    names = [f"noise_{i}" for i in range(n_genes)]
    for c in range(n_classes):
        x[y == c, c] += 4.0
        names[c] = f"marker_{c}"
    return torch.tensor(x[:, None, :], dtype=torch.float32), torch.tensor(y), names


@pytest.fixture
def fake_corpus():
    """Two-class synthetic corpus with informative genes."""
    from ageas.tool import Multimodal_Corpus, make_fake_adata
    adata = make_fake_adata(n_cells=120, n_genes=12, n_informative=2,
                            n_redundant=2, n_class=2, n_clusters_per_class=1, seed=1)
    return Multimodal_Corpus(adata=adata, label_key="celltype", backed=False)


@pytest.fixture
def tiny_hangar_dir(tmp_path):
    """Config folder with two logistic regressions and one linear SVC."""
    def _write(model_type, name, config):
        folder = tmp_path / model_type
        folder.mkdir(exist_ok=True)
        with open(folder / f"{name}.json", "w") as fh:
            json.dump(config, fh)

    base = {"max_epochs": 1, "dataloader_workers": 0}
    _write("logreg", "lr_strong", dict(base, model_params=dict(LOGREG_PARAMS, C=1.0)))
    _write("logreg", "lr_weak", dict(base, model_params=dict(LOGREG_PARAMS, C=1e-4)))
    _write("svc", "svc", dict(base, model_params=dict(SVC_PARAMS)))
    return str(tmp_path)


@pytest.fixture
def mlp_hangar_dir(tmp_path):
    """Config folder with one small MLP (dropout on) and one logistic regression."""
    for model_type, name, config in [
        ("logreg", "lr", {"max_epochs": 1, "dataloader_workers": 0,
                          "model_params": dict(LOGREG_PARAMS)}),
        ("mlp", "mlp", {
            "max_epochs": 3, "dataloader_workers": 0,
            "model_params": {
                "block": "Residual_Encoder", "block_nums": [1], "block_dims": [8],
                "bias": True, "dropout": 0.5, "norm_layer": "LayerNorm", "latent_fea_dim": 16,
            },
            "train_config": dict(MLP_TRAIN_CONFIG),
        }),
    ]:
        folder = tmp_path / model_type
        folder.mkdir(exist_ok=True)
        with open(folder / f"{name}.json", "w") as fh:
            json.dump(config, fh)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Ageas v2 (feature-routed pipeline) helpers, plain values and functions that
# the v2 tests import directly.
# ---------------------------------------------------------------------------

#: The checkout whose ``ageas`` the tests import (the run record's ``ageas_path``).
AGEAS_ROOT = ROOT

MLP_PARAMS = {
    "block": "Residual_Encoder", "block_nums": [1], "block_dims": [8],
    "bias": True, "dropout": 0.0, "norm_layer": "LayerNorm", "latent_fea_dim": 16,
}

_BASE = {"max_epochs": 1, "dataloader_workers": 0}
#: Two logistic regressions (one nearly flat) and a linear SVC.
CLASSICAL_UNITS = {
    ("logreg", "lr_strong"): dict(_BASE, model_params=dict(LOGREG_PARAMS, C=1.0)),
    ("logreg", "lr_weak"): dict(_BASE, model_params=dict(LOGREG_PARAMS, C=1e-4)),
    ("svc", "svc"): dict(_BASE, model_params=dict(SVC_PARAMS)),
}
#: One small MLP.
MLP_UNITS = {
    ("mlp", "mlp"): {
        "max_epochs": 5, "dataloader_workers": 0,
        "model_params": dict(MLP_PARAMS), "train_config": dict(MLP_TRAIN_CONFIG),
    },
}


def write_hangar(folder, units):
    """Write ``{(model_type, name): config}`` as an Ageas config folder."""
    for (model_type, name), config in units.items():
        os.makedirs(os.path.join(folder, model_type), exist_ok=True)
        with open(os.path.join(folder, model_type, f"{name}.json"), "w") as fh:
            json.dump(config, fh)
    return str(folder)


def make_fake_corpus():
    """120 cells, 12 genes (2 informative, 2 redundant), two classes."""
    from ageas.tool import Multimodal_Corpus, make_fake_adata
    adata = make_fake_adata(n_cells=120, n_genes=12, n_informative=2,
                            n_redundant=2, n_class=2, n_clusters_per_class=1, seed=1)
    return Multimodal_Corpus(adata=adata, label_key="celltype", backed=False)
