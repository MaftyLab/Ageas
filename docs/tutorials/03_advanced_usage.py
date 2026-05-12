#!/usr/bin/env python3
"""Tutorial 03 — Advanced usage: boost selection, extraction, and custom configs.

This tutorial covers the two iterative pipelines that go beyond basic model
selection, plus how to craft and tune model configs from scratch.

Topics covered:
  - n_iter_boost_selection: iteratively narrow the feature set before final
    model selection (useful when starting with thousands of genes)
  - n_iter_extraction: identify top per-class regulatory factors across
    multiple rounds of selection + explanation
  - Writing custom JSON config files for any supported model family
  - Inspecting per-unit reports stored inside Deck.squad
"""
import json
import os
import shutil
import tempfile
import warnings

import pandas as pd

from ageas import Hangar, n_iter_boost_selection, n_iter_extraction
from ageas.tool import Multimodal_Corpus, make_fake_adata

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
# Use a slightly larger synthetic dataset so the iterative pipelines have
# something meaningful to prune.
adata = make_fake_adata(n_class=2, n_clusters_per_class=2)
adata_path = "ageas_tut03.h5ad"
adata.write_h5ad(adata_path)

corpus = Multimodal_Corpus(adata_path, label_key="celltype", backed=False)
print("Corpus:", len(corpus), "cells,", corpus.adata.n_vars, "features")


# ---------------------------------------------------------------------------
# 1. Custom model configs
# ---------------------------------------------------------------------------
# Each JSON config file in a sub-folder of the config directory corresponds
# to one Unit (one trained model instance).  The required keys are:
#
#   max_epochs    — number of training epochs (Lightning) or fitting passes
#   model_params  — constructor keyword arguments for the model class
#   train_config  — training-loop arguments (learning rate, optimizer, etc.)
#
# Sub-folder name determines the model family:
#   logreg / svc / xgb / mnb  → classical  (sklearn / xgboost)
#   mlp / resnet / rnn         → neural     (PyTorch Lightning)
#
# Let's build a minimal config folder programmatically:

config_dir = tempfile.mkdtemp(prefix="ageas_configs_")

os.makedirs(os.path.join(config_dir, "logreg"), exist_ok=True)
logreg_cfg = {
    "max_epochs": 1,
    "model_params": {
        "penalty": "l1",
        "C": 1.0,
        "solver": "saga",
        "tol": 1e-4,
    },
}
with open(os.path.join(config_dir, "logreg", "logreg_l1_C1.json"), "w") as f:
    json.dump(logreg_cfg, f, indent=4)

os.makedirs(os.path.join(config_dir, "xgb"), exist_ok=True)
xgb_cfg = {
    "max_epochs": 1,
    "model_params": {
        "booster": "gbtree",
        "max_depth": 4,
        "eta": 0.1,
    },
    "train_config": {
        "num_boost_round": 50,
        "early_stopping_rounds": None,
        "verbose_eval": False,
    },
}
with open(os.path.join(config_dir, "xgb", "xgb_fast.json"), "w") as f:
    json.dump(xgb_cfg, f, indent=4)

print("\nConfig dir:", config_dir)
print("Families:", os.listdir(config_dir))

hangar = Hangar(config_dir)
print(f"Units loaded: {sum(len(v) for v in hangar.units.values())}")


# ---------------------------------------------------------------------------
# 2. n_iter_boost_selection
# ---------------------------------------------------------------------------
# Boost selection iteratively:
#   a) runs n_kfold_selection on the current feature set,
#   b) debriefs the surviving models to get feature importance scores,
#   c) trims the dataset to the top extract_top_n / extract_ratio^(iter) genes,
#   d) repeats, then runs a final selection on the trimmed feature set.
#
# This is helpful when you start with many features (e.g. 2000 HVGs) and
# want to converge to a compact set of candidate regulatory factors before
# final model training.

deck, fea_kept = n_iter_boost_selection(
    hangar=hangar,
    query_dataset=corpus,
    test_dataset=corpus,
    max_boost_iter=2,
    extract_ratio=0.5,
    extract_top_n=8,          # target: keep 8 genes after boosting
    seed=42,
    verbose=False,
    selection_args=dict(
        kfold_selection_list=[2],
        valid_fraction=0.1,
        monitor_metric="test.accuracy",
        retention_point=0.5,
        cutoff_point=0.0,
    ),
)
shutil.rmtree("cache", ignore_errors=True)

print(f"\nBoost selection — features kept: {len(fea_kept)}")
print("Kept genes:", fea_kept[:5], "...")
print(f"Surviving units: {len(deck.squad)}")


# ---------------------------------------------------------------------------
# 3. n_iter_extraction
# ---------------------------------------------------------------------------
# n_iter_extraction is the full regulatory factor discovery pipeline:
#   a) runs n_kfold_selection → surviving deck,
#   b) calls deck.debrief() → per-class explanation scores,
#   c) prunes IQR-outlier features between iterations,
#   d) L1-aggregates explanations across iterations,
#   e) ranks the top extract_top_n factors per class.
#
# Returns (top_factors DataFrame, final_exp DataFrame).

top_factors, final_exp = n_iter_extraction(
    hangar=hangar,
    query_dataset=corpus,
    test_dataset=corpus,
    max_extraction_iter=2,
    extract_top_n=5,          # top 5 factors per class
    use_gene_names=True,      # use adata.var['name'] (gene symbols)
    seed=42,
    verbose=False,
    selection_args=dict(
        kfold_selection_list=[2],
        valid_fraction=0.1,
        monitor_metric="test.accuracy",
        retention_point=0.5,
        cutoff_point=0.0,
    ),
)
shutil.rmtree("cache", ignore_errors=True)

print("\nTop regulatory factors:")
print(top_factors)

print("\nFinal integrated explanation (first 5 rows):")
print(final_exp.head())


# ---------------------------------------------------------------------------
# 4. Inspecting per-unit reports
# ---------------------------------------------------------------------------
# After n_kfold_selection (called inside both pipelines above) each Unit in
# deck.squad stores a nested report dict:
#
#   unit.report[operation_name]['fold_1']    — per-fold metrics
#   unit.report[operation_name]['final']     — last-mission metrics
#
# Let's look at what's stored for the last deck produced by n_iter_extraction.
print("\n--- Per-unit report keys ---")
for uid, unit in deck.squad.items():
    print(f"  {uid}: operations = {list(unit.report.keys())}")

# Drill into the final-mission test metrics for each unit:
print("\n--- Final test accuracy per unit ---")
for uid, unit in deck.squad.items():
    for op_name, op_report in unit.report.items():
        if "final" not in op_report:
            continue
        final = op_report["final"]
        if final["test"] is not None:
            acc = final["test"].get("test.accuracy", "n/a")
        else:
            acc = final["vali"].get("vali.accuracy", "n/a")
        print(f"  {uid:30s} acc={acc}")


# ---------------------------------------------------------------------------
# 5. Saving / loading a trained model
# ---------------------------------------------------------------------------
# Each Unit wraps a model that can be saved and reloaded independently.
# The path convention is arbitrary — you choose it.
save_dir = tempfile.mkdtemp(prefix="ageas_saved_")
for uid, unit in deck.squad.items():
    model_path = os.path.join(save_dir, f"{uid}.model")
    unit.model.save_model(model_path)
    print(f"Saved {uid} → {model_path}")

# To reload: unit.model.load_model(model_path)

# Clean up temp directories
shutil.rmtree(config_dir, ignore_errors=True)
shutil.rmtree(save_dir, ignore_errors=True)
