#!/usr/bin/env python3
"""Tutorial 01 — Quickstart.

This script walks through a minimal Ageas workflow on synthetic data:
  1. Generate a synthetic AnnData object with informative and noise genes.
  2. Wrap it in a Multimodal_Corpus (the dataset object Ageas consumes).
  3. Load a model hangar from a config folder.
  4. Run n_kfold_selection to pick the best classifiers.
  5. Predict cell-type probabilities on the corpus.
  6. Call Deck.debrief() to obtain per-class feature importances.

The synthetic data has 2 classes, 20 features (2 informative, rest noise)
and 100 observations, so the whole script runs in under a minute on CPU.
"""
import shutil
import warnings

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import ageas
from ageas import Hangar, n_kfold_selection
from ageas.tool import Multimodal_Corpus, make_fake_adata

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Synthetic data
# ---------------------------------------------------------------------------
# make_fake_adata creates an AnnData with:
#   - n_class × n_clusters_per_class cell clusters stored in obs['celltype']
#   - a 'name' column in var (gene symbols)
#   - 16 noise features + 2 informative + 2 redundant (defaults)
adata = make_fake_adata(n_class=2, n_clusters_per_class=1)
print("Synthetic AnnData:\n", adata)

# Save so Multimodal_Corpus can memory-map or read it back.
# In a real workflow you would point this at your own .h5ad file.
adata_path = "ageas_tutorial_fake.h5ad"
adata.write_h5ad(adata_path)

# ---------------------------------------------------------------------------
# 2. Build corpus
# ---------------------------------------------------------------------------
# Multimodal_Corpus wraps an AnnData file and tracks the label → integer
# mapping needed by the classifiers.
corpus = Multimodal_Corpus(
    adata_path,
    label_key="celltype",
    backed=False,      # load fully into RAM (fine for small data)
)
print("\nLabel map:", corpus.label_dict)

# ---------------------------------------------------------------------------
# 3. Load hangar
# ---------------------------------------------------------------------------
# A Hangar reads a folder of JSON config files, one sub-folder per model
# family (logreg, svc, xgb, mlp, resnet, rnn).  The sample_panel bundled
# with the repo is a good starting point.
config_folder = "data/configs/sample_panel"
hangar = Hangar(config_folder)
print(f"Hangar loaded {len(hangar.units)} units")

# ---------------------------------------------------------------------------
# 4. Model selection
# ---------------------------------------------------------------------------
# n_kfold_selection runs k-fold cross-validation and keeps the models that
# clear the retention_point threshold.  We use skip_final=False so the
# survivors get retrained on the full dataset in a "last mission" pass.
deck = n_kfold_selection(
    hangar=hangar,
    query_dataset=corpus,
    test_dataset=corpus,     # use same data as held-out test (tutorial only)
    kfold_selection_list=[2],
    valid_fraction=0.1,
    monitor_metric="test.accuracy",
    retention_point=0.5,     # low bar — keeps most models for demonstration
    cutoff_point=0.0,
    seed=42,
    verbose=False,
)
shutil.rmtree("cache", ignore_errors=True)

print(f"\nSurviving units: {list(deck.squad.keys())}")

# ---------------------------------------------------------------------------
# 5. Predict
# ---------------------------------------------------------------------------
all_preds, all_labels = deck.predict(query_dataset=corpus)
acc = accuracy_score(all_labels, all_preds.argmax(axis=-1))
auroc = roc_auc_score(all_labels, all_preds[:, 1])
print(f"\nEnsemble accuracy : {acc:.4f}")
print(f"Ensemble AUROC    : {auroc:.4f}")

# ---------------------------------------------------------------------------
# 6. Feature importance (debrief)
# ---------------------------------------------------------------------------
# Deck.debrief() calls each surviving unit's explain() method, aggregates
# per-class scores, and returns a DataFrame indexed by feature name.
importance = deck.debrief(exp_dataset=corpus, verbose=False)
print("\nTop-5 features for Class 0:")
print(importance["Class_0_Scores"].sort_values(ascending=False).head(5))
