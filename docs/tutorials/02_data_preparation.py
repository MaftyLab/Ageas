#!/usr/bin/env python3
"""Tutorial 02 — Data preparation and corpus construction.

Ageas consumes AnnData (.h5ad) files through its corpus classes.  This
tutorial explains the expected AnnData layout, demonstrates how to construct
and inspect a Multimodal_Corpus, and shows the built-in k-fold splitter.

Topics covered:
  - Required AnnData structure (obs labels, var gene names, X values)
  - Multimodal_Corpus vs Repr_Corpus
  - kfold_random_split: training / validation / test splits
  - Oversampling with 'repeat' to balance classes
  - Inspecting batches from a corpus DataLoader
"""
import warnings

import anndata as ad
import numpy as np
import pandas as pd

from ageas.tool import (
    Multimodal_Corpus,
    Repr_Corpus,
    kfold_random_split,
    make_fake_adata,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. AnnData layout required by Ageas
# ---------------------------------------------------------------------------
# Ageas expects an AnnData where:
#   adata.obs[label_key]  — cell-type / condition label (string or int)
#   adata.var['name']     — human-readable gene symbol (optional but needed
#                           for use_gene_names=True in n_iter_extraction)
#   adata.X               — non-negative expression matrix (cells × genes)
#                           already normalised / log-transformed
#
# The simplest way to create test data is make_fake_adata:
adata = make_fake_adata(n_class=2, n_clusters_per_class=2)
print("AnnData overview:")
print(adata)
print("\nFirst 5 obs labels:\n", adata.obs["celltype"].head())
print("\nVar table (first 5):\n", adata.var.head())

# ---------------------------------------------------------------------------
# 2. Build corpus from file or from an existing AnnData
# ---------------------------------------------------------------------------
# Multimodal_Corpus supports loading from file.  Back to disk first:
adata_path = "ageas_tut02.h5ad"
adata.write_h5ad(adata_path)

# backed=True uses memory-mapping (good for large datasets).
corpus = Multimodal_Corpus(adata_path, label_key="celltype", backed=False)
print("\nCorpus label_dict:", corpus.label_dict)
print("Number of cells  :", len(corpus))

# ---------------------------------------------------------------------------
# 3. Repr_Corpus: multiple representation layers
# ---------------------------------------------------------------------------
# Repr_Corpus can load additional layers (e.g. PCA embeddings) stored in
# adata.obsm.  Pass layers=['X', 'X_pca'] to use both.
#
# For this demo we only have 'X', so we use Multimodal_Corpus.
# When you have pre-computed embeddings:
#   corpus = Repr_Corpus(adata_path, label_key='celltype', layers=['X', 'X_pca'])

# ---------------------------------------------------------------------------
# 4. Accessing corpus internals
# ---------------------------------------------------------------------------
# A corpus is a PyTorch Dataset: corpus[i] returns (data_tensor, label_tensor).
data, label = corpus[0]
print(f"\nSingle sample — data shape: {data.shape}, label: {label}")

# The underlying AnnData is always accessible:
print("adata.X dtype:", corpus.adata.X.dtype)

# ---------------------------------------------------------------------------
# 5. K-fold splitting
# ---------------------------------------------------------------------------
# kfold_random_split produces balanced train / validation / test folds.
# It returns three lists (one entry per fold), each containing a sub-corpus.

train_list, valid_list, test_list = kfold_random_split(
    corpus,
    n_splits=3,            # 3-fold: each fold uses ~1/3 as test, rest as train
    valid_fraction=0.1,    # held-out validation within the training portion
    stratified_test=True,  # preserve class ratio in test fold
    stratified_valid=True,
    random_seed=42,
)

print(f"\n3-fold split — {len(train_list)} folds")
for i, (tr, va, te) in enumerate(zip(train_list, valid_list, test_list)):
    print(f"  Fold {i+1}: train={len(tr)}, valid={len(va)}, test={len(te)}")

# ---------------------------------------------------------------------------
# 6. Class imbalance: oversampling with 'repeat'
# ---------------------------------------------------------------------------
# If your classes are heavily imbalanced, pass oversample_method='repeat'
# to duplicate minority-class samples until all classes reach the target size.
#
# Create an imbalanced corpus to demonstrate:
adata_imbal = make_fake_adata(n_class=2, n_clusters_per_class=1)
# Manually drop 80% of class-1 cells
class1_idx = adata_imbal.obs[adata_imbal.obs["celltype"] == "celltype_1"].index
keep_mask = ~adata_imbal.obs.index.isin(class1_idx[10:])  # keep only first 10
adata_imbal = adata_imbal[keep_mask].copy()
adata_imbal.write_h5ad("ageas_tut02_imbal.h5ad")

imbal_corpus = Multimodal_Corpus(
    "ageas_tut02_imbal.h5ad", label_key="celltype", backed=False
)
counts_before = pd.Series(
    [imbal_corpus[i][1] for i in range(len(imbal_corpus))]
).value_counts()
print("\nClass counts before oversampling:\n", counts_before.to_dict())

train_over, _, _ = kfold_random_split(
    imbal_corpus,
    n_splits=2,
    oversample_method="repeat",
    oversample_by="max",  # upsample minority to match the largest class
    random_seed=0,
)
counts_after = pd.Series(
    [train_over[0][i][1] for i in range(len(train_over[0]))]
).value_counts()
print("Class counts after oversampling (fold 0 train):\n", counts_after.to_dict())

# ---------------------------------------------------------------------------
# 7. Corpus copy and slicing
# ---------------------------------------------------------------------------
# corpus.copy() deep-copies the corpus so you can safely subset its adata.
# This is how the operation pipelines trim the feature axis between iterations.
sub_corpus = corpus.copy()
sub_corpus.adata = sub_corpus.adata[:, :5]  # keep first 5 genes
print(f"\nSliced corpus — genes: {sub_corpus.adata.n_vars}")
print("Original corpus untouched — genes:", corpus.adata.n_vars)
