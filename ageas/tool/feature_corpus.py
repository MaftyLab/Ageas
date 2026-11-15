#!/usr/bin/env python3
"""In-memory corpora and fold splits for Ageas v2."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from ageas.tool.corpus_loader import Tensor_Corpus


class Feature_Corpus(Tensor_Corpus):
    """Labelled in-memory corpus whose feature names travel with its data.

    :ivar features: Feature names, one per entry of the last data axis.
    :ivar label_dict: Integer-to-label mapping.
    :ivar label_key: ``adata.obs`` column the labels came from.
    """

    def __init__(
        self,
        data,
        labels,
        features: list,
        label_dict: dict = None,
        label_key: str = None,
    ) -> None:
        """Initialize a Feature_Corpus.

        :param data: Tensor shaped ``(n_cells, n_layers, n_features)``.
        :param labels: Integer label tensor shaped ``(n_cells,)``.
        :param features: Feature names for the last data axis.
        :param label_dict: Integer-to-label mapping.
        :param label_key: ``adata.obs`` column the labels came from.
        """
        super().__init__(data=data, labels=labels, parent=None)
        self.features = list(features)
        self.label_dict = label_dict
        self.label_key = label_key

    def subset(self, indices) -> 'Feature_Corpus':
        """The cells at ``indices``, with every feature."""
        return Feature_Corpus(
            self.data[indices], self.labels[indices],
            self.features, self.label_dict, self.label_key,
        )

    def restrict(self, features: list) -> 'Feature_Corpus':
        """Every cell, with only ``features`` (in the given order)."""
        index = pd.Index(self.features).get_indexer(features)
        assert (index >= 0).all(), "Unknown features requested."
        return Feature_Corpus(
            self.data[..., index], self.labels,
            features, self.label_dict, self.label_key,
        )


def kfold_indices(
    dataset,
    n_splits: int,
    valid_fraction: float = 0.1,
    stratified_test: bool = False,
    stratified_valid: bool = True,
    random_seed: int = None,
) -> list:
    """Cell indices of every fold, split as :func:`ageas.tool.kfold_random_split` splits them.

    The same splitter calls are made in the same order, so a v2 run and an
    R1 run with the same seed see the same folds.

    :param dataset: ``AnnData``-backed corpus with ``label_key`` set.
    :param n_splits: Number of folds.
    :param valid_fraction: Fraction of each training part held out for
        validation.
    :param stratified_test: If ``True``, use class-stratified folds.
    :param stratified_valid: If ``True``, stratify the validation split.
    :param random_seed: Seed for reproducible splits.
    :returns: One ``{'train', 'valid', 'test'}`` dict of index arrays per
        fold.
    """
    obs = dataset.adata.obs
    shuffle = random_seed is not None
    if stratified_test:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_seed,
        ).split(obs.index, obs[dataset.label_key])
    else:
        splitter = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_seed,
        ).split(obs.index)

    folds = []
    for train, test in splitter:
        stratify = (
            [dataset.reverse_label_dict[v] for v in obs[dataset.label_key].iloc[train]]
            if stratified_valid else None
        )
        train, valid = train_test_split(
            train,
            test_size=valid_fraction,
            stratify=stratify,
            random_state=random_seed,
        )
        folds.append({
            'train': np.asarray(train),
            'valid': np.asarray(valid),
            'test': np.asarray(test),
        })
    return folds
