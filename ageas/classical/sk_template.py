#!/usr/bin/env python3
"""Lightning wrapper template for scikit-learn classifiers.

Provides a common base class that adapts scikit-learn estimators to the
``LightningModule`` API used everywhere else in Ageas, so non-NN models can
be trained, predicted, and explained through the same trainer/deck pipeline.
"""
import numpy as np
import pandas as pd
import pytorch_lightning as pl

import ageas.tool.JSON as JSON
from ageas.tool import class_contrast, l1_normalize, to_2d_numpy


class Classifier_Template(pl.LightningModule):
    """Lightning-compatible base class for scikit-learn style classifiers.

    Subclasses are expected to instantiate ``self.model`` with a
    scikit-learn estimator that exposes ``fit``, ``predict_proba``, and a
    coefficient matrix ``coef_`` for the default explanation method.

    Attributes:
        model: The underlying scikit-learn estimator.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the template, saving all kwargs as hyper-parameters."""
        super().__init__()
        self.model = None
        self.save_hyperparameters()

    def forward(self, x, y):
        """Fit the underlying scikit-learn estimator on ``(x, y)``.

        Args:
            x: Input feature tensor; squeezed and converted to
                :class:`numpy.ndarray`.
            y: Label tensor.
        """
        self.model.fit(to_2d_numpy(x), np.array(y))

    def predict(self, x, y=None) -> np.ndarray:
        """Return per-class probabilities from the underlying estimator.

        Args:
            x: Input feature tensor.
            y: Unused; kept for API symmetry with NN classifiers.

        Returns:
            Output of ``self.model.predict_proba``.
        """
        return self.model.predict_proba(to_2d_numpy(x))

    def class_coefficients(self) -> np.ndarray:
        """Per-class coefficient vectors oriented toward each class.

        scikit-learn stores a single coefficient row for binary problems,
        oriented toward ``classes_[1]``; the row is assigned to
        ``classes_[1]`` and its negation to ``classes_[0]``. Multi-class
        estimators with one row per class (multinomial or one-vs-rest) are
        mapped row-by-row onto ``classes_``.

        Returns:
            Array of shape ``(num_class, n_features)``.
        """
        coef = np.atleast_2d(np.asarray(self.model.coef_, dtype=float))
        classes = [int(c) for c in self.model.classes_]
        scores = np.zeros((self.hparams.model_params['num_class'], coef.shape[1]))
        if coef.shape[0] == 1:
            scores[classes[1]] = coef[0]
            scores[classes[0]] = -coef[0]
        else:
            for row, c in zip(coef, classes):
                scores[c] = row
        return scores

    def explain(self, score_name: str = 'Scores', **kwargs) -> pd.DataFrame:
        """Default coefficient-based feature importance.

        Each class's coefficient vector (see :meth:`class_coefficients`) is
        L1-normalised and then passed through the same class-contrastive
        specificity step (``d_c = s_c - max_{k != c} s_k``) used for the
        neural-network and tree-based explanations.

        Note:
            This default implementation works on linear models with a
            ``coef_`` attribute. Subclasses with non-linear models should
            override this method.

        Args:
            score_name: Column suffix for the score columns (unused in the
                default implementation but kept for subclass compatibility).
            **kwargs: Ignored extra arguments.

        Returns:
            Per-feature, per-class score table as a
            :class:`~pandas.DataFrame`.
        """
        scores = np.array([
            l1_normalize(row, mode='numpy') for row in self.class_coefficients()
        ])
        scores = class_contrast(scores)

        return pd.DataFrame(
            scores.T,
            index=self.hparams.fea_names,
            columns=[f'Class_{i}_Scores' for i in range(scores.shape[0])],
        )

    def save_model(self, path: str) -> None:
        """Persist the estimator's hyper-parameters as JSON.

        Args:
            path: Destination path for the JSON file.
        """
        JSON.encode(self.model.get_params(), path)

    def load_model(self, path: str) -> None:
        """Restore the estimator's hyper-parameters from JSON.

        Args:
            path: Path to a JSON file produced by :meth:`save_model`.
        """
        self.model = self.model.set_params(**JSON.decode(path))

    def trans_preds(self, preds: np.ndarray, **kwargs) -> np.ndarray:
        """Convert per-class probabilities into discrete class labels.

        Args:
            preds: Probability matrix from :meth:`predict`.
            **kwargs: Ignored extra arguments.

        Returns:
            ``argmax`` along the class axis.
        """
        return preds.argmax(axis=-1)
