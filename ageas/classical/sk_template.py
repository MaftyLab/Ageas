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
from ageas.tool import l1_normalize


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
        self.model.fit(np.array(x.squeeze()), np.array(y))

    def predict(self, x, y=None) -> np.ndarray:
        """Return per-class probabilities from the underlying estimator.

        Args:
            x: Input feature tensor.
            y: Unused; kept for API symmetry with NN classifiers.

        Returns:
            Output of ``self.model.predict_proba``.
        """
        return self.model.predict_proba(np.array(x.squeeze()))

    def explain(self, score_name: str = 'Scores', **kwargs) -> pd.DataFrame:
        """Default coefficient-based feature importance.

        For binary classifiers the negative-class scores are appended to
        match the multi-class layout used everywhere else; multi-class
        coefficients are L1-normalised per class.

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
        if self.hparams.model_params['num_class'] == 2:
            ans = l1_normalize(self.model.coef_.T, mode='numpy')
            ans = np.concatenate((ans, -ans), axis=1)
        else:
            ans = self.model.coef_.T
            for i, exp in enumerate(ans):
                ans[i] = l1_normalize(exp, mode='numpy')

        return pd.DataFrame(
            ans,
            index=self.hparams.fea_names,
            columns=[f'Class_{i}_Scores' for i in range(ans.shape[1])],
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
