#!/usr/bin/env python3
"""Multinomial Naive Bayes classifier.

Wraps :class:`sklearn.naive_bayes.MultinomialNB` in a Lightning-style
classifier and uses ``feature_log_prob_`` to produce a class-contrastive
feature importance table.
"""
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.naive_bayes import MultinomialNB

import ageas.tool.JSON as JSON
from ageas.tool import l1_normalize


class MNB_Classifier(pl.LightningModule):
    """Lightning wrapper around :class:`sklearn.naive_bayes.MultinomialNB`.

    Uses :meth:`~sklearn.naive_bayes.MultinomialNB.partial_fit` so that the
    model can be trained incrementally over multiple training batches with a
    fixed list of classes provided through ``model_params['num_class']``.

    Attributes:
        model: Underlying :class:`~sklearn.naive_bayes.MultinomialNB`
            instance.
    """

    def __init__(
        self,
        fea_names=None,
        model_params: dict = None,
        **kwargs,
    ) -> None:
        """Initialize an MNB_Classifier.

        Args:
            fea_names: Feature names (gene symbols or Ensembl IDs).
            model_params: Dict of hyper-parameters. Recognized keys:
                ``alpha``, ``force_alpha``, ``fit_prior``, ``class_prior``,
                ``num_class``.
            **kwargs: Forwarded to
                :class:`~pytorch_lightning.LightningModule`.
        """
        if model_params is None:
            model_params = {
                'alpha': 1.0,
                'force_alpha': True,
                'fit_prior': True,
                'class_prior': None,
                'num_class': 2,
            }

        super().__init__()
        self.model = MultinomialNB(
            alpha=model_params['alpha'],
            force_alpha=model_params['force_alpha'],
            fit_prior=model_params['fit_prior'],
            class_prior=model_params['class_prior'],
        )
        self.save_hyperparameters()

    def forward(self, x, y) -> None:
        """Incrementally fit the Naive Bayes model with the current batch.

        Args:
            x: Input feature tensor.
            y: Label tensor.
        """
        self.model.partial_fit(
            np.array(x.squeeze()),
            np.array(y),
            classes=list(range(self.hparams.model_params['num_class'])),
        )

    def predict(self, x, y=None) -> np.ndarray:
        """Return per-class probabilities from ``predict_proba``.

        Args:
            x: Input feature tensor.
            y: Unused; kept for API symmetry.

        Returns:
            Probability matrix of shape ``(batch, num_class)``.
        """
        return self.model.predict_proba(np.array(x.squeeze()))

    def explain(self, score_name: str = 'Scores', **kwargs) -> pd.DataFrame:
        """Class-contrastive feature importance from ``feature_log_prob_``.

        Computes a per-feature contribution magnitude, subtracts the
        cross-class background (max over the other classes), and
        L1-normalises the result per class.

        Args:
            score_name: Unused; kept for API symmetry.
            **kwargs: Ignored extra arguments.

        Returns:
            Score table indexed by feature with columns
            ``Class_{i}_Scores``.
        """
        # Shape: (n_features, n_classes)
        ans = self.model.feature_log_prob_.T

        contribution = np.mean(np.abs(ans), axis=0)
        contribution /= np.sum(contribution)

        backgrounds = np.zeros_like(ans)
        for ind in range(len(ans)):
            backgrounds[ind] = np.max(np.delete(ans, ind, axis=0), axis=0)

        for ind, exp in enumerate(ans):
            ans[ind] = l1_normalize((exp - backgrounds[ind]) * contribution)

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
