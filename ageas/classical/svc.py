#!/usr/bin/env python3
"""Support vector machine classifier.

Wraps :class:`sklearn.svm.SVC` in a Lightning-style template, with a
:class:`~sklearn.preprocessing.StandardScaler` applied to the features
before fitting and predicting.

Note:
    Only the ``'linear'`` kernel exposes a usable ``coef_`` attribute, so
    explanations are only meaningful for linear SVMs. Non-linear kernels
    emit a warning at construction time.
"""
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ageas.tool import to_2d_numpy
from .sk_template import Classifier_Template


class SVM_Classifier(Classifier_Template):
    """Lightning wrapper around :class:`sklearn.svm.SVC` with input scaling.

    Attributes:
        model: Underlying :class:`~sklearn.svm.SVC` instance.
        scaler: :class:`~sklearn.preprocessing.StandardScaler` applied
            before fitting and predicting.
    """

    def __init__(
        self,
        fea_names=None,
        model_params: dict = None,
        **kwargs,
    ) -> None:
        """Initialize an SVM_Classifier.

        Args:
            fea_names: Feature names (gene symbols or Ensembl IDs).
            model_params: Dict of hyper-parameters. Scaler keys:
                ``scaler_copy``, ``scaler_with_mean``, ``scaler_with_std``.
                SVC keys: ``C``, ``kernel``, ``degree``, ``gamma``,
                ``coef0``, ``shrinking``, ``tol``, ``cache_size``,
                ``class_weight``, ``verbose``, ``max_iter``,
                ``decision_func_shape``, ``break_ties``, ``random_state``,
                ``num_class``.
            **kwargs: Forwarded to the parent
                :class:`~ageas.classical.sk_template.Classifier_Template`.
        """
        if model_params is None:
            model_params = {
                'scaler_copy': True,
                'scaler_with_mean': True,
                'scaler_with_std': True,
                'C': 1.0,
                'kernel': 'linear',
                'degree': 3,
                'gamma': 'scale',
                'coef0': 0.0,
                'shrinking': True,
                'tol': 0.001,
                'cache_size': 200,
                'class_weight': None,
                'verbose': False,
                'max_iter': -1,
                'decision_func_shape': 'ovr',
                'break_ties': False,
                'random_state': None,
                'num_class': 2,
            }

        super().__init__(fea_names=fea_names, model_params=model_params, **kwargs)
        self.model = SVC(
            C=model_params['C'],
            kernel=model_params['kernel'],
            degree=model_params['degree'],
            gamma=model_params['gamma'],
            coef0=model_params['coef0'],
            shrinking=model_params['shrinking'],
            probability=True,
            tol=model_params['tol'],
            cache_size=model_params['cache_size'],
            class_weight=model_params['class_weight'],
            verbose=model_params['verbose'],
            max_iter=model_params['max_iter'],
            decision_function_shape=model_params['decision_func_shape'],
            break_ties=model_params['break_ties'],
            random_state=model_params['random_state'],
        )
        self.scaler = StandardScaler(
            copy=model_params['scaler_copy'],
            with_mean=model_params['scaler_with_mean'],
            with_std=model_params['scaler_with_std'],
        )

        if model_params['kernel'] != 'linear':
            warnings.warn(
                'SVM with non-linear kernel does not support coefficient-based '
                'explanation; explanations will be meaningless.'
            )

    def apply_scaler(self, x, fit: bool = False) -> np.ndarray:
        """Standard-scale ``x`` with the training-set statistics.

        The scaler is fitted only when ``fit=True`` (i.e. during training);
        prediction inputs are transformed with the stored training mean and
        standard deviation, so predictions do not depend on how query cells
        are batched.

        Args:
            x: Input feature tensor.
            fit: If ``True``, (re)fit the scaler on ``x`` first.

        Returns:
            Standard-scaled feature matrix as :class:`numpy.ndarray`.
        """
        x = to_2d_numpy(x)
        if fit:
            self.scaler.fit(x)
        return self.scaler.transform(x)

    def forward(self, x, y) -> None:
        """Fit the scaler and the underlying SVC on the training batch.

        Args:
            x: Input feature tensor.
            y: Label tensor.
        """
        self.model.fit(self.apply_scaler(x, fit=True), np.array(y))

    def predict(self, x, y=None) -> np.ndarray:
        """Standard-scale ``x`` and return per-class probabilities.

        Args:
            x: Input feature tensor.
            y: Unused; kept for API symmetry.

        Returns:
            Output of ``self.model.predict_proba`` on the scaled inputs.
        """
        return self.model.predict_proba(self.apply_scaler(x))

    def class_coefficients(self) -> np.ndarray:
        """Per-class coefficient vectors of a linear SVC.

        Binary problems follow the template behaviour. For more than two
        classes, :class:`~sklearn.svm.SVC` trains one-vs-one machines whose
        rows are ordered ``(0, 1), (0, 2), ..., (1, 2), ...``; a positive
        decision value votes for the first class of each pair. Each class
        therefore receives the mean of its pairwise coefficient vectors,
        oriented toward that class.

        Returns:
            Array of shape ``(num_class, n_features)``.
        """
        coef = np.atleast_2d(np.asarray(self.model.coef_, dtype=float))
        classes = [int(c) for c in self.model.classes_]
        if coef.shape[0] == 1:
            return super().class_coefficients()

        scores = np.zeros((self.hparams.model_params['num_class'], coef.shape[1]))
        counts = np.zeros(scores.shape[0])
        pair = 0
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                scores[classes[i]] += coef[pair]
                scores[classes[j]] -= coef[pair]
                counts[classes[i]] += 1
                counts[classes[j]] += 1
                pair += 1
        return scores / np.maximum(counts, 1)[:, None]
