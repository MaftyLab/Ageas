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

    def apply_scaler(self, x) -> np.ndarray:
        """Fit (or refit) the scaler on ``x`` and return the scaled features.

        Args:
            x: Input feature tensor.

        Returns:
            Standard-scaled feature matrix as :class:`numpy.ndarray`.
        """
        self.scaler.fit(np.array(x.squeeze()))
        return self.scaler.transform(np.array(x.squeeze()))

    def forward(self, x, y) -> None:
        """Standard-scale ``x`` and fit the underlying SVC.

        Args:
            x: Input feature tensor.
            y: Label tensor.
        """
        self.model.fit(self.apply_scaler(x), np.array(y))

    def predict(self, x, y=None) -> np.ndarray:
        """Standard-scale ``x`` and return per-class probabilities.

        Args:
            x: Input feature tensor.
            y: Unused; kept for API symmetry.

        Returns:
            Output of ``self.model.predict_proba`` on the scaled inputs.
        """
        return self.model.predict_proba(self.apply_scaler(x))
