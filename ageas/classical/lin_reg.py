#!/usr/bin/env python3
"""Linear regression classifier.

Wraps :class:`sklearn.linear_model.LinearRegression` in the Lightning-style
:class:`~ageas.classical.sk_template.Classifier_Template` so it can be
trained and explained through the standard Ageas pipeline.
"""
from sklearn.linear_model import LinearRegression

from .sk_template import Classifier_Template


class LinReg_Classifier(Classifier_Template):
    """Lightning wrapper around :class:`sklearn.linear_model.LinearRegression`.

    Attributes:
        model: Underlying :class:`~sklearn.linear_model.LinearRegression`
            instance.
    """

    def __init__(
        self,
        fea_names=None,
        model_params: dict = None,
        **kwargs,
    ) -> None:
        """Initialize a LinReg_Classifier.

        Args:
            fea_names: Feature names (gene symbols or Ensembl IDs).
            model_params: Dict of hyper-parameters forwarded to
                :class:`~sklearn.linear_model.LinearRegression`. Recognized
                keys: ``fit_intercept`` (bool, default ``True``),
                ``positive`` (bool, default ``False``).
            **kwargs: Forwarded to the parent
                :class:`~ageas.classical.sk_template.Classifier_Template`.
        """
        if model_params is None:
            model_params = {'fit_intercept': True, 'positive': False}

        super().__init__(fea_names=fea_names, model_params=model_params, **kwargs)
        self.model = LinearRegression(
            fit_intercept=model_params['fit_intercept'],
            positive=model_params['positive'],
            copy_X=True,
        )
