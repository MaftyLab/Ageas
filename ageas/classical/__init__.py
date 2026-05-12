#!/usr/bin/env python3
"""
Classical (non neural network) classifiers for tabular data.

Re-exports XGBoost, linear regression, logistic regression, SVM, and
Multinomial Naive Bayes classifiers, all wrapped in a Lightning-compatible
interface so they can be trained and explained alongside the neural network
models in the same Ageas pipeline.

author: jy
"""
from .xgboost import XGB_Classifier
from .lin_reg import LinReg_Classifier
from .log_reg import LogReg_Classifier
from .svc import SVM_Classifier
from .mnb import MNB_Classifier



__all__ = [
    "XGB_Classifier",
    "LinReg_Classifier",
    "LogReg_Classifier",
    "SVM_Classifier",
    "MNB_Classifier",
]
