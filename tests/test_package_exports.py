"""The package's public names: everything an ``__all__`` lists can be imported from it."""
import importlib

import pytest


@pytest.mark.parametrize("package", ["ageas", "ageas.nn", "ageas.classical", "ageas.ops"])
def test_every_name_in_all_is_an_attribute_of_its_package(package):
    module = importlib.import_module(package)
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []


def test_star_import_brings_the_classifiers_blocks_and_pipeline():
    namespace = {}
    exec("from ageas import *", namespace)
    for name in ("XGB_Classifier", "LogReg_Classifier", "SVM_Classifier", "MNB_Classifier", "LinReg_Classifier",
                 "NN_Classifier", "Mixer_Classifier", "Residual_Encoder", "RNN_Basic", "Deck", "run_pipeline"):
        assert name in namespace, name


def test_top_level_classifiers_are_the_subpackages_own_classes():
    import ageas
    assert ageas.XGB_Classifier is ageas.classical.XGB_Classifier
    assert ageas.NN_Classifier is ageas.nn.NN_Classifier
    assert ageas.Residual_Encoder is ageas.nn.blocks.Residual_Encoder
