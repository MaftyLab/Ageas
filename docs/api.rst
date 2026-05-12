.. _api:

API Reference
=============

This section provides detailed API documentation for the modules, classes,
and functions in the **Ageas** package.

Core Objects
------------

.. autosummary::
   :toctree: generated
   :recursive:

   ageas.Hangar
   ageas.Deck
   ageas.Unit

Operations
----------

.. autosummary::
   :toctree: generated
   :recursive:

   ageas.n_kfold_selection
   ageas.n_iter_boost_selection
   ageas.n_iter_extraction

.. Neural Network Classifiers
.. --------------------------

.. .. autosummary::
..    :toctree: generated
..    :recursive:

..    ageas.NN_Classifier
..    ageas.Mixer_Classifier
..    ageas.nn.blocks

.. Classical Classifiers
.. ---------------------

.. .. autosummary::
..    :toctree: generated
..    :recursive:

..    ageas.XGB_Classifier
..    ageas.LinReg_Classifier
..    ageas.LogReg_Classifier
..    ageas.SVM_Classifier
..    ageas.MNB_Classifier

.. Tools and Utilities
.. -------------------

.. .. autosummary::
..    :toctree: generated
..    :recursive:

..    ageas.tool
..    ageas.tool.corpus_loader
..    ageas.tool.explainer
..    ageas.tool.trainer
..    ageas.tool.config_optim
..    ageas.tool.knn
..    ageas.tool.JSON
