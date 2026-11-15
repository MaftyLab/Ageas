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

Feature-Routed Pipeline (v2)
----------------------------

The label-free workflow (see :ref:`v2`) and the objects it returns.

.. autosummary::
   :toctree: generated
   :recursive:

   ageas.run_pipeline
   ageas.PipelineResult
   ageas.moe_selection
   ageas.MoE_Deck

Data Profile and Panel Design (v2)
----------------------------------

.. autosummary::
   :toctree: generated
   :recursive:

   ageas.profile_dataset
   ageas.design_panel
   ageas.extend_panel
   ageas.write_panel
   ageas.UnitSpec
   ageas.check_strategy

Validation, Routing and Integration (v2)
----------------------------------------

.. autosummary::
   :toctree: generated
   :recursive:

   ageas.shift_weights
   ageas.explain_unit
   ageas.key_features
   ageas.search_ladder
   ageas.no_harm_selection
   ageas.competent_units
   ageas.select_committee
   ageas.vote

Graph Stage (v2)
----------------

.. autosummary::
   :toctree: generated
   :recursive:

   ageas.graph_stage
   ageas.graph_fate
   ageas.fuse

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
