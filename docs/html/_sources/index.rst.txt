Welcome to Ageas's documentation!
=================================

**Ageas** (AutoML-based Genetic regulatory Element extrAction System) is a computational framework
for inferring cell fate bias from static single-cell and spatial multi-omics data.

Ageas trains a heterogeneous panel of classifiers — neural networks
(MLP/RNN, ResNet-mixer) alongside classical models (XGBoost, Logistic
Regression, SVM, Naive Bayes) — over an :class:`~ageas.Hangar` of candidate
units. A :class:`~ageas.Deck` orchestrates the sortie pipeline of
n-iteration k-fold cross-validation, retains the strongest units, and
aggregates per-class explanations (Integrated Gradients, SHAP, or model
coefficients) into a unified factor importance table.

Since version 0.1, Ageas also offers a label-free, feature-routed pipeline
(Ageas v2, :func:`~ageas.run_pipeline`). It designs the panel for each
dataset from a profile of the data, and narrows every unit to the features
it relies on. It integrates only the units that help each other: a no-harm
mixture and the vote of a committee of competent, distinct units. See
:ref:`v2`.

.. note::

   This project is under active development.

Key Features
------------

* **Heterogeneous Model Panel:** Mixes deep learning classifiers
  (:class:`~ageas.NN_Classifier`, :class:`~ageas.Mixer_Classifier`) with
  classical estimators (:class:`~ageas.XGB_Classifier`,
  :class:`~ageas.LogReg_Classifier`, :class:`~ageas.SVM_Classifier`,
  :class:`~ageas.MNB_Classifier`) under a single Lightning-style API.
* **N-Iteration K-Fold Selection:** :func:`~ageas.n_kfold_selection` runs
  successive cross-validation rounds and prunes the squad to the units
  that survive the configured retention/cutoff thresholds.
* **Iterative Factor Extraction:** :func:`~ageas.n_iter_extraction` and
  :func:`~ageas.n_iter_boost_selection` chain selection and
  explanation passes to converge on the most informative regulatory
  factors per cell class.
* **Unified Explanation Pipeline:** :meth:`~ageas.Deck.debrief` weights
  per-unit explanations by their validation/test metrics and produces a
  single integrated importance table across the squad.
* **Label-Free Panel Design (v2):** :func:`~ageas.profile_dataset` and
  :func:`~ageas.design_panel` derive each dataset's panel from rules stated
  before any run. :func:`~ageas.shift_weights` steers validation toward
  progenitor-like terminal cells; no step uses progenitor labels.
* **Feature Routing and No-Harm Integration (v2):**
  :func:`~ageas.moe_selection` narrows every unit to the features its own
  attributions rank highest, as far as validation allows. It integrates units
  only when they lower the ensemble's out-of-fold loss
  (:func:`~ageas.no_harm_selection`), next to a committee vote
  (:func:`~ageas.vote`).
* **Graph Stage (v2):** where unlabeled intermediate cells exist,
  :func:`~ageas.graph_stage` fuses the vote with a random walk through
  them toward the terminal states.

Citation
--------

If you use Ageas in your research, please cite:

.. code-block:: bibtex

   @software{ageas,
     title={Ageas enables time-agnostic cell fate inference from single-cell and spatial multi-omics data},
     author={Junyao Jiang, Alex Kong, Jack Yu},
     url={https://github.com/MaftyLab/Ageas},
   }

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   Installation <installation>
   Examples <examples>
   Feature-Routed Pipeline (v2) <v2>
   API Reference <api>
