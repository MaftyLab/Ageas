Welcome to Ageas's documentation!
=================================

**Ageas** (AutoML-based Genetic regulatory Element extrAction System) is a
multi-model framework for identifying class-specific regulatory factors from
single-cell omics data.

Ageas trains a heterogeneous panel of classifiers — neural networks
(MLP/RNN, ResNet-mixer) alongside classical models (XGBoost, Logistic
Regression, SVM, Naive Bayes) — over an :class:`~ageas.Hangar` of candidate
units. A :class:`~ageas.Deck` orchestrates the sortie pipeline of
n-iteration k-fold cross-validation, retains the strongest units, and
aggregates per-class explanations (Integrated Gradients, SHAP, or model
coefficients) into a unified factor importance table.

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
   API Reference <api>
