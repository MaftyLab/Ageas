#!/usr/bin/env python3
"""Unit object for Ageas.

A :class:`Unit` wraps a single model with its hyper-parameters, runtime
device configuration, the trained model itself, and the per-operation
reports. :class:`~ageas.Hangar` and :class:`~ageas.Deck` both operate on
collections of these units.
"""
import logging
from warnings import warn

from ageas.nn import NN_Classifier, Mixer_Classifier
from ageas.classical import (
    SVM_Classifier,
    LogReg_Classifier,
    MNB_Classifier,
    XGB_Classifier,
)

_logger = logging.getLogger(__name__)

#: Mapping of supported short type names to their classifier classes.
SUPPORTED_TYPES = {
    'svc': SVM_Classifier,
    'logreg': LogReg_Classifier,
    'mnb': MNB_Classifier,
    'xgb': XGB_Classifier,
    'mlp': NN_Classifier,
    'rnn': NN_Classifier,
    'resnet': Mixer_Classifier,
}

#: Model types that are forced onto CPU regardless of the requested accelerator.
CPU_MODELS = ['svc', 'mnb', 'logreg']


class Unit:
    """Single classifier model bundled with its configuration and runtime state.

    The unit stores everything needed to reproduce, train, and inspect one
    model: the classifier type, the configuration dictionary, the chosen
    accelerator, the trained model weights, the trainer object, and the
    accumulated reports from each Ageas operation.

    :ivar tail: Short name used to disambiguate units sharing the same type.
    :ivar type: Model type key (one of :data:`SUPPORTED_TYPES`).
    :ivar config: Hyper-parameter dictionary for the classifier and trainer.
    :ivar accelerator: Current accelerator string (``'cpu'`` or ``'cuda'``).
    :ivar device: GPU device count / index resolved at spike time.
    :ivar model: Trained classifier instance (``None`` until after a sortie).
    :ivar trainer: Trainer object (``None`` until after a sortie).
    :ivar report: Per-operation metric records accumulated during selection.
    """

    def __init__(
        self,
        tail: str,
        unit_type: str,
        config: dict,
        accelerator: str = 'cpu',
    ) -> None:
        """Initialize a Unit.

        :param tail: Short tail name used to disambiguate units that share the
            same model type (e.g. ``'unit1'``, ``'unit2'``).
        :param unit_type: Model type key. Must be one of :data:`SUPPORTED_TYPES`.
        :param config: Hyper-parameter dictionary forwarded to the underlying
            classifier and trainer.
        :param accelerator: Initial accelerator hint, ``'cpu'`` or ``'cuda'``.
            Re-evaluated at sortie time by :meth:`spike`.
        """
        self.tail = tail
        self.type = unit_type
        self.config = config
        self.accelerator = accelerator
        self.device = 1
        self.model = None
        self.trainer = None
        self.report: dict = {}

    @property
    def id(self) -> str:
        """Public unit ID combining the model type and tail.

        :returns: ``"{type}_{tail}"`` formatted identifier.
        """
        return f"{self.type}_{self.tail}"

    @property
    def clf_object(self) -> type:
        """Classifier class associated with this unit's type.

        :returns: The class (not an instance) registered under ``self.type``
            in :data:`SUPPORTED_TYPES`.
        """
        return SUPPORTED_TYPES[self.type]

    def spike(self, accelerator: str, available_devices: list = None):
        """Verify that the unit is ready to launch and resolve its accelerator.

        Performs three checks in order:

        1. The model type is supported by Ageas.
        2. CPU-only models are pinned to the CPU accelerator.
        3. CUDA models receive a valid device list, or fall back to CPU
           with a warning.

        :param accelerator: Requested accelerator (``'cpu'`` or ``'cuda'``).
        :param available_devices: Optional list of GPU device indices that the
            deck has reserved for this unit.
        :returns: The classifier class to instantiate, or ``None`` if the unit
            cannot be launched (unsupported type, unknown accelerator).
        """
        if self.type not in SUPPORTED_TYPES:
            warn(f"Unit type {self.type} is not supported.")
            return None

        if self.type in CPU_MODELS:
            self.accelerator = 'cpu'
            return self.clf_object

        if accelerator == 'cpu':
            self.accelerator = 'cpu'
            return self.clf_object

        if accelerator == 'cuda':
            if not available_devices:
                warn("No available CUDA devices found.")
                warn(f"Switching {self.id} to CPU mode.")
                self.accelerator = 'cpu'
                return self.clf_object
            self.accelerator = 'cuda'
            self.device = (
                len(available_devices) if self.type != 'xgb' else 0
            )
            return self.clf_object

        warn(f"Unknown accelerator type: {accelerator}")
        return None
