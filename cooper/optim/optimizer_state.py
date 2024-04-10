# coding: utf8

import dataclasses
from collections.abc import Sequence
from typing import Dict, Optional


@dataclasses.dataclass
class OptimizerState:
    """Represents the state of a constrained (or unconstrained) optimizer in terms of
    the state dicts of the primal optimizers, as well as those of the dual optimizer
    and the dual scheduler if applicable. This class can be used for producing
    checkpoints of a Cooper-related optimizer.

    Args:
        primal_optimizer_states: State dicts for the primal optimizers.
        dual_optimizer_states: State dicts for the dual optimizers.
    """

    primal_optimizer_states: Sequence[Dict]
    dual_optimizer_states: Optional[Sequence[Dict]] = None
