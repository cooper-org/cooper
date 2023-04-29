# coding: utf8

import dataclasses
from collections.abc import Sequence
from typing import Dict, Literal, Optional, Union

from cooper.utils import validate_state_dicts

from .types import AlternatingType


@dataclasses.dataclass
class CooperOptimizerState:
    """Represents the state of a constrained (or unconstrained) optimizer in terms of
    the state dicts of the primal optimizers, as well as those of the dual optimizer
    and the dual scheduler if applicable. This class can be used for producing
    checkpoints of a Cooper-related optimizer.

    Args:
        primal_optimizer_states: State dicts for the primal optimizers.
        dual_optimizer_states: State dicts for the dual optimizers.
        multiplier_states: State dicts for the multipliers.
        extrapolation: Flag indicating if the optimizer performs extrapolation updates.
        alternating: Flag indicating if the optimizer performs alternating updates.
    """

    primal_optimizer_states: Sequence[Dict]
    dual_optimizer_states: Optional[Sequence[Dict]] = None
    multiplier_states: Optional[Sequence[Dict]] = None
    extrapolation: bool = False
    alternating: AlternatingType = AlternatingType.FALSE

    def asdict(self):
        return dataclasses.asdict(self)

    def __eq__(self, other):
        assert isinstance(other, CooperOptimizerState)

        flag_names = ["extrapolation", "alternating"]
        for flag_name in flag_names:
            if getattr(self, flag_name) != getattr(other, flag_name):
                return False

        state_dict_names = ["primal_optimizer_states", "dual_optimizer_states", "multiplier_states"]
        for state_dict_name in state_dict_names:
            try:
                dicts_match = validate_state_dicts(getattr(self, state_dict_name), getattr(other, state_dict_name))
            except:
                dicts_match = False
            if not dicts_match:
                return False

        return True
