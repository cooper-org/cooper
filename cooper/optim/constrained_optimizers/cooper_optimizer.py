# coding: utf8
"""
Implementation of :py:class:`CooperOptimizer` class, which has 2 main
methods:

- :py:meth:`~CooperOptimizer.zero_grad`

- :py:meth:`~CooperOptimizer.step`
"""

import abc
import dataclasses
from typing import Dict, List, Optional

from cooper.utils import validate_state_dicts


@dataclasses.dataclass
class CooperOptimizerState:
    """Represents the "state" of a Constrained Optimizer in terms of the state
    dicts of the primal optimizers, as well as those of the dual optimizer and
    the dual scheduler if applicable. This is used for checkpointing.

    Args:
        primal_optimizer_states: State dict for the primal optimizers.
        dual_optimizer_state: State dict for the dual optimizer.
        dual_scheduler_state: State dict for the dual scheduler.
    """

    primal_optimizer_states: List[Dict]
    dual_optimizer_state: Optional[Dict] = None
    dual_scheduler_state: Optional[Dict] = None
    extrapolation: bool = False
    alternating: bool = False
    dual_restarts: bool = False

    def __eq__(self, other):

        assert isinstance(other, CooperOptimizerState)

        def compare_state_dicts(dict_name):
            try:
                return validate_state_dicts(
                    getattr(self, dict_name), getattr(other, dict_name)
                )
            except:
                return False

        state_dict_names = [
            "primal_optimizer_states",
            "dual_optimizer_state",
            "dual_scheduler_state",
        ]

        all_checks = [compare_state_dicts(_) for _ in state_dict_names]
        all_checks.append(self.extrapolation == other.extrapolation)
        all_checks.append(self.alternating == other.alternating)
        all_checks.append(self.dual_restarts == other.dual_restarts)

        return all(all_checks)

    def asdict(self):
        return dataclasses.asdict(self)


class CooperOptimizer(abc.ABC):
    """
    Base class for Cooper constrained and unconstrained optimizers.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def zero_grad(self):
        pass

    @abc.abstractmethod
    def state_dict(self) -> CooperOptimizerState:
        pass
