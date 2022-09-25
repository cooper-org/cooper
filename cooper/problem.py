import abc
from dataclasses import dataclass
from typing import Optional

import torch

# Formulation, and some other classes below, are heavily inspired by the design
# of the TensorFlow Constrained Optimization (TFCO) library :
# https://github.com/google-research/tensorflow_constrained_optimization


@dataclass
class CMPState:
    """Represents the "state" of a Constrained Minimization Problem in terms of
    the value of its loss and constraint violations/defects.

    Args:
        loss: Value of the loss or main objective to be minimized :math:`f(x)`
        ineq_defect: Violation of the inequality constraints :math:`g(x)`
        eq_defect: Violation of the equality constraints :math:`h(x)`
        proxy_ineq_defect: Differentiable surrogate for the inequality
            constraints as proposed by :cite:t:`cotter2019JMLR`.
        proxy_eq_defect: Differentiable surrogate for the equality constraints
            as proposed by :cite:t:`cotter2019JMLR`.
        misc: Optional additional information to be store along with the state
            of the CMP
    """

    loss: Optional[torch.Tensor] = None
    ineq_defect: Optional[torch.Tensor] = None
    eq_defect: Optional[torch.Tensor] = None
    proxy_ineq_defect: Optional[torch.Tensor] = None
    proxy_eq_defect: Optional[torch.Tensor] = None
    misc: Optional[dict] = None

    def as_tuple(self) -> tuple:
        return (
            self.loss,
            self.ineq_defect,
            self.eq_defect,
            self.proxy_ineq_defect,
            self.proxy_eq_defect,
            self.misc,
        )


class ConstrainedMinimizationProblem(abc.ABC):
    """
    Base class for constrained minimization problems.

    Args:
        is_constrained: We request the problem to be explicitly declared as
            constrained or unconstrained to perform sanity checks when
            initializing the :py:class:`~cooper.ConstrainedOptimizer`. Defaults
            to ``False``.
    """

    def __init__(self, is_constrained: bool = False):
        self.is_constrained = is_constrained
        self._state = CMPState()

    @property
    def state(self) -> CMPState:
        return self._state

    @state.setter
    def state(self, value: CMPState):
        self._state = value

    @abc.abstractmethod
    def closure(self, *args, **kwargs) -> CMPState:
        """
        Computes the state of the CMP based on the current value of the primal
        parameters.

        The signature of this abstract function may be changed to accommodate
        situations that require a model, (mini-batched) inputs/targets, or
        other arguments to be passed.

        Structuring the CMP class around this closure method, enables the re-use
        of shared sections of a computational graph. For example, consider a
        case where we want to minimize a model's cross entropy loss subject to
        a constraint on the entropy of its predictions. Both of these quantities
        depend on the predicted logits (on a minibatch). This closure-centric
        design allows flexible problem specifications while avoiding
        re-computation.
        """

    def defect_fn(self) -> CMPState:
        """
        Computes the constraints of the CMP based on the current value of the
        primal parameters. This function returns a
        :py:class:`cooper.problem.CMPState` collecting the values of the (proxy
        and/or non-proxy) constraints. Note that this returned ``CMPState`` may
        have a ``loss`` attribute with value ``None`` since, by design, the loss
        is not necessarily computed when evaluating `only` the constraints.

        The signature of this "abstract" function may be changed to accommodate
        situations that require a model, (mini-batched) inputs/targets, or
        other arguments to be passed.

        Depending on the problem at hand, the computation of the constraints
        can be compartimentalized in a way that is independent of the evaluation
        of the loss. Alternatively,
        :py:meth:`~.ConstrainedMinimizationProblem.defect_fn` may be used in the
        implementation of the :py:meth:`~.ConstrainedMinimizationProblem.closure`
        function.
        """
        raise NotImplementedError
