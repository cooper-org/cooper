# coding: utf8
"""
Implementation of the :py:class:`ExtrapolationConstrainedOptimizer` class.
"""

import torch

from cooper.cmp import CMPState, ConstrainedMinimizationProblem, LagrangianStore
from cooper.utils import OneOrSequence

from ..types import AlternationType
from .constrained_optimizer import ConstrainedOptimizer


class ExtrapolationConstrainedOptimizer(ConstrainedOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing extrapolation updates to the primal and dual variables.
    """

    # TODO(gallego-posada): Add equations to illustrate the extrapolation updates

    extrapolation = True
    alternation_type = AlternationType.FALSE

    def __init__(
        self,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: OneOrSequence[torch.optim.Optimizer],
        cmp: ConstrainedMinimizationProblem,
    ):
        super().__init__(primal_optimizers, dual_optimizers, cmp)

        self.base_sanity_checks()

        self.custom_sanity_checks()

    def custom_sanity_checks(self):
        """
        Perform sanity checks on the initialization of
        ``ExtrapolationConstrainedOptimizer``.

        Raises:
            RuntimeError: Tried to construct an ExtrapolationConstrainedOptimizer but
                some of the provided optimizers do not have an extrapolation method.
        """

        are_primal_extra_optims = [hasattr(_, "extrapolation") for _ in self.primal_optimizers]
        are_dual_extra_optims = [hasattr(_, "extrapolation") for _ in self.dual_optimizers]

        if not all(are_primal_extra_optims) or not all(are_dual_extra_optims):
            raise RuntimeError(
                """Some of the provided optimizers do not have an extrapolation method.
                Please ensure that all optimizers are extrapolation capable."""
            )

    def step(self, call_extrapolation: bool = False):
        """Performs an extrapolation step or update step on both the primal and dual
        variables.

        Args:
            call_extrapolation: Whether to call ``primal_optimizer.extrapolation()`` as
                opposed to ``primal_optimizer.step()``. Defaults to False.
        """

        call_method = "extrapolation" if call_extrapolation else "step"

        for primal_optimizer in self.primal_optimizers:
            getattr(primal_optimizer, call_method)()  # type: ignore

            # FIXME(gallego-posada): This line should not be indented inside the loop!
            self.dual_step(call_extrapolation=call_extrapolation)

    def roll(self, compute_cmp_state_kwargs: dict = {}) -> tuple[CMPState, LagrangianStore, LagrangianStore]:
        """Performs a full extrapolation step on the primal and dual variables.

        Note that the forward and backward computations associated with the CMPState
        and Lagrangian are carried out twice, since we compute an "extra" gradient.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the ``compute_cmp_state`` method.
        """

        for call_extrapolation in [True, False]:
            self.zero_grad()
            cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)

            primal_lagrangian_store = cmp_state.compute_primal_lagrangian()
            dual_lagrangian_store = cmp_state.compute_dual_lagrangian()

            primal_lagrangian_store.backward()
            dual_lagrangian_store.backward()

            self.step(call_extrapolation=call_extrapolation)

        return cmp_state, primal_lagrangian_store, dual_lagrangian_store
