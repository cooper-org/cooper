"""Implementation of the :py:class:`ExtrapolationConstrainedOptimizer` class."""

import torch

from cooper.optim.constrained_optimizers.constrained_optimizer import ConstrainedOptimizer
from cooper.optim.optimizer import RollOut


class ExtrapolationConstrainedOptimizer(ConstrainedOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing extrapolation updates to the primal and dual variables.
    """

    # TODO(gallego-posada): Add equations to illustrate the extrapolation updates

    def custom_sanity_checks(self):
        """Perform sanity checks on the initialization of
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

    @torch.no_grad()
    def primal_extrapolation_step(self):
        """Perform an extrapolation step on the parameters associated with the primal variables."""
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.extrapolation()

    @torch.no_grad()
    def dual_extrapolation_step(self):
        """Perform an extrapolation step on the parameters associated with the dual variables.

        After being updated by the dual optimizer steps, the multipliers are
        post-processed (e.g. to ensure non-negativity for inequality constraints).
        """
        # Update multipliers based on current constraint violations (gradients)
        # For unobserved constraints the gradient is None, so this is a no-op.
        for dual_optimizer in self.dual_optimizers:
            dual_optimizer.extrapolation()

        for multiplier in self.cmp.multipliers():
            multiplier.post_step_()

    def roll(self, compute_cmp_state_kwargs: dict = None) -> RollOut:
        """Performs a full extrapolation step on the primal and dual variables.

        Note that the forward and backward computations associated with the CMPState
        and Lagrangian are carried out twice, since we compute an "extra" gradient.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the ``compute_cmp_state`` method.
        """
        if compute_cmp_state_kwargs is None:
            compute_cmp_state_kwargs = {}
        for call_extrapolation in (True, False):
            self.zero_grad()
            cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)

            primal_lagrangian_store = cmp_state.compute_primal_lagrangian()
            dual_lagrangian_store = cmp_state.compute_dual_lagrangian()

            primal_lagrangian_store.backward()
            dual_lagrangian_store.backward()

            if call_extrapolation:
                self.primal_extrapolation_step()
                self.dual_extrapolation_step()
            else:
                self.primal_step()
                self.dual_step()

        return RollOut(cmp_state.loss, cmp_state, primal_lagrangian_store, dual_lagrangian_store)
