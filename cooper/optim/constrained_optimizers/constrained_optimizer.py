# coding: utf8
"""
Implementation of the :py:class:`ConstrainedOptimizer` class.
"""

import warnings

from cooper.formulation import AugmentedLagrangianFormulation

from .cooper_optimizer import CooperOptimizer, CooperOptimizerState


class ConstrainedOptimizer(CooperOptimizer):

    """
    Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    given a provided :py:class:`~cooper.formulation.Formulation`.

    A ``ConstrainedOptimizer`` includes one or more
    :class:`torch.optim.Optimizer`\\s for the primal variables. It also includes
    a :class:`torch.optim.Optimizer` for the dual variables associated with the
    provided ``Formulation``.

    For handling unconstrained problems, we provide an
    :py:class:`~cooper.optim.cooper_optimizer.UnconstrainedOptimizer`. Please
    refer to the documentation of the
    :py:class:`~cooper.problem.ConstrainedMinimizationProblem` and
    :py:class:`~cooper.formulation.Formulation` classes for further details on
    handling unconstrained problems.

    Args:
        formulation: ``Formulation`` of the ``ConstrainedMinimizationProblem``
            to be optimized.

        primal_optimizers: Fully instantiated ``torch.optim.Optimizer``\\s used
            to optimize the primal parameters (e.g. model parameters). The primal
            parameters can be partitioned into multiple optimizers, in this case
            ``primal_optimizers`` accepts a list of ``torch.optim.Optimizer``\\s.

        dual_optimizer: Partially instantiated ``torch.optim.Optimizer``
            used to optimize the dual variables (e.g. Lagrange multipliers).
            If dealing with an unconstrained problem, please use a
            :py:class:`~cooper.optim.cooper_optimizer.UnconstrainedOptimizer`
            instead.

        dual_scheduler: Partially instantiated
            ``torch.optim.lr_scheduler._LRScheduler`` used to schedule the
            learning rate of the dual variables. Defaults to None.

        extrapolation: Whether to perform extragradient updates. Defaults to False.

        alternating: Whether to alternate parameter updates between primal and
            dual parameters. Otherwise, do simultaneous parameter updates.
            Defaults to False.

        dual_restarts: If True, perform "restarts" on the Lagrange
            multipliers associated with inequality constraints: whenever the
            constraint is satisfied, directly set the multiplier to zero.
            Defaults to False.

    """

    def base_sanity_checks(self):
        """
        Perform sanity checks on the initialization of ``ConstrainedOptimizer``.
        """

        if self.dual_optimizer is None:
            raise RuntimeError("No dual optimizer was provided.")

        if self.alternating and self.dual_restarts:
            warnings.warn(
                """Using alternating updates with dual restarts is untested.
                Please use with caution."""
            )

        if isinstance(self.formulation, AugmentedLagrangianFormulation):
            if not self.alternating:
                raise RuntimeError(
                    "Augmented Lagrangian formulation requires alternating updates."
                )

    def instantiate_dual_optimizer_and_scheduler(self):
        """Instantiates the dual optimizer and scheduler."""

        # Makes sure that dual optimizer indeed requires to be initialized
        assert self.dual_optimizer is not None and callable(self.dual_optimizer)

        # Checks if needed and instantiates dual_optimizer
        self.dual_optimizer = self.dual_optimizer(self.formulation.dual_parameters)

        if self.dual_scheduler is not None:
            assert callable(self.dual_scheduler), "dual_scheduler must be callable"
            # Instantiates the dual_scheduler
            self.dual_scheduler = self.dual_scheduler(self.dual_optimizer)

    def restart_dual_variables(self):
        """
        Execute restart for multipliers associated with inequality constraints
        """
        self.formulation.ineq_multipliers.restart_if_feasible_()

    def zero_grad(self, ignore_primal: bool = False, ignore_dual: bool = False):
        """
        Sets the gradients of all optimized
        :py:class:`~torch.nn.parameter.Parameter`\\s to zero. This includes both
        the primal and dual variables.

        Args:
            ignore_primal: If True, the gradients of the primal variables will
                not be zeroed. Defaults to False.

            ignore_dual: If True, the gradients of the dual variables will not
                be zeroed. Defaults to False.
        """

        if not ignore_primal:
            for primal_optimizer in self.primal_optimizers:
                primal_optimizer.zero_grad()

        if not ignore_dual:

            if self.formulation.is_state_created:
                if self.dual_optimizer is None:
                    raise RuntimeError(
                        "Requested zeroing gradients but dual_optimizer is None."
                    )
                else:
                    self.dual_optimizer.zero_grad()

    def state_dict(self) -> CooperOptimizerState:
        """
        Returns the state of the ConstrainedOptimizer. See
        :py:class:`~cooper.optim.constrained_optimizers.cooper_optimizer.CooperOptimizerState`.
        """

        primal_optimizer_states = [_.state_dict() for _ in self.primal_optimizers]
        dual_optimizer_state = self.dual_optimizer.state_dict()

        if self.dual_scheduler is not None:
            dual_scheduler_state = self.dual_scheduler.state_dict()
        else:
            dual_scheduler_state = None

        return CooperOptimizerState(
            primal_optimizer_states=primal_optimizer_states,
            dual_optimizer_state=dual_optimizer_state,
            dual_scheduler_state=dual_scheduler_state,
            extrapolation=self.extrapolation,
            alternating=self.alternating,
            dual_restarts=self.dual_restarts,
        )
