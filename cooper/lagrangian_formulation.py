"""Lagrangian formulation"""

import abc
from typing import Callable, List, Optional, Tuple, Union, no_type_check

import torch

from .multipliers import DenseMultiplier
from .problem import CMPState, ConstrainedMinimizationProblem, Formulation


class BaseLagrangianFormulation(Formulation, metaclass=abc.ABCMeta):
    """
    Base class for Lagrangian Formulations.

    Attributes:
        cmp: :py:class:`~cooper.problem.ConstrainedMinimizationProblem` we aim
            to solve and which gives rise to the Lagrangian.
        ineq_multipliers: Trainable
            :py:class:`cooper.multipliers.DenseMultiplier`\\s associated with
            the inequality constraints.
        eq_multipliers: Trainable
            :py:class:`cooper.multipliers.DenseMultiplier`\\s associated with
            the equality constraints.
    """

    def __init__(
        self,
        cmp: ConstrainedMinimizationProblem,
        ineq_init: Optional[torch.Tensor] = None,
        eq_init: Optional[torch.Tensor] = None,
        aug_lag_coefficient: float = 0.0,
    ):
        """Construct new `LagrangianFormulation`"""

        self.cmp = cmp

        self.ineq_multipliers = None
        self.eq_multipliers = None

        # Store user-provided initializations for dual variables
        self.ineq_init = ineq_init
        self.eq_init = eq_init

        self.state_update: List[torch.Tensor] = []

        if aug_lag_coefficient < 0:
            raise ValueError("Augmented Lagrangian coefficient must be non-negative.")
        self.aug_lag_coefficient = aug_lag_coefficient

    @property
    def dual_parameters(self) -> List[torch.Tensor]:
        """Returns a list gathering all dual parameters"""
        return [_ for _ in self.state() if _ is not None]

    def state(self) -> Tuple[Union[None, torch.Tensor]]:
        """
        Collects all dual variables and returns a tuple containing their
        :py:class:`torch.Tensor` values. Note that the *values* are a different
        type from the :py:class:`cooper.multipliers.DenseMultiplier` objects.
        """
        if self.ineq_multipliers is None:
            ineq_state = None
        else:
            ineq_state = self.ineq_multipliers()

        if self.eq_multipliers is None:
            eq_state = None
        else:
            eq_state = self.eq_multipliers()

        return ineq_state, eq_state  # type: ignore

    def create_state(self, cmp_state):
        """Initialize dual variables and optimizers given list of equality and
        inequality defects. :py:class:`cooper.multipliers.DenseMultiplier`

        Args:
            eq_defect: Defects for equality constraints
            ineq_defect: Defects for inequality constraints.
        """

        # Ensure that dual variables are not re-initialized
        for constraint_type in ["eq", "ineq"]:

            mult_name = constraint_type + "_multipliers"

            defect = getattr(cmp_state, constraint_type + "_defect")
            proxy_defect = getattr(cmp_state, "proxy_" + constraint_type + "_defect")

            has_defect = defect is not None
            has_proxy_defect = proxy_defect is not None

            if has_defect or has_proxy_defect:

                # Ensure dual variables have not been initialized previously
                assert getattr(self, constraint_type + "_multipliers") is None

                # If given proxy and non-proxy defects, sanity-check shapes
                if has_defect and has_proxy_defect:
                    assert defect.shape == proxy_defect.shape

                # Choose a tensor for getting device and dtype information
                defect_for_init = defect if has_defect else proxy_defect

                init_tensor = getattr(self, constraint_type + "_init")
                if init_tensor is None:
                    # If not provided custom initialization, Lagrange
                    # multipliers are initialized at 0

                    # This already preserves dtype and device of defect
                    casted_init = torch.zeros_like(defect_for_init)
                else:
                    casted_init = torch.tensor(
                        init_tensor,
                        device=defect_for_init.device,
                        dtype=defect_for_init.dtype,
                    )
                    assert defect_for_init.shape == casted_init.shape

                # Enforce positivity if dealing with inequality
                is_positive = constraint_type == "ineq"
                multiplier = DenseMultiplier(casted_init, positive=is_positive)

                setattr(self, mult_name, multiplier)

    @property
    def is_state_created(self):
        """
        Returns ``True`` if any Lagrange multipliers have been initialized.
        """
        return self.ineq_multipliers is not None or self.eq_multipliers is not None

    def purge_state_update(self):
        self.state_update = []

    def weighted_violation(
        self, cmp_state: CMPState, constraint_type: str
    ) -> torch.Tensor:
        """
        Computes the dot product between the current multipliers and the
        constraint violations of type ``constraint_type``. If proxy-constraints
        are provided in the :py:class:`.CMPState`, the non-proxy (usually
        non-differentiable) constraints are used for computing the dot product,
        while the "proxy-constraint" dot products are stored under
        ``self.state_update``.

        Args:
            cmp_state: current ``CMPState``
            constraint_type: type of constrained to be used

        """

        defect = getattr(cmp_state, constraint_type + "_defect")
        has_defect = defect is not None

        proxy_defect = getattr(cmp_state, "proxy_" + constraint_type + "_defect")
        has_proxy_defect = proxy_defect is not None

        if not has_proxy_defect:
            # If not given proxy constraints, then the regular defects are
            # used for computing gradients and evaluating the multipliers
            proxy_defect = defect

        if not has_defect:
            # We should always have at least the regular defects, if not, then
            # the problem instance does not have `constraint_type` constraints
            proxy_violation = torch.tensor([0.0], device=cmp_state.loss.device)
        else:
            multipliers = getattr(self, constraint_type + "_multipliers")()

            # We compute (primal) gradients of this object
            proxy_violation = torch.sum(multipliers.detach() * proxy_defect)

            # This is the violation of the "actual" constraint. We use this
            # to update the value of the multipliers by lazily filling the
            # multiplier gradients in `populate_gradients`
            violation_for_update = torch.sum(multipliers * defect.detach())
            self.state_update.append(violation_for_update)

        return proxy_violation


class LagrangianFormulation(BaseLagrangianFormulation):
    """
    Provides utilities for computing the Lagrangian associated with a
    ``ConstrainedMinimizationProblem`` and for populating the gradients for the
    primal and dual parameters.

    Args:
        cmp: ``ConstrainedMinimizationProblem`` we aim to solve and which gives
            rise to the Lagrangian.
        ineq_init: Initialization values for the inequality multipliers.
        eq_init: Initialization values for the equality multipliers.
        aug_lag_coefficient: Coefficient used for the augmented Lagrangian.
    """

    @no_type_check
    def composite_objective(
        self,
        closure: Callable[..., CMPState],
        *closure_args,
        write_state: bool = True,
        **closure_kwargs
    ) -> torch.Tensor:
        """
        Computes the Lagrangian based on a new evaluation of the
        :py:class:`~cooper.problem.CMPState``.

        If no explicit proxy-constraints are provided, we use the given
        inequality/equality constraints to compute the Lagrangian and to
        populate the primal and dual gradients.

        In case proxy constraints are provided in the CMPState, the non-proxy
        constraints (potentially non-differentiable) are used for computing the
        Lagrangian, while the proxy-constraints are used in the backward
        computation triggered by :py:meth:`._populate_gradient` (and thus must
        be differentiable).

        Args:
            closure: Callable returning a :py:class:`cooper.problem.CMPState`
            write_state: If ``True``, the ``state`` of the formulation's
                :py:class:`cooper.problem.ConstrainedMinimizationProblem`
                attribute is replaced by that returned by the ``closure``
                argument. This flag can be used (when set to ``False``) to
                evaluate the Lagrangian, e.g. for logging validation metrics,
                without overwritting the information stored in the formulation's
                :py:class:`cooper.problem.ConstrainedMinimizationProblem`.

        """

        cmp_state = closure(*closure_args, **closure_kwargs)
        if write_state:
            self.cmp.state = cmp_state

        # Extract values from ProblemState object
        loss = cmp_state.loss
        ineq_defect, eq_defect = cmp_state.ineq_defect, cmp_state.eq_defect

        if self.cmp.is_constrained and (not self.is_state_created):
            # If not done before, instantiate and initialize dual variables
            self.create_state(cmp_state)

        # Compute Lagrangian based on current loss and values of multipliers
        self.purge_state_update()

        if self.cmp.is_constrained:
            # Compute contribution of the constraint violations, weighted by the
            # current multiplier values

            # If given proxy constraints, these are used to compute the terms
            # added to the Lagrangian, and the multiplier updates are based on
            # the non-proxy violations.
            # If not given proxy constraints, then gradients and multiplier
            # updates are based on the "regular" constraints.
            ineq_viol = self.weighted_violation(cmp_state, "ineq")
            eq_viol = self.weighted_violation(cmp_state, "eq")

            # Lagrangian = loss + \sum_i multiplier_i * defect_i
            lagrangian = loss + ineq_viol + eq_viol

            # TODO(JGP): [1] verify that current implementation of proxy
            # constraints works properly with augmented lagrangian below.

            # If using augmented Lagrangian, add squared sum of constraints
            # Following the formulation on Marc Toussaint slides (p 17-20)
            # https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/03-constrainedOpt.pdf
            if self.aug_lag_coefficient > 0:

                # TODO(JGP): [2] I guess one would like to filter based on
                # non-proxy feasibility but penalize based on the proxy defect
                if ineq_defect is not None:
                    assert self.ineq_multipliers is not None
                    ineq_filter = (ineq_defect >= 0) + (self.ineq_multipliers() > 0)
                    ineq_square = torch.sum(torch.square(ineq_defect[ineq_filter]))
                else:
                    ineq_square = torch.tensor([0.0])

                if eq_defect is not None:
                    eq_square = torch.sum(torch.square(eq_defect))
                else:
                    eq_square = torch.tensor([0.0])

                lagrangian += self.aug_lag_coefficient * (ineq_square + eq_square)

        else:
            assert cmp_state.loss is not None
            lagrangian = cmp_state.loss

        return lagrangian

    @no_type_check
    def _populate_gradients(
        self, lagrangian: torch.Tensor, ignore_primal: bool = False
    ):
        """
        Performs the actual backward computation which populates the gradients
        for the primal and dual variables.

        Args:
            lagrangian: Value of the computed Lagrangian based on which the
                gradients for the primal and dual variables are populated.
            ignore_primal: If ``True``, only the gradients with respect to the
                dual variables are populated (these correspond to the constraint
                violations). This feature is mainly used in conjunction with
                ``alternating`` updates, which require updating the multipliers
                based on the constraints violation *after* having updated the
                primal parameters. Defaults to False.
        """

        if ignore_primal and self.cmp.is_constrained:
            # Only compute gradients wrt Lagrange multipliers
            # No need to call backward on Lagrangian as the dual variables have
            # been detached when computing the `weighted_violation`s
            pass
        else:
            # Compute gradients wrt primal parameters only.
            # The gradient for the dual variables is computed based on the
            # non-proxy violations below.
            lagrangian.backward()

        # Fill in the gradients for the dual variables based on the violation of
        # the non-proxy constraint
        # This is equivalent to setting `dual_vars.grad = defect`
        if self.cmp.is_constrained:
            for violation_for_update in self.state_update:
                dual_vars = [_ for _ in self.state() if _ is not None]
                violation_for_update.backward(inputs=dual_vars)


class ProxyLagrangianFormulation(BaseLagrangianFormulation):
    """
    Placeholder class for the proxy-Lagrangian formulation proposed by
    :cite:t:`cotter2019JMLR`.

    .. todo::
        Implement Proxy-Lagrangian formulation as described in
        :cite:t:`cotter2019JMLR`

    """

    pass
