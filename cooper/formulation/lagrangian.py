"""Lagrangian formulation"""

import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, no_type_check

import torch

from cooper.multipliers import DenseMultiplier
from cooper.problem import CMPState, ConstrainedMinimizationProblem

from .formulation import Formulation


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
        cmp: Optional[ConstrainedMinimizationProblem] = None,
        ineq_init: Optional[torch.Tensor] = None,
        eq_init: Optional[torch.Tensor] = None,
    ):
        """Construct new `LagrangianFormulation`"""

        self.cmp = cmp

        self.ineq_multipliers = None
        self.eq_multipliers = None

        # Store user-provided initializations for dual variables
        self.ineq_init = ineq_init
        self.eq_init = eq_init

        self.accumulated_violation_dot_prod: torch.Tensor = None

    @property
    def dual_parameters(self) -> List[torch.Tensor]:
        """Returns a list gathering all dual parameters."""

        all_dual_params = []

        for mult in [self.ineq_multipliers, self.eq_multipliers]:
            if mult is not None:
                all_dual_params.extend(list(mult.parameters()))

        return all_dual_params

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

    def create_state_from_metadata(
        self,
        dtype: torch.dtype,
        device: torch.device,
        ineq_size: torch.Size = None,
        eq_size: torch.Size = None,
    ) -> None:
        """
        Initialize the formulation state based on the shape of the constraints.
        """

        assert ineq_size is not None or eq_size is not None

        dummy_cmp_state = CMPState()

        if ineq_size is not None:
            dummy_cmp_state.ineq_defect = torch.empty(
                size=ineq_size, dtype=dtype, device=device
            )

        if eq_size is not None:
            dummy_cmp_state.eq_defect = torch.empty(
                size=eq_size, dtype=dtype, device=device
            )

        self.create_state(dummy_cmp_state)

    def create_state(self, cmp_state: CMPState) -> None:
        """Initialize multipliers and optimizers given list of equality and
        inequality defects.

        Args:
            eq_defect: Defects for equality constraints
            ineq_defect: Defects for inequality constraints.
        """

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

    def update_accumulated_violation(self, update: Optional[torch.Tensor] = None):
        """
        Update the cumulative dot product between the constraint violations
        (aka defects) and the current multipliers.

        Args:
            update: Dot product between multipliers and constraints. If value is
                ``None``, then ``self.accumulated_violation_dot_prod`` is set to
                ``None``.
        """
        if update is None or self.accumulated_violation_dot_prod is None:
            self.accumulated_violation_dot_prod = update
        elif self.accumulated_violation_dot_prod is not None:
            self.accumulated_violation_dot_prod += update

    @abc.abstractmethod
    def weighted_violation(
        self, cmp_state: CMPState, constraint_type: str
    ) -> torch.Tensor:
        """
        Abstract method for computing the weighted violation of a constraint.

        Args:
            cmp_state: Current :py:class:`cooper.problem.CMPState`.
            constraint_type: Constraint type, either ``"eq"`` or ``"ineq"``.
        """
        pass

    def state_dict(self) -> Dict[str, Any]:
        """
        Generates the state dictionary for a Lagrangian formulation.
        """

        state_dict = {
            "ineq_multipliers": self.ineq_multipliers,
            "eq_multipliers": self.eq_multipliers,
            "accumulated_violation_dot_prod": self.accumulated_violation_dot_prod,
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Loads the state dictionary of a Lagrangian formulation.

        Args:
            state_dict: state dictionary to be loaded.
        """

        known_attrs = [
            "ineq_multipliers",
            "eq_multipliers",
            "accumulated_violation_dot_prod",
        ]
        for key, val in state_dict.items():
            assert (
                key in known_attrs
            ), "LagrangianFormulation received unknown key: {}".format(key)
            setattr(self, key, val)

    def flip_dual_gradients(self):
        """
        Flip the sign of the gradients of the dual variables.
        """
        for multiplier in self.state():
            if multiplier is not None:
                multiplier.grad.mul_(-1.0)


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
    """

    @no_type_check
    def compute_lagrangian(
        self,
        closure: Callable[..., CMPState] = None,
        *closure_args,
        pre_computed_state: Optional[CMPState] = None,
        write_state: Optional[bool] = True,
        **closure_kwargs
    ) -> torch.Tensor:
        """
        Computes the Lagrangian based on a new evaluation of the
        :py:class:`~cooper.problem.CMPState` via the ``closure`` function.

        If no explicit proxy-constraints are provided, we use the given
        inequality/equality constraints to compute the Lagrangian and to
        populate the primal and dual gradients. Note that gradients are _not_
        populated by this function, but rather :py:meth:`.backward`.

        In case proxy constraints are provided in the CMPState, the non-proxy
        constraints (potentially non-differentiable) are used for computing the
        value of the Lagrangian. The accumulated proxy-constraints are used in
        the backward computation triggered by
        :py:meth:`.backward` (and thus must be differentiable).

        Args:
            closure: Callable returning a :py:class:`cooper.problem.CMPState`
            pre_computed_state: Pre-computed CMP state to avoid wasteful
                computation when only dual gradients are required.
            write_state: If ``True``, the ``state`` of the formulation's
                :py:class:`cooper.problem.ConstrainedMinimizationProblem`
                attribute is replaced by that returned by the ``closure``
                argument. This flag can be used (when set to ``False``) to
                evaluate the Lagrangian, e.g. for logging validation metrics,
                without overwritting the information stored in the formulation's
                :py:class:`cooper.problem.ConstrainedMinimizationProblem`.

        """

        assert (
            closure is not None or pre_computed_state is not None
        ), "At least one of closure or pre_computed_state must be provided"

        if pre_computed_state is not None:
            cmp_state = pre_computed_state
        else:
            cmp_state = closure(*closure_args, **closure_kwargs)

        if write_state and self.cmp is not None:
            self.write_cmp_state(cmp_state)

        # Extract values from ProblemState object
        loss = cmp_state.loss

        if not self.is_state_created:
            # If not done before, instantiate and initialize dual variables
            self.create_state(cmp_state)

        # Purge previously accumulated constraint violations
        self.update_accumulated_violation(update=None)

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

        return lagrangian

    def weighted_violation(
        self, cmp_state: CMPState, constraint_type: str
    ) -> torch.Tensor:
        """
        Computes the dot product between the current multipliers and the
        constraint violations of type ``constraint_type``. If proxy-constraints
        are provided in the :py:class:`.CMPState`, the non-proxy (usually
        non-differentiable) constraints are used for computing the dot product,
        while the "proxy-constraint" dot products are accumulated under
        ``self.accumulated_violation_dot_prod``.

        Args:
            cmp_state: current ``CMPState``
            constraint_type: type of constrained to be used, e.g. "eq" or "ineq".
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
            # We should always have at least the "regular" defects, if not, then
            # the problem instance does not have `constraint_type` constraints
            proxy_violation = torch.tensor([0.0], device=cmp_state.loss.device)
        else:
            multipliers = getattr(self, constraint_type + "_multipliers")()

            # We compute (primal) gradients of this object
            proxy_violation = torch.sum(multipliers.detach() * proxy_defect)

            # This is the violation of the "actual/hard" constraint. We use this
            # to update the multipliers.
            # The gradients for the dual variables are computed via a backward
            # on `accumulated_violation_dot_prod`. This enables easy
            # extensibility to multiplier classes beyond DenseMultiplier.

            # TODO (JGP): Verify that call to backward is general enough for
            # Lagrange Multiplier models
            violation_for_update = torch.sum(multipliers * defect.detach())
            self.update_accumulated_violation(update=violation_for_update)

        return proxy_violation

    @no_type_check
    def backward(
        self,
        lagrangian: torch.Tensor,
        ignore_primal: bool = False,
        ignore_dual: bool = False,
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
            ignore_dual: If ``True``, the gradients with respect to the dual
                variables are not populated.
        """

        if ignore_primal:
            # Only compute gradients wrt Lagrange multipliers
            # No need to call backward on Lagrangian as the dual variables have
            # been detached when computing the `weighted_violation`s
            pass
        else:
            # Compute gradients wrt _primal_ parameters only.
            # The gradient for the dual variables is computed based on the
            # non-proxy violations below.
            lagrangian.backward()

        # Fill in the gradients for the dual variables based on the violation of
        # the non-proxy constraints
        if not ignore_dual:
            dual_vars = [_ for _ in self.state() if _ is not None]
            self.accumulated_violation_dot_prod.backward(inputs=dual_vars)


class ProxyLagrangianFormulation(BaseLagrangianFormulation):
    """
    Placeholder class for the proxy-Lagrangian formulation proposed by
    :cite:t:`cotter2019JMLR`.

    .. todo::
        Implement Proxy-Lagrangian formulation as described in
        :cite:t:`cotter2019JMLR`

    """

    pass
