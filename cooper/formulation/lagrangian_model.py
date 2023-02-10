from dataclasses import dataclass
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    no_type_check,
    Dict,
    Any,
    Iterator,
)

import torch

from .lagrangian import BaseLagrangianFormulation
from cooper.multipliers import MultiplierModel

@dataclass
class CMPModelState:
    """
    Represents the "state" of a Constrained Minimization Problem in terms of
    the value of its loss and constraint violations/defects. The main difference between
    this object and `CMPState` is that it also stores the features for the constraints
    that are going to be passed to the multiplier models to predict the Lagrange
    multipliers.

    Args:
        loss: Value of the loss or main objective to be minimized :math:`f(x)`
        ineq_defect: Violation of the inequality constraints :math:`g(x)`
        eq_defect: Violation of the equality constraints :math:`h(x)`
        proxy_ineq_defect: Differentiable surrogate for the inequality
            constraints as proposed by :cite:t:`cotter2019JMLR`.
        proxy_eq_defect: Differentiable surrogate for the equality constraints
            as proposed by :cite:t:`cotter2019JMLR`.
        ineq_constraint_features: Features for the inequality constraints that are
            going to be passed to the inequality multiplier model to predict the Lagrange
            multipliers.
        eq_constraint_features: Features for the equality constraints that are going to
            be passed to the equality multiplier model to predict the Lagrange multipliers.
        misc: Optional additional information to be store along with the state
            of the CMP
    """

    loss: Optional[torch.Tensor] = None
    ineq_defect: Optional[torch.Tensor] = None
    eq_defect: Optional[torch.Tensor] = None
    proxy_ineq_defect: Optional[torch.Tensor] = None
    proxy_eq_defect: Optional[torch.Tensor] = None
    ineq_constraint_features: Optional[torch.Tensor] = None
    eq_constraint_features: Optional[torch.Tensor] = None
    misc: Optional[dict] = None

    def as_tuple(self) -> tuple:
        return (
            self.loss,
            self.ineq_defect,
            self.eq_defect,
            self.proxy_ineq_defect,
            self.proxy_eq_defect,
            self.ineq_constraint_features,
            self.eq_constraint_features,
            self.misc,
        )


class LagrangianModelFormulation(BaseLagrangianFormulation):
    """
    Computes the Lagrangian based on the predictions of a `MultiplierModel`. This
    formulation is useful when the Lagrange multipliers are not kept explicitly, but
    are instead predicted by a model, e.i. neural network. This formulation is meant to
    be used along the :py:class:`~cooper.multipliers.MultiplierModel`.

    Attributes:
        ineq_multiplier_model: The model used to predict the Lagrange multipliers
            associated with the inequality constraints. If ``None``, the
            :py:meth:`~cooper.formulation.lagrangian_model.LagrangianModelFormulation.state`
            method will not return the Lagrange multipliers associated with the
            inequality constraints.
        eq_multiplier_model: The model used to predict the Lagrange multipliers
            associated with the equality constraints. If ``None``, the
            :py:meth:`~cooper.formulation.lagrangian_model.LagrangianModelFormulation.state`
            method will not return the Lagrange multipliers associated with the
            equality constraints.
        **kwargs: Additional keyword arguments to be passed to the
    """

    def __init__(
        self,
        *args,
        ineq_multiplier_model: Optional[MultiplierModel] = None,
        eq_multiplier_model: Optional[MultiplierModel] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.ineq_multiplier_model = ineq_multiplier_model
        self.eq_multiplier_model = eq_multiplier_model

        self.base_sanity_checks()

    def base_sanity_checks(self):
        """
        Perform sanity checks on the initialization of ``LagrangianModelFormulation``.
        """

        if self.ineq_multiplier_model is None and self.eq_multiplier_model is None:
            # This formulation cannot perform any prediction if no multiplier model is
            # provided.
            raise ValueError("At least one multiplier model must be provided.")

        if self.ineq_multiplier_model is not None and not isinstance(
            self.ineq_multiplier_model, MultiplierModel
        ):
            raise ValueError("The `ineq_multiplier_model` must be a `MultiplierModel`.")

        if self.eq_multiplier_model is not None and not isinstance(
            self.eq_multiplier_model, MultiplierModel
        ):
            raise ValueError("The `eq_multiplier_model` must be a `MultiplierModel`.")

    @property
    def dual_parameters(self) -> List[torch.Tensor]:
        """Returns a list gathering all dual parameters."""
        all_dual_params = []

        for mult in [self.ineq_multiplier_model, self.eq_multiplier_model]:
            if mult is not None:
                all_dual_params.extend(list(mult.parameters()))

        return all_dual_params

    def state(self) -> Tuple[Union[None, Iterator[torch.nn.Parameter]]]:

        """
        Collects all dual variables and returns a tuple containing their
        :py:class:`Iterator[torch.nn.Parameter]` values. Note that the *values*
        correspond to the parameters of the `MultiplierModel`.
        """

        if self.ineq_multiplier_model is None:
            ineq_state = None
        else:
            ineq_state = self.ineq_multiplier_model.parameters()

        if self.eq_multiplier_model is None:
            eq_state = None
        else:
            eq_state = self.eq_multiplier_model.parameters()

        return ineq_state, eq_state

    def create_state(self):
        """This method is not implemented for this formulation. It originally
        instantiates the dual variables, but in this formulation this is done since the
        instantiation of the object, since it is necessary to provide a `MultiplerModel`
        for each of the contraint types."""
        pass

    @property
    def is_state_created(self):
        """
        Returns ``True`` if any Multiplier Model have been initialized.
        """
        return (
            self.ineq_multiplier_model is not None
            or self.eq_multiplier_model is not None
        )

    def state_dict(self) -> Dict[str, Any]:
        """
        Generates the state dictionary for a Lagrangian model formulation.
        """

        state_dict = {
            "ineq_multiplier_model": self.ineq_multiplier_model.state_dict(),
            "eq_multiplier_model": self.eq_multiplier_model.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Loads the state dictionary of a Lagrangian formulation.

        Args:
            state_dict: state dictionary to be loaded.
        """

        known_attrs = ["ineq_multiplier_model", "eq_multiplier_model"]

        for key, val in state_dict.items():

            if key not in known_attrs:
                raise ValueError(
                    f"LagrangianModelFormulation received unknown key: {key}. Valid keys are {known_attrs}"
                )

            if key in ["ineq_multiplier_model", "eq_multiplier_model"]:
                multiplier_model = getattr(self, key)
                multiplier_model.load_state_dict(val)

    def flip_dual_gradients(self) -> None:
        """
        Flips the sign of the gradients for the dual variables. This is useful
        when using the dual formulation in conjunction with the alternating
        update scheme.
        """

        for multiplier_model_state in self.state():
            if multiplier_model_state is not None:
                for param in multiplier_model_state:
                    if param.grad is not None:
                        param.grad.mul_(-1.0)

    @no_type_check
    def compute_lagrangian(
        self,
        closure: Callable[..., CMPModelState] = None,
        *closure_args,
        pre_computed_state: Optional[CMPModelState] = None,
        write_state: bool = True,
        **closure_kwargs,
    ) -> torch.Tensor:
        """
        Computes the Lagrangian of the problem, given the current state of the
        optimization problem. This method is used to compute the loss function
        for the optimization problem.

        Args:
            closure: A function that returns a :py:class:`CMPModelState` object. This
                function is used to compute the loss function and the constraint
                violations. If ``None``, the ``pre_computed_state`` argument must be
                provided.
            *closure_args: Positional arguments to be passed to the ``closure``
                function.
            pre_computed_state: A :py:class:`CMPModelState` object containing the
                pre-computed loss function and constraint violations. If ``None``,
                the ``closure`` argument must be provided.
            write_state: If ``True``, the state of the optimization problem is
                written to the ``cmp_model_state`` attribute of the :py:class:`CMPModelState`
                object.
            **closure_kwargs: Keyword arguments to be passed to the ``closure``
                function.
        """

        assert (
            closure is not None or pre_computed_state is not None
        ), "At least one of closure or pre_computed_state must be provided"

        if pre_computed_state is not None:
            cmp_model_state = pre_computed_state
        else:
            cmp_model_state = closure(*closure_args, **closure_kwargs)

        if write_state:
            self.write_cmp_state(cmp_model_state)

        # Extract values from ProblemState object
        loss = cmp_model_state.loss

        # Purge previously accumulated constraint violations
        self.update_accumulated_violation(update=None)

        # Compute contribution of the sampled constraint violations, weighted by the
        # current multiplier values predicted by the multuplier model.
        ineq_viol = self.weighted_violation(cmp_model_state, "ineq")
        eq_viol = self.weighted_violation(cmp_model_state, "eq")

        # Lagrangian = loss + \sum_i multiplier_i * defect_i
        lagrangian = loss + ineq_viol + eq_viol

        return lagrangian

    def weighted_violation(
        self, cmp_model_state: CMPModelState, constraint_type: str
    ) -> torch.Tensor:
        """
        Computes the dot product between the current multipliers and the
        constraint violations of type ``constraint_type``. The multiplier correspond to
        the output of a `MultiplierModel` provided when the formulation was initialized.
        The model is trained on `constraint_features` provided in the CMPModelState.
        If proxy-constraints are provided in the :py:class:`.CMPModelState`, the non-
        proxy (usually non-differentiable) constraints are used for computing the dot
        product, while the "proxy-constraint" dot products are accumulated under
        ``self.accumulated_violation_dot_prod``.

        Args:
            cmp_model_state: current ``CMPModelState``
            constraint_type: type of constrained to be used, e.g. "eq" or "ineq".
        """

        defect = getattr(cmp_model_state, constraint_type + "_defect")
        has_defect = defect is not None

        proxy_defect = getattr(cmp_model_state, "proxy_" + constraint_type + "_defect")
        has_proxy_defect = proxy_defect is not None

        if not has_proxy_defect:
            # If not given proxy constraints, then the regular defects are
            # used for computing gradients and evaluating the multipliers
            proxy_defect = defect

        if not has_defect:
            # We should always have at least the "regular" defects, if not, then
            # the problem instance does not have `constraint_type` constraints
            violation = torch.tensor([0.0], device=cmp_model_state.loss.device)
        else:
            multiplier_model = getattr(self, constraint_type + "_multiplier_model")

            # Get multipliers by performing a prediction over the features of the
            # sampled constraints
            constraint_features = getattr(cmp_model_state, constraint_type + "_constraint_features")

            multipliers = multiplier_model.forward(constraint_features)

            # The violations are computed via inner product between the multipliers
            # and the defects, they should have the same shape. If given proxy-defects
            # then their shape has to be checked as well.
            assert defect.shape == proxy_defect.shape == multipliers.shape

            # Store the multiplier values
            setattr(self, constraint_type + "_multipliers", multipliers)

            # We compute (primal) gradients of this object with the sampled
            # constraints
            violation = torch.sum(multipliers.detach() * proxy_defect)

            # This is the violation of the "actual/hard" constraint. We use this
            # to update the multipliers.
            # The gradients for the dual variables are computed via a backward
            # on `accumulated_violation_dot_prod`. This enables easy
            # extensibility to multiplier classes beyond DenseMultiplier.

            # TODO (gallego-posada): Verify that call to backward is general enough for
            # Lagrange Multiplier models
            violation_for_update = torch.sum(multipliers * defect.detach())
            self.update_accumulated_violation(update=violation_for_update)

        return violation

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
            dual_vars = self.dual_parameters
            self.accumulated_violation_dot_prod.backward(inputs=dual_vars)
