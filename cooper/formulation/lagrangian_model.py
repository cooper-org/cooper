from typing import Callable, no_type_check, Tuple, Optional, Union, List

from cooper.multipliers import MultiplierModel
from cooper.formulation.lagrangian import BaseLagrangianFormulation
from cooper.problem import CMPState

import torch


class LagrangianModelFormulation(BaseLagrangianFormulation):
    """
    Computes the Lagrangian based on the predictions of a `MultiplierModel`.
    """

    def __init__(
        self,
        ineq_multiplier_model: Optional[MultiplierModel] = None,
        eq_multiplier_model: Optional[MultiplierModel] = None,
        *args,
        **kwargs,
        ):
        super().__init__(*args, **kwargs)

        self.ineq_multiplier_model = ineq_multiplier_model
        self.eq_multiplier_model = eq_multiplier_model


        if self.ineq_multiplier_model is None and self.eq_multiplier_model is None:
            # TODO: document this
            raise ValueError("At least one multiplier model must be provided.")

        elif (not isinstance(self.ineq_multiplier_model, MultiplierModel) and
            not isinstance(self.eq_multiplier_model, MultiplierModel)):
            # TODO: document this
            raise ValueError("Multiplier models must be instances of MultiplierModel.")


        # TODO: ensure output of multiplier model is 1 dim

    def create_state(
            self,
            cmp_state: CMPState,
            mult_model: torch.nn.Module,
    ):
        """
        """
        pass

        # for constraint_type in ["eq", "ineq"]:

        #     mult_model_name = constraint_type + "_multiplier_model"

        #     defect = getattr(cmp_state, constraint_type + "_defect")
        #     proxy_defect = getattr(cmp_state, "proxy_" + constraint_type + "_defect")

        #     has_defect = defect is not None
        #     has_proxy_defect = proxy_defect is not None

        #     if has_defect or has_proxy_defect:

        #         # Ensure dual variables have not been initialized previously
        #         assert getattr(self, mult_model_name) is None

        #         # If given proxy and non-proxy defects, sanity-check shapes
        #         if has_defect and has_proxy_defect:
        #             assert defect.shape == proxy_defect.shape

        #         # TODO: Ensure model's output is 1 dim

        #         # Choose a tensor for getting device and dtype information
        #         defect_for_init = defect if has_defect else proxy_defect

        #         casted_init_model = mult_model

        #         # Enforce positivity if dealing with inequality
        #         is_positive = constraint_type == "ineq"
        #         multiplier = MultiplierModel(casted_init_model, positive=is_positive)

        #         setattr(self, mult_model_name, multiplier)

    @property
    def is_state_created(self):
        """
        Returns ``True`` if any Lagrange multipliers have been initialized.
        """
        return self.ineq_multiplier_model is not None or self.eq_multiplier_model is not None

    def state(
            self,
            eq_features: Optional[torch.Tensor],
            ineq_features: Optional[torch.Tensor]
    ) -> Tuple[Union[None, torch.Tensor]]:

        # TODO: fix this
        """
        Collects all dual variables and returns a tuple containing their
        :py:class:`torch.Tensor` values. Note that the *values* are a different
        type from the :py:class:`cooper.multipliers.DenseMultiplier` objects.
        """

        assert eq_features is not None or ineq_features is not None

        if ineq_features is None:
            ineq_state = None
        else:
            ineq_state = self.ineq_multiplier_model.forward(ineq_features)

        if eq_features is None:
            eq_state = None
        else:
            eq_state = self.eq_multiplier_model.forward(eq_features)

        return ineq_state, eq_state

    def flip_dual_gradients(self):
        """
        Flips the sign of the gradients for the dual variables. This is useful
        when using the dual formulation in conjunction with the alternating
        update scheme.
        """
        # TODO: implement in every formulation
        for constraint_type in ["eq", "ineq"]:
            mult_name = constraint_type + "_multipliers"
            multiplier = getattr(self, mult_name)
            if multiplier is not None:
                for param_grad in multiplier.grad():
                    if param_grad is not None:
                        param_grad._mul(-1.0)

    @no_type_check
    def composite_objective(
        self,
        closure: Callable[..., CMPState] = None,
        *closure_args,
        pre_computed_state: Optional[CMPState] = None,
        write_state: bool = True,
        **closure_kwargs
    ) -> torch.Tensor:
        """
        """

        assert (
            closure is not None or pre_computed_state is not None
        ), "At least one of closure or pre_computed_state must be provided"

        if pre_computed_state is not None:
            cmp_state = pre_computed_state
        else:
            cmp_state = closure(*closure_args, **closure_kwargs)

        if write_state:
            self.cmp.state = cmp_state

        # Extract values from ProblemState object
        loss = cmp_state.loss

        # if not self.is_state_created:
        #     # If not done before, instantiate and initialize dual variables
        #     self.create_state(cmp_state, multiplier_model_constructor)

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
            mult_model = getattr(self, constraint_type + "_multiplier_model")

            # Get multipliers by performing a prediction over the features of the
            # sampled constraints
            constraint_features = cmp_state.misc[constraint_type + "_constraint_features"]
            multipliers = mult_model.forward(constraint_features)

            # Get sampled defects. It is assumed that the provided defects correspond to
            # the sampled feature constraints
            defects = getattr(cmp_state, constraint_type + "_defect")

            # We compute (primal) gradients of this object with the sampled
            # constraints
            proxy_violation = torch.sum(multipliers.detach() * defects)

            # This is the violation of the "actual/hard" constraint. We use this
            # to update the multipliers.
            # The gradients for the dual variables are computed via a backward
            # on `accumulated_violation_dot_prod`. This enables easy
            # extensibility to multiplier classes beyond DenseMultiplier.

            # TODO (JGP): Verify that call to backward is general enough for
            # Lagrange Multiplier models
            violation_for_update = torch.sum(multipliers * defects.detach())
            self.update_accumulated_violation(update=violation_for_update)

        return proxy_violation

    def dual_parameters(self) -> List[torch.Tensor]:
        """Returns a list gathering all dual parameters."""

        all_dual_params = []

        for mult in [self.ineq_multiplier_model, self.eq_multiplier_model]:
            if mult is not None:
                all_dual_params.extend(list(mult.parameters()))

        return all_dual_params

    @no_type_check
    def _populate_gradients(
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
            dual_vars = self.dual_parameters()
            self.accumulated_violation_dot_prod.backward(inputs=dual_vars)
