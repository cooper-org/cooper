import copy

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import style_utils
import torch
from IPython.display import display
from ipywidgets import HBox, Layout, VBox, fixed, interactive
from matplotlib.gridspec import GridSpec

import cooper
from cooper import ConstraintGroup, ConstraintType
from cooper.optim import SimultaneousOptimizer


class Toy2DWidget:
    def __init__(
        self,
        cmp_class,
        cmp_kwargs=None,
        problem_type=None,
        epsilon=None,
        primal_lr=None,
        dual_lr=None,
        primal_optim=None,
        dual_optim=None,
        num_iters=None,
        x=None,
        y=None,
        extrapolation=None,
        dual_restarts=None,
    ):
        if cmp_kwargs is None:
            constraint_group = ConstraintGroup(
                constraint_type=ConstraintType.INEQUALITY, multiplier_kwargs={"shape": 1, "device": "cpu"}
            )
            cmp_kwargs = {"ineq_group": constraint_group}

        # --------------------------------------- Create some control elements
        if problem_type is None:
            problem_type_dropdown = widgets.Dropdown(
                options=["Convex", "Concave"],
                description="Problem type",
            )
        else:
            problem_type_dropdown = fixed(problem_type)

        if epsilon is None:
            epsilon_slider = widgets.FloatSlider(min=-0.2, max=1.5, step=0.05, value=0.7, description="Const. level")
        else:
            epsilon_slider = fixed(epsilon)

        if primal_lr is None:
            primal_lr_slider = widgets.FloatLogSlider(
                base=10,
                min=-4,
                max=0,
                step=0.1,
                value=2e-2,
                description="Primal LR",
                continuous_update=False,
            )
        else:
            primal_lr_slider = fixed(primal_lr)

        if primal_optim is None:
            primal_optim_dropdown = widgets.Dropdown(
                value="SGD",
                options=["SGD", "SGDM_0.9", "Adam"],
                description="Primal opt.",
            )
        else:
            primal_optim_dropdown = fixed(primal_optim)

        if dual_lr is None:
            dual_lr_slider = widgets.FloatLogSlider(
                base=10,
                min=-4,
                max=0,
                step=0.1,
                value=5e-1,
                description="Dual LR",
                continuous_update=False,
            )
        else:
            dual_lr_slider = fixed(dual_lr)

        if dual_optim is None:
            dual_optim_dropdown = widgets.Dropdown(
                value="SGD",
                options=["SGD", "SGDM_0.9", "Adam"],
                description="Dual opt.",
            )
        else:
            dual_optim_dropdown = fixed(dual_optim)

        if x is None:
            x_slider = widgets.FloatSlider(
                min=0,
                max=np.pi / 2,
                step=0.01,
                value=0.9,
                description="x init.",
                continuous_update=False,
            )
        else:
            x_slider = fixed(x)

        if y is None:
            y_slider = widgets.FloatSlider(
                min=0,
                max=3.0,
                step=0.01,
                value=2.0,
                description="y init.",
                continuous_update=False,
            )
        else:
            y_slider = fixed(y)

        num_iters = 300 if num_iters is None else num_iters
        iters_textbox = widgets.IntSlider(min=100, max=3000, value=num_iters, step=100, description="Max Iters")

        if dual_restarts is None:
            restarts_checkbox = widgets.Checkbox(value=False, description="Dual restarts")
        else:
            restarts_checkbox = fixed(dual_restarts)

        if extrapolation is None:
            extrapolation_checkbox = widgets.Checkbox(value=False, description="Extrapolation")
        else:
            extrapolation_checkbox = fixed(extrapolation)

        # --------------------------------- Indicate what each option observes
        widget = interactive(
            self.update,
            x=x_slider,
            y=y_slider,
            num_iters=iters_textbox,
            epsilon=epsilon_slider,
            problem_type=problem_type_dropdown,
            primal_lr=primal_lr_slider,
            primal_optim=primal_optim_dropdown,
            dual_lr=dual_lr_slider,
            dual_optim=dual_optim_dropdown,
            dual_restarts=restarts_checkbox,
            extrapolation=extrapolation_checkbox,
        )
        controls_layout = Layout(
            display="flex",
            flex_flow="row wrap",
            border="solid 2px",
            justify_content="space-around",
            align_items="center",
            align_content="space-around",
            max_width="1050px",
        )
        controls = HBox(widget.children[:-1], layout=controls_layout)
        output = widget.children[-1]
        display(VBox([controls, output]))

        # ------------------------------ Initialize the CMP and its formulation
        self.cmp = cmp_class(**cmp_kwargs)
        self.ineq_group = cmp_kwargs["ineq_group"]

        # # Run the update a first time
        widget.update()

    def reset_problem(self, epsilon=None, problem_type="convex"):
        """Reset the cmp and formulation for new training loops."""

        self.cmp.problem_type = problem_type

        # Reset the state of the CMP. Update epsilon if necessary.
        self.cmp.epsilon = epsilon
        self.cmp.state = None

        # Reset multipliers
        # self.ineq_group.multiplier = None

    def update(
        self,
        problem_type,
        epsilon,
        num_iters,
        primal_optim,
        dual_optim,
        x,
        primal_lr,
        dual_lr,
        y,
        extrapolation,
        dual_restarts,
    ):
        # Initialize the figure
        self.fig = plt.figure(figsize=(15, 5))

        widths = [1, 1, 1, 0.1]
        grid_specs = GridSpec(2, 4, figure=self.fig, width_ratios=widths)

        self.loss_iter_axis = self.fig.add_subplot(grid_specs[0, 0])
        self.defect_iter_axis = self.fig.add_subplot(grid_specs[1, 0])
        self.xy_axis = self.fig.add_subplot(grid_specs[:, 1])
        self.loss_defect_axis = self.fig.add_subplot(grid_specs[:, 2])
        self.cax = self.fig.add_subplot(grid_specs[:, 3])

        # Reset the state of cmp and formulation. Indicate the new epsilon.
        self.reset_problem(epsilon=epsilon, problem_type=problem_type)

        # Plot the loss contours. Done once as loss does not change with sliders
        # The feasible set does change and is plotted in self.update.
        self.contour_params = self.loss_contours()

        # Plot the Pareto front.
        self.plot_pareto_front()

        # Update the filled contour indicating the feasible set (x, y) space and
        # epsilon hline (f, g) space
        self.plot_feasible_set()

        # New initialization
        params = torch.nn.Parameter(torch.tensor([[x, y]]))

        # Construct a new optimizer
        self.constrained_optimizer = self.create_optimizer(
            params=params,
            primal_optim=primal_optim,
            dual_optim=dual_optim,
            primal_lr=primal_lr,
            dual_lr=dual_lr,
            dual_restarts=dual_restarts,
            extrapolation=extrapolation,
        )

        state_history = self.train(params=params, num_iters=num_iters)
        self.update_trajectory_plots(state_history)

    def create_optimizer(
        self,
        params,
        primal_optim,
        primal_lr,
        dual_optim,
        dual_lr,
        dual_restarts,
        extrapolation,
    ):
        # Check if any optimizer has momentum and add to kwargs it if necessary
        primal_kwargs = {"lr": primal_lr}
        if primal_optim == "SGDM_0.9":
            primal_optim = "SGD"
            primal_kwargs["momentum"] = 0.9
        dual_kwargs = {"lr": dual_lr, "maximize": True}
        if dual_optim == "SGDM_0.9":
            dual_optim = "SGD"
            dual_kwargs["momentum"] = 0.9

        # Indicate if we are using extrapolation
        if extrapolation:
            primal_optim = "Extra" + primal_optim
            dual_optim = "Extra" + dual_optim

        primal_opt_class = getattr(cooper.optim, primal_optim) if extrapolation else getattr(torch.optim, primal_optim)
        primal_optimizer = primal_opt_class([params], **primal_kwargs)

        dual_opt_class = getattr(cooper.optim, dual_optim) if extrapolation else getattr(torch.optim, dual_optim)
        dual_optimizer = dual_opt_class([self.ineq_group.multiplier.weight], **dual_kwargs)

        constrained_optimizer = SimultaneousOptimizer(
            constraint_groups=self.ineq_group,
            primal_optimizers=primal_optimizer,
            dual_optimizers=dual_optimizer,
        )

        return constrained_optimizer

    def train(self, params, num_iters):
        """Train."""

        # Store CMPStates and param values throughout the optimization process
        # state_history = cooper.StateLogger(save_metrics=["loss", "ineq_defect", "ineq_multipliers"])
        state_history = {}

        for iter_num in range(num_iters):
            self.constrained_optimizer.zero_grad()
            cmp_state = self.cmp.compute_cmp_state(params)
            _ = cmp_state.populate_lagrangian()
            cmp_state.backward()
            self.constrained_optimizer.step()

            # Ensure parameters remain in the domain of the functions
            params[:, 0].data.clamp_(min=0, max=np.pi / 2)
            params[:, 1].data.clamp_(min=0, max=3)

            # Store optimization metrics at each step
            state_history[iter_num] = {
                "params": copy.deepcopy(params.data),
                "loss": copy.deepcopy(cmp_state.loss.item()),
                "ineq_defect": copy.deepcopy(cmp_state.observed_constraints[0].state.violation.item()),
                "ineq_multipliers": copy.deepcopy(cmp_state.observed_constraints[0].multiplier.weight.data),
            }

        return state_history

    def loss_contours(self):
        """Plot the loss contours."""
        # Initial contours for plot
        x_range = torch.tensor(np.linspace(0, np.pi / 2, 100))
        y_range = torch.tensor(np.linspace(0, 2.0, 100))
        grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing="ij")

        grid_params = torch.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
        all_states = self.cmp.compute_cmp_state(grid_params)
        loss_grid = all_states.loss.reshape(len(x_range), len(y_range))

        # Plot the contours
        loss_contours = self.xy_axis.contour(
            grid_x,
            grid_y,
            loss_grid,
            levels=[0.05, 0.125, 0.25, 0.5, 1, 1.5],
            alpha=1.0,
            colors="gray",
        )

        # Add styling
        self.xy_axis.clabel(loss_contours, inline=1)

        defect_grid = all_states.observed_constraints[0].state.violation.reshape(len(x_range), len(y_range))
        return (grid_x, grid_y, defect_grid)

    def plot_pareto_front(self):
        """Plot the Pareto front in the loss vs defect plane. This part is done
        once."""
        # y parametrizes distance to front. Regardless of epsilon, y=0 poses a
        # non-dominated solution. x parametrizes location on the Pareto front
        x_range = torch.tensor(np.linspace(0, np.pi / 2, 100))
        y_range = torch.tensor(100 * [1.0])
        all_states = self.cmp.compute_cmp_state(torch.stack([x_range, y_range], axis=1))
        self.pareto_front = (all_states.loss, all_states.observed_constraints[0].state.violation.squeeze())
        self.loss_defect_axis.plot(self.pareto_front[0], self.pareto_front[1], c="black", alpha=0.7)

        # Add styling
        self.loss_defect_axis.set_xlabel(r"Objective $f$")
        self.loss_defect_axis.set_ylabel(r"Constraint $g$")

    def update_trajectory_plots(self, state_history):
        blue = style_utils.COLOR_DICT["blue"]
        red = style_utils.COLOR_DICT["red"]
        green = style_utils.COLOR_DICT["green"]
        yellow = style_utils.COLOR_DICT["yellow"]

        iters, params_hist, loss_hist, multipliers_hist, violation_hist = zip(
            *[(k, v["params"], v["loss"], v["ineq_multipliers"], v["ineq_defect"]) for k, v in state_history.items()]
        )
        all_metrics = {
            "iters": iters,
            "params": params_hist,
            "loss": loss_hist,
            "ineq_defect": violation_hist,
            "ineq_multipliers": multipliers_hist,
        }
        cmap_vals = np.linspace(0, 1, len(all_metrics["loss"]))
        cmap_name = "viridis"

        # --------------------------------- Trajectory in x-y plane
        params_hist = np.stack(all_metrics["params"]).squeeze().reshape(-1, 2)

        self.xy_axis.scatter(
            params_hist[:, 0],
            params_hist[:, 1],
            c=cmap_vals,
            cmap=cmap_name,
            s=20,
            alpha=0.5,
            zorder=10,
        )
        # Add marker signaling the final iterate
        self.xy_axis.scatter(
            *params_hist[-1, :],
            marker="*",
            s=150,
            zorder=100,
            c=yellow,
        )
        self.xy_axis.set_xlabel(r"Param. $x$")
        self.xy_axis.set_ylabel(r"Param. $y$")
        self.xy_axis.set_title(r"Parameter $(x, y)$ space")
        # Constrain domain
        self.xy_axis.set_xlim(0, np.pi / 2)
        self.xy_axis.set_ylim(0, 2.0)
        self.xy_axis.grid(True)
        self.xy_axis.set_aspect(1.0 / self.xy_axis.get_data_ratio(), adjustable="box")

        # -------------------------------- Trajectory in loss-defect plane
        defects = np.stack(all_metrics["ineq_defect"]).squeeze()
        self.loss_defect_axis.scatter(all_metrics["loss"], defects, alpha=0.5, s=20, c=cmap_vals, cmap=cmap_name)
        # Add marker signaling the final iterate
        self.loss_defect_axis.scatter(all_metrics["loss"][-1], defects[-1], marker="*", s=150, zorder=10, c=yellow)
        self.loss_defect_axis.set_title(r"Loss vs. constraint $(f, g)$ space")
        self.loss_defect_axis.set_xlim(-0.1, 1.3)
        self.loss_defect_axis.set_ylim(-self.cmp.epsilon - 0.1, 1.3 - self.cmp.epsilon)
        self.loss_defect_axis.set_aspect("equal")

        # -------------------------------- Loss history
        self.loss_iter_axis.plot(all_metrics["iters"], all_metrics["loss"], c=blue, linewidth=2)
        self.loss_iter_axis.set_title(r"Objective $f$")
        self.loss_iter_axis.set_xlabel("Iteration")

        # -------------------------------- Multiplier and defect history
        self.defect_iter_axis.plot(
            all_metrics["iters"],
            defects,
            c=red,
            linewidth=2,
            label="Defect",
            zorder=10,
        )
        self.defect_iter_axis.plot(
            all_metrics["iters"],
            np.stack(all_metrics["ineq_multipliers"]).squeeze(),
            c=green,
            linewidth=2,
            label="Multiplier",
        )
        self.defect_iter_axis.set_xlabel("Iteration")
        self.defect_iter_axis.legend(ncol=2, loc="upper right", bbox_to_anchor=(0.9, 1.3))

        # -------------------------------- Colorbar
        last = len(all_metrics["loss"])

        cmap = matplotlib.cm.viridis
        norm = matplotlib.colors.Normalize(vmin=0, vmax=last)

        self.fig.tight_layout(w_pad=3.0)

        self.fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=self.cax,
            label="Iteration",
            ticks=np.arange(0, last + 1, last // 5),
            # pad=0.,
        )

    def plot_feasible_set(self):
        """Plot the feasible set."""
        # the values of g(x, y) have been computed in self.loss_contours for
        # the whole grid. The feasibility boundary changes based on the epsilon
        self.xy_axis.contourf(
            *self.contour_params,
            levels=[-10, 0],
            colors=style_utils.COLOR_DICT["blue"],
            alpha=0.1,
        )

        self.defect_iter_axis.axhline(0, c="gray", alpha=0.7, linestyle="--")

        # In loss vs defect plane, a line is drawn at the epsilon value
        neg_eps = -self.cmp.epsilon
        if self.cmp.problem_type == "Concave":
            y = torch.cat((self.pareto_front[1], torch.tensor([neg_eps])))
            x = torch.cat((self.pareto_front[0], torch.tensor([1.3])))
        else:
            y = torch.cat((torch.tensor([neg_eps]), self.pareto_front[1]))
            x = torch.cat((torch.tensor([1.3]), self.pareto_front[0]))

        self.loss_defect_axis.fill_between(
            x=x,
            y1=y,
            y2=0,
            where=y <= 0,
            step="mid",
            color=style_utils.COLOR_DICT["blue"],
            alpha=0.1,
        )
        self.loss_defect_axis.axhline(0, c="gray", alpha=0.7, linestyle="--")
