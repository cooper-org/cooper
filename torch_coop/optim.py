"""(Extrapolation) Optimizer aliases"""

import torch


def extrapolation_class(optimizer_class):
    class ExtraOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            self.old_iterate = []
            super().__init__(*args, **kwargs)

        @torch.no_grad()
        def extrapolation(self, _):
            """Performs the extrapolation step and save a copy of the current
            parameters for the update step."""
            if self.old_iterate:
                raise RuntimeError(
                    "Need to call step before calling extrapolation again."
                )
            for group in self.param_groups:
                for p in group["params"]:
                    self.old_iterate.append(p.detach().clone())

            # Move to extrapolation point
            super().step()

        @torch.no_grad()
        def step(self, closure=None):
            if len(self.old_iterate) == 0:
                raise RuntimeError("Must call extrapolation before step.")

            i = -1
            for group in self.param_groups:
                for p in group["params"]:
                    i += 1
                    # Move back to the previous point
                    p.data = self.old_iterate[i].data
            super().step()

            # Free the old parameters
            self.old_iterate.clear()

    return ExtraOptimizer


# Define aliases
SGD = torch.optim.SGD
Adam = torch.optim.Adam
Adagrad = torch.optim.Adagrad
RMSprop = torch.optim.RMSprop

# Extrapolation optimizers
ExtraSGD = extrapolation_class(SGD)
ExtraAdam = extrapolation_class(Adam)
ExtraAdagrad = extrapolation_class(Adagrad)
ExtraRMSprop = extrapolation_class(RMSprop)
