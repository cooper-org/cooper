"""(Extrapolation) Optimizer aliases"""

from .extra_optimizer import extrapolation_class
import torch

SGD = torch.optim.SGD
Adam = torch.optim.Adam
Adagrad = torch.optim.Adagrad
RMSprop = torch.optim.RMSprop

# Extrapolation optimizers
ExtraSGD = extrapolation_class(SGD)
ExtraAdam = extrapolation_class(Adam)
ExtraAdagrad = extrapolation_class(Adagrad)
ExtraRMSprop = extrapolation_class(RMSprop)
